"""Local single-turn Q4 demo. Reference answers are never loaded by this app."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
from pathlib import Path
import argparse,datetime,hashlib,importlib.metadata,json,queue,sys,threading,time,traceback,uuid
ROOT=Path(__file__).resolve().parents[1]
CFG=ROOT/'configs'/'runtime'
INSTRUCTION='请只依据给出的校园资料回答问题，保留年份、学期、年级和学院条件。资料不足或相互矛盾时明确说明无法确认，不凭常识补充具体规定。'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def write_json(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def now():return datetime.datetime.now().astimezone().isoformat(timespec='seconds')
def input_text(source,context,question):return f'资料来源：{source}\n资料片段：\n{context}\n问题：{question}'
def checked_request(source,locator,context,question,example_id=None):
 source,locator,context,question=(s.strip() for s in [source,locator,context,question])
 if not context or not question:raise ValueError('请先填写资料片段和问题。')
 if not source:raise ValueError('请填写资料出处；自拟资料请标明“自拟演示资料”。')
 return {'source':source,'locator':locator,'context':context,'question':question,'example_id':example_id,'instruction':INSTRUCTION,'input':input_text(source,context,question)}
def static_check():
 f=read(CFG/'experiment_freeze.json');g=read(CFG/'generation_settings.json')
 adapter=Path(f['formal_configs']['Q4']['output_dir'])
 expected=sha(adapter/'adapter_model.safetensors')
 if sha(CFG/'generation_settings.json')!=f['generation_settings_sha256']:raise ValueError('生成配置摘要已变化。')
 if sha(Path(f['formal_configs']['Q4']['path']))!=f['formal_configs']['Q4']['sha256']:raise ValueError('Q4训练配置已变化。')
 base=Path(f['model_name_or_path'])
 for name in ['config.json','tokenizer_config.json','model.safetensors']:
  if not (base/name).is_file():raise FileNotFoundError(f'缺少本地基座文件：{base/name}')
 ac=read(adapter/'adapter_config.json')
 if (ac['r'],ac['lora_alpha'])!=(4,8):raise ValueError('演示入口要求rank=4、alpha=8的适配器。')
 examples=read(ROOT/'demo'/'examples.json')
 for item in examples:
  if set(item)&{'output','reference','answer','prediction'}:raise ValueError('演示样例不得内置参考答案或缓存回答。')
  checked_request(item['source'],item['locator'],item['context'],item['question'],item['id'])
 return f,g,adapter,expected,examples

class Session:
 def __init__(self,purpose):
  stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
  parent='demo_sessions' if purpose=='manual_demo' else 'demo_checks'
  self.path=ROOT/'reports'/parent/stamp;self.path.mkdir(parents=True)
  (self.path/'screenshots').mkdir();self.count=0
  write_json(self.path/'session.json',{'started_at':now(),'purpose':purpose,'program':str(Path(__file__).resolve()),'program_sha256':sha(Path(__file__).resolve()),'python':sys.executable,'records':'answers.jsonl','notes':'人工记录.txt','status':'started'})
  (self.path/'人工记录.txt').write_text('演示人工记录（请填写，不自动判定正确性）\n\n演示人：\n开始时间：\n结束时间：\n是否插电、是否有其他GPU程序：\n\n每条填写：记录编号；截图文件名；核对结论（正确/部分正确/错误/无法确认）；问题说明。\n示例：第1条；01_教务.png；正确；与材料中的日期、时段一致。\n\n运行报错或卡顿：\n其他备注：\n',encoding='utf-8')
 def event(self,kind,**fields):
  with (self.path/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'time':now(),'kind':kind,**fields},ensure_ascii=False)+'\n')
 def record(self,result):
  self.count+=1;result={**result,'record_number':self.count,'saved_at':now()}
  with (self.path/'answers.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
  block=f"\n===== 第{self.count}条 | {result['saved_at']} =====\n资料来源：{result['source']}\n定位：{result['locator']}\n资料片段：\n{result['context']}\n问题：{result['question']}\n模型原始回答：\n{result['prediction']}\n生成耗时：{result['generate_seconds']:.3f}秒\n生成token：{result['generated_tokens_including_stop']}\n输入token：{result['prompt_tokens']}\n触及256上限：{result['reached_token_limit']}\n"
  with (self.path/'演示记录.txt').open('a',encoding='utf-8') as f:f.write(block)
  return result
class Engine:
 def __init__(self,session):
  import torch
  from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
  from peft import PeftModel,get_peft_model_state_dict
  from safetensors.torch import load_file
  from llamafactory.data.template import TEMPLATES
  self.torch=torch;self.session=session
  f,g,adapter,self.adapter_sha,unused=static_check();self.gen=g
  if not torch.cuda.is_available():raise RuntimeError('CUDA不可用，请激活安装了CUDA版PyTorch的Python环境，并设置QLORA_PYTHON。')
  base_digest=sha(Path(f['model_name_or_path'])/'model.safetensors')
  if base_digest!=f['base_model_sha256']:raise ValueError('基座权重摘要与本次配置记录不一致。')
  torch.manual_seed(42);begin=time.perf_counter()
  self.tokenizer=AutoTokenizer.from_pretrained(f['model_name_or_path'],local_files_only=True)
  self.template=TEMPLATES['qwen']
  if self.template.default_system!=g['system'] or self.template.get_stop_token_ids(self.tokenizer)!=g['stop_token_ids']:raise ValueError('本地qwen模板与冻结生成配置不一致。')
  self.model=AutoModelForCausalLM.from_pretrained(f['model_name_or_path'],local_files_only=True,dtype=torch.bfloat16,quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16),device_map={'':0},attn_implementation='sdpa')
  self.model=PeftModel.from_pretrained(self.model,str(adapter),is_trainable=False)
  saved=load_file(str(adapter/'adapter_model.safetensors'),device='cpu');loaded=get_peft_model_state_dict(self.model)
  if saved.keys()!=loaded.keys() or not all(torch.equal(saved[k],loaded[k].detach().cpu()) for k in saved):raise ValueError('Q4保存权重与实际加载权重不一致。')
  del saved,loaded
  self.model.eval();torch.cuda.synchronize();self.load_seconds=time.perf_counter()-begin
  packages={n:importlib.metadata.version(n) for n in ['torch','transformers','peft','bitsandbytes','llamafactory']}
  meta=read(session.path/'session.json');meta.update({'status':'model_ready','ready_at':now(),'base_model':f['model_name_or_path'],'base_model_sha256':base_digest,'base_model_revision':f['model_revision'],'model':'Q4','adapter_path':str(adapter),'adapter_sha256':self.adapter_sha,'frozen_generation_sha256':f['generation_settings_sha256'],'generation_settings':g,'gpu':torch.cuda.get_device_name(0),'gpu_total_bytes':torch.cuda.get_device_properties(0).total_memory,'packages':packages,'model_load_seconds':self.load_seconds,'fresh_adapter_state_match':True,'no_reference_answers_loaded':True})
  write_json(session.path/'session.json',meta);session.event('model_ready',model_load_seconds=self.load_seconds)
 def encode(self,request):
  prompt,_=self.template.encode_oneturn(self.tokenizer,[{'role':'user','content':request['instruction']+'\n'+request['input']},{'role':'assistant','content':''}],system=None,tools=None)
  if len(prompt)>=512:raise ValueError(f'输入共{len(prompt)} token，需低于512；请缩短资料，仅保留相关原文。程序不会自动截断。')
  return prompt
 def answer(self,request):
  torch=self.torch;prompt=self.encode(request)
  ids=torch.tensor([prompt],device='cuda');mask=torch.ones_like(ids)
  torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();begin=time.perf_counter()
  with torch.inference_mode():out=self.model.generate(input_ids=ids,attention_mask=mask,do_sample=False,num_beams=1,max_new_tokens=256,eos_token_id=self.gen['stop_token_ids'],pad_token_id=self.gen['pad_token_id'])
  torch.cuda.synchronize();seconds=time.perf_counter()-begin;tokens=out[0,len(prompt):].tolist()
  prediction=self.tokenizer.decode(tokens,skip_special_tokens=True)
  if not prediction.strip():raise RuntimeError('模型返回空回答；请保留events日志检查。')
  return self.session.record({**request,'model':'Q4','adapter_sha256':self.adapter_sha,'prediction':prediction,'prompt_tokens':len(prompt),'generated_tokens_including_stop':len(tokens),'generate_seconds':seconds,'stopped_with_eos':bool(tokens and tokens[-1] in self.gen['stop_token_ids']),'reached_token_limit':len(tokens)>=256,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'timing_scope':'完整generate，不含模型加载、编码、日志保存；未warmup，首次回答标记cold','cold_first_answer':self.session.count==0,'reference_answer_used':False,'protocol':'provided_context_demo'})
def main():
 p=argparse.ArgumentParser(description='Public-copy Q4 inference checks; no bundled model weights.')
 p.add_argument('--check-only',action='store_true')
 args=p.parse_args()
 if not args.check_only:p.error('Use the Rust UI or pass --check-only.')
 f,g,adapter,digest,examples=static_check()
 print(json.dumps({'status':'passed','rank':4,'alpha':8,'examples':len(examples),'no_reference_answers':True},ensure_ascii=False))
if __name__=='__main__':main()
