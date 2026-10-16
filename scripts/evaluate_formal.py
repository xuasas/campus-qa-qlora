"""Cache four models' grounded predictions; test requires validation selection first."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['PYTHONUTF8']='1'
from pathlib import Path
import argparse, hashlib, json, math, subprocess, sys, time
ROOT=Path(__file__).resolve().parents[1]
CFG=ROOT/'configs'/'runtime'
EVAL=ROOT/'evaluations'/'runtime'
p=argparse.ArgumentParser();p.add_argument('--split',choices=['val','test'],required=True);p.add_argument('--model',choices=['Q0','Q4','Q8','Q16']);a=p.parse_args()
F=json.loads((CFG/'experiment_freeze.json').read_text(encoding='utf-8'))
GEN=json.loads((CFG/'generation_settings.json').read_text(encoding='utf-8'))
DATA=Path(F['data_dir'])
for name,digest in F['data_files_sha256'].items():assert hashlib.sha256((DATA/name).read_bytes()).hexdigest()==digest
assert hashlib.sha256((CFG/'generation_settings.json').read_bytes()).hexdigest()==F['generation_settings_sha256']
if a.split=='test':
    decision=json.loads((EVAL/'validation_selection.json').read_text(encoding='utf-8'))
    assert decision['selected_using_validation_only'] and decision['status']=='frozen_before_test'
    for name,digest in decision['validation_prediction_sha256'].items():assert hashlib.sha256((EVAL/'val'/name).read_bytes()).hexdigest()==digest
EVAL.mkdir(parents=True,exist_ok=True)
(ROOT/'current_evaluation.json').write_text(json.dumps({'directory':str(EVAL),'protocol':'provided_context'},ensure_ascii=False,indent=2),encoding='utf-8')
dest=EVAL/a.split;dest.mkdir(exist_ok=True)
rows=json.loads((DATA/f'{a.split}_metadata.json').read_text(encoding='utf-8'))
if a.model is None:
    for name in ['Q0','Q4','Q8','Q16']:
        output=dest/(name+'.jsonl')
        if output.exists():
            done=[json.loads(line) for line in output.read_text(encoding='utf-8').splitlines()]
            assert [q['id'] for q in done]==[q['id'] for q in rows]
            assert all(q['model']==name and q['generation_settings_sha256']==F['generation_settings_sha256'] for q in done)
            print('CACHE REUSED',a.split,name,flush=True);continue
        with (dest/(name+'_console.txt')).open('w',encoding='utf-8') as log:
            proc=subprocess.Popen([sys.executable,'-X','utf8','-u',str(Path(__file__).resolve()),'--split',a.split,'--model',name],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
            for line in proc.stdout:log.write(line);log.flush();print(line,end='',flush=True)
            code=proc.wait()
        assert code==0,(name,code)
    print('ALL FOUR COMPLETED',a.split,flush=True);sys.exit(0)

import torch
from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
from peft import PeftModel,get_peft_model_state_dict
from safetensors.torch import load_file
from llamafactory.data.template import TEMPLATES
torch.manual_seed(42)
assert torch.cuda.is_available()
out=dest/(a.model+'.jsonl');partial=dest/(a.model+'.jsonl.partial')
assert not out.exists() and not partial.exists(),'Preserve existing cached outputs or incomplete run; inspect before retry.'
tokenizer=AutoTokenizer.from_pretrained(F['model_name_or_path'],local_files_only=True)
template=TEMPLATES['qwen'];assert template.default_system==GEN['system']
assert template.get_stop_token_ids(tokenizer)==GEN['stop_token_ids']
model=AutoModelForCausalLM.from_pretrained(F['model_name_or_path'],local_files_only=True,dtype=torch.bfloat16,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16),device_map={'':0},attn_implementation='sdpa')
adapter_hash=None
if a.model!='Q0':
    directory=Path(F['formal_configs'][a.model]['output_dir']);weights=directory/'adapter_model.safetensors'
    adapter_hash=hashlib.sha256(weights.read_bytes()).hexdigest()
    model=PeftModel.from_pretrained(model,str(directory),is_trainable=False)
    assert model.peft_config['default'].r==F['formal_configs'][a.model]['rank']
    saved=load_file(str(weights),device='cpu');loaded=get_peft_model_state_dict(model)
    assert loaded.keys()==saved.keys() and all(torch.equal(saved[k],loaded[k].detach().cpu()) for k in saved)
    del saved,loaded
model.eval()
def encode(row):
    prompt,_=template.encode_oneturn(tokenizer,[{'role':'user','content':row['instruction']+'\n'+row['input']},{'role':'assistant','content':''}],system=None,tools=None)
    assert len(prompt)<512
    ids=torch.tensor([prompt],device='cuda');return ids,torch.ones_like(ids)
def generate(row):
    ids,mask=encode(row);torch.cuda.synchronize();start=time.perf_counter()
    with torch.inference_mode():result=model.generate(input_ids=ids,attention_mask=mask,do_sample=False,num_beams=1,max_new_tokens=256,eos_token_id=GEN['stop_token_ids'],pad_token_id=GEN['pad_token_id'])
    torch.cuda.synchronize();elapsed=time.perf_counter()-start;tokens=result[0,ids.shape[1]:].tolist()
    return tokenizer.decode(tokens,skip_special_tokens=True),len(tokens),elapsed,bool(tokens and tokens[-1] in GEN['stop_token_ids']),ids.shape[1]
generate(rows[0]) # One common first-question warmup, excluded from saved prediction timing.
torch.cuda.reset_peak_memory_stats();start=time.perf_counter();limit_count=0
with partial.open('x',encoding='utf-8') as file:
    for index,row in enumerate(rows,1):
        prediction,n,seconds,stopped,prompt_tokens=generate(row)
        assert prediction.strip() and math.isfinite(seconds) and seconds>0
        record={**row,'prediction':prediction,'model':a.model,'split':a.split,'generated_tokens_including_stop':n,'generate_seconds':seconds,'stopped_with_eos':stopped,'reached_token_limit':n>=256,'prompt_tokens':prompt_tokens,'adapter_sha256':adapter_hash,'generation_settings_sha256':F['generation_settings_sha256'],'reference_not_in_prompt':True}
        file.write(json.dumps(record,ensure_ascii=False)+'\n');file.flush();limit_count+=n>=256
        if index%5==0 or index==len(rows):print(f'{a.split} {a.model} {index}/{len(rows)}; limits={limit_count}',flush=True)
partial.rename(out)
summary={'model':a.model,'split':a.split,'n':len(rows),'prediction_file':str(out),'prediction_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'adapter_sha256':adapter_hash,'generation_settings_sha256':F['generation_settings_sha256'],'seconds_after_warmup':time.perf_counter()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'reached_token_limit_count':limit_count,'fresh_adapter_state_match':a.model!='Q0','references_not_used_as_inputs':True}
(dest/(a.model+'_generation_report.json')).write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False),flush=True)
