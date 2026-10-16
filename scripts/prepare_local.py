"""Create ignored local configs; never train or download a model."""
from pathlib import Path
import argparse,hashlib,importlib.metadata,json,math

ROOT=Path(__file__).resolve().parents[1]
PACKAGES=['torch','transformers','peft','accelerate','bitsandbytes','llamafactory','datasets','trl']

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def validate_data(directory):
    metadata={};all_ids=set();groups=[];questions=[]
    required={'instruction','input','output','id','split_group_id','question','context','source_file','source_locator'}
    for split in ['train','val','test']:
        rows=read(directory/f'{split}_metadata.json');training=read(directory/f'campus_{split}.json')
        if not rows or len(rows)!=len(training):raise ValueError(f'{split}: empty or misaligned data')
        for row,target in zip(rows,training):
            if not required<=row.keys():raise ValueError(f'{split}: missing metadata fields')
            if row['id'] in all_ids:raise ValueError('Duplicate sample ID')
            all_ids.add(row['id'])
            if any(not isinstance(row[k],str) or not row[k].strip() for k in required):raise ValueError('Empty or invalid data field')
            if target!={k:row[k] for k in ['instruction','input','output']}:raise ValueError('Training and metadata disagree')
        metadata[split]=rows
        groups.append({r['split_group_id'] for r in rows})
        questions.append({''.join(r['question'].split()) for r in rows})
    for i in range(3):
        for j in range(i):
            if groups[i]&groups[j] or questions[i]&questions[j]:raise ValueError('Cross-split fact-family or question overlap')
    read(directory/'dataset_info.json');read(directory/'split_manifest.json')
    return metadata

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path',type=Path)
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data/synthetic')
    parser.add_argument('--check-data-only',action='store_true')
    args=parser.parse_args();directory=args.data_dir.resolve()
    metadata=validate_data(directory)
    counts={k:len(v) for k,v in metadata.items()}
    if args.check_data_only:print(json.dumps({'status':'passed','counts':counts,'training_started':False}));return
    if args.model_path is None:parser.error('--model-path is required unless --check-data-only is used')
    import yaml
    from transformers import AutoTokenizer
    from llamafactory.data.template import TEMPLATES
    model=args.model_path.resolve()
    for name in ['config.json','tokenizer_config.json','model.safetensors']:
        if not (model/name).is_file():raise FileNotFoundError(f'Missing local model file: {name}')
    config=read(model/'config.json')
    if config.get('model_type')!='qwen2':raise ValueError('This example targets the Qwen2.5 Qwen2 architecture')
    tokenizer=AutoTokenizer.from_pretrained(str(model),local_files_only=True)
    template=TEMPLATES['qwen'];lengths=[]
    for rows in metadata.values():
        for row in rows:
            prompt,answer=template.encode_oneturn(tokenizer,[{'role':'user','content':row['instruction']+'\n'+row['input']},{'role':'assistant','content':row['output']}],system=None,tools=None)
            lengths.append(len(prompt)+len(answer))
    if max(lengths)>512:raise ValueError('Full prompt+answer exceeds 512 tokens; curate data before training')
    runtime=ROOT/'configs/runtime'
    if runtime.exists():raise FileExistsError('Local configs already exist; use a separate work copy for a new experiment')
    for rank in [4,8,16]:
        if (ROOT/f'runs/runtime/Q{rank}').exists():raise FileExistsError('Training output already exists')
    templates=ROOT/'configs/templates';configs={}
    runtime.mkdir(parents=True)
    for rank in [4,8,16]:
        cfg=yaml.safe_load((templates/f'Q{rank}.yaml').read_text(encoding='utf-8'))
        cfg.update(model_name_or_path=str(model),dataset_dir=str(directory),output_dir=str(ROOT/f'runs/runtime/Q{rank}'))
        path=runtime/f'Q{rank}.yaml';path.write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding='utf-8')
        configs[f'Q{rank}']={'path':str(path),'sha256':sha(path),'rank':rank,'alpha':rank*2,'output_dir':cfg['output_dir']}
    generation=read(templates/'generation_settings.json')
    generation.update(system=template.default_system,stop_token_ids=template.get_stop_token_ids(tokenizer),pad_token_id=tokenizer.pad_token_id)
    write(runtime/'generation_settings.json',generation)
    data_hashes={p.name:sha(p) for p in directory.iterdir() if p.is_file() and p.suffix=='.json'}
    freeze={'model_name_or_path':str(model),'model_revision':'local-content-sha256','base_model_sha256':sha(model/'model.safetensors'),
        'data_dir':str(directory),'data_files_sha256':data_hashes,'formal_configs':configs,'generation_settings_sha256':sha(runtime/'generation_settings.json'),
        'packages':{name:importlib.metadata.version(name) for name in PACKAGES},'counts':counts,
        'expected_optimizer_steps_per_group':math.ceil(counts['train']/8)*2,'max_full_reference_tokens':max(lengths),
        'data_are_synthetic':all(r.get('synthetic') is True for rows in metadata.values() for r in rows),'training_started':False}
    write(runtime/'experiment_freeze.json',freeze)
    print(json.dumps({'status':'configured','counts':counts,'max_full_reference_tokens':max(lengths),'training_started':False},ensure_ascii=False))

if __name__=='__main__':main()
