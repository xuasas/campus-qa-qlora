"""Freeze a validation-only choice; incomplete or misaligned ratings are rejected."""
from pathlib import Path
import argparse,datetime,hashlib,json,math
ROOT=Path(__file__).resolve().parents[1]
MODELS=['Q0','Q4','Q8','Q16']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review-method',required=True,choices=['human','assistant','mixed']);a=p.parse_args()
    directory=ROOT/'evaluations/runtime';output=directory/'validation_selection.json'
    if output.exists():raise FileExistsError('Do not overwrite frozen selection')
    if list((directory/'test').glob('Q*.jsonl')):raise ValueError('Select on validation before generating test predictions')
    scores_path=directory/'val/ratings.json';scores=read(scores_path)
    metrics={r['model']:r for r in read(directory/'val/metrics_summary.json')}
    expected=set();prediction_hashes={}
    for model in MODELS:
        path=directory/'val'/(model+'.jsonl');rows=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
        expected.update((r['id'],model) for r in rows);prediction_hashes[model+'.jsonl']=sha(path)
        if metrics[model]['prediction_sha256']!=sha(path) or metrics[model]['n']!=len(rows):raise ValueError('Metrics are stale')
    pairs=[(r['id'],r['model']) for r in scores]
    if len(pairs)!=len(set(pairs)) or set(pairs)!=expected:raise ValueError('Ratings must cover each prediction exactly once')
    for r in scores:
        for key,limit in [('fact_score',2),('condition_score',2),('no_fabrication_score',1)]:
            if type(r.get(key)) is not int or not 0<=r[key]<=limit:raise ValueError('Missing or invalid rating')
        if r.get('total_score')!=r['fact_score']+r['condition_score']+r['no_fabrication_score']:raise ValueError('Total score disagrees with subscores')
        if not isinstance(r.get('comment'),str) or not r['comment'].strip():raise ValueError('Record a justification for each rating')
    summary=[]
    for model in MODELS:
        rows=[r for r in scores if r['model']==model]
        if not rows:raise ValueError('Empty model ratings')
        row={'model':model,'n':len(rows),'mean_score_0_5':sum(r['total_score'] for r in rows)/len(rows),'rougeL':metrics[model]['rougeL'],'corpus_bleu4':metrics[model]['corpus_bleu4']}
        if not all(math.isfinite(row[k]) for k in ['mean_score_0_5','rougeL','corpus_bleu4']):raise ValueError('Non-finite metric')
        summary.append(row)
    best=max(r['mean_score_0_5'] for r in summary);tied=[r for r in summary if abs(r['mean_score_0_5']-best)<1e-10]
    best_rouge=max(r['rougeL'] for r in tied);eligible=[r for r in tied if best_rouge-r['rougeL']<=1.0]
    selected=min(eligible,key=lambda r:0 if r['model']=='Q0' else int(r['model'][1:]))['model']
    report={'date':datetime.date.today().isoformat(),'status':'frozen_before_test','selected_using_validation_only':True,'selected_model':selected,
        'validation_summary':summary,'review_method':a.review_method,'validation_prediction_sha256':prediction_hashes,
        'ratings_sha256':sha(scores_path),'no_test_scores_inspected':True,
        'rule':'Highest mean evidence score; tie uses ROUGE-L; among tied scores within one ROUGE-L point prefer lower rank, Q0 eligible.'}
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
