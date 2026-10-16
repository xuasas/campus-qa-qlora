"""Create empty ratings for actual cached predictions; never fabricate scores."""
from pathlib import Path
import argparse,json
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--split',choices=['val','test'],required=True);a=p.parse_args()
    directory=ROOT/'evaluations/runtime'/a.split;output=directory/'ratings.json'
    if output.exists():raise FileExistsError('Do not overwrite existing ratings')
    records=[];ids=None
    for model in ['Q0','Q4','Q8','Q16']:
        rows=[json.loads(line) for line in (directory/(model+'.jsonl')).read_text(encoding='utf-8').splitlines()]
        current=[r['id'] for r in rows]
        if ids is None:ids=current
        if current!=ids or not rows:raise ValueError('Prediction rows are not aligned')
        for r in rows:
            records.append({'id':r['id'],'model':model,'fact_score':None,'condition_score':None,'no_fabrication_score':None,'total_score':None,'comment':''})
    output.write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'Created {len(records)} unscored rows. Review original context and actual predictions before filling them.')
if __name__=='__main__':main()
