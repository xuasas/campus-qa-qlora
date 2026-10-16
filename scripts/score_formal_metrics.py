"""Frozen Chinese ROUGE-L and corpus BLEU-4 for cached, aligned predictions."""
from pathlib import Path
import argparse, hashlib, importlib.metadata, json, math
import jieba
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
ROOT=Path(__file__).resolve().parents[1];EVAL=ROOT/'evaluations'/'runtime'
p=argparse.ArgumentParser();p.add_argument('--split',choices=['val','test'],required=True);a=p.parse_args()
class ChineseTokenizer:
    def tokenize(self,text):return [x for x in jieba.lcut(text) if x.strip()]
tok=ChineseTokenizer();rouge=rouge_scorer.RougeScorer(['rougeL'],use_stemmer=False,tokenizer=tok)
bleu=BLEU(tokenize='none',smooth_method='exp',effective_order=False)
def rl(ref,pred):return 100*rouge.score(ref,pred)['rougeL'].fmeasure
assert abs(rl('申请缓考需要提供证明','申请缓考需要提供证明')-100)<1e-8
assert rl('申请缓考需要提供证明','')==0 and rl('缓考','重修')==0
assert 0<rl('申请缓考需要提供证明','申请缓考')<100
sentence='申请 缓考 需要 提供 正式 证明 材料';assert abs(bleu.corpus_score([sentence],[[sentence]]).score-100)<1e-8
summaries=[];ids=None
for name in ['Q0','Q4','Q8','Q16']:
    path=EVAL/a.split/(name+'.jsonl');rows=[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
    if ids is None:ids=[r['id'] for r in rows]
    assert [r['id'] for r in rows]==ids and len(set(ids))==len(ids)
    per=[{'id':r['id'],'group':r['split_group_id'],'rougeL':rl(r['output'],r['prediction'])} for r in rows]
    seg=lambda s:' '.join(tok.tokenize(s))
    score=bleu.corpus_score([seg(r['prediction']) for r in rows],[[seg(r['output']) for r in rows]])
    summary={'model':name,'split':a.split,'n':len(rows),'rougeL':sum(x['rougeL'] for x in per)/len(per),'corpus_bleu4':score.score,'bleu_signature':str(bleu.get_signature()),'token_limit_count':sum(r['reached_token_limit'] for r in rows),'prediction_sha256':hashlib.sha256(path.read_bytes()).hexdigest(), 'versions':{p:importlib.metadata.version(p) for p in ['jieba','rouge-score','sacrebleu']}}
    assert all(math.isfinite(summary[k]) for k in ['rougeL','corpus_bleu4'])
    (EVAL/a.split/(name+'_metrics.json')).write_text(json.dumps({'summary':summary,'per_item':per},ensure_ascii=False,indent=2),encoding='utf-8')
    summaries.append(summary)
(EVAL/a.split/'metrics_summary.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summaries,ensure_ascii=False,indent=2))
