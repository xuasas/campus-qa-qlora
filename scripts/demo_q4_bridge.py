"""UTF-8 JSON-lines transport for the Rust UI; uses the existing Q4 Engine."""
import argparse,contextlib,json,sys,traceback
from pathlib import Path
from demo_q4 import Engine,Session,checked_request,sha,static_check,write_json,read
def emit(kind,**data):
 print(json.dumps({'type':kind,**data},ensure_ascii=False),flush=True)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--purpose',default='manual_demo',choices=['manual_demo','assistant_smoke_check']);parser.add_argument('--frontend-exe',default='')
 args=parser.parse_args();session=None
 try:
  with contextlib.redirect_stdout(sys.stderr):static_check();session=Session(args.purpose)
  emit('started',session_dir=str(session.path))
  with contextlib.redirect_stdout(sys.stderr):engine=Engine(session)
  meta=read(session.path/'session.json');meta['interface']='Rust eframe 0.36.2 / Q4 Python engine';meta['bridge_sha256']=sha(Path(__file__).resolve())
  if args.frontend_exe:
   exe=Path(args.frontend_exe);meta['frontend_executable']=str(exe);meta['frontend_sha256']=sha(exe)
  write_json(session.path/'session.json',meta)
  emit('ready',model_load_seconds=engine.load_seconds,adapter_sha256=engine.adapter_sha)
 except Exception as e:
  detail=traceback.format_exc();print(detail,file=sys.stderr,flush=True)
  if session:session.event('bridge_startup_error',error=str(e),traceback=detail)
  emit('fatal',message=str(e));return 1
 for line in sys.stdin:
  try:
   message=json.loads(line)
   if message.get('action')=='shutdown':session.event('window_closed',saved_answers=session.count);emit('closed');return 0
   if message.get('action')!='generate':raise ValueError('不支持的演示指令。')
   req=checked_request(message.get('source',''),message.get('locator',''),message.get('context',''),message.get('question',''),message.get('example_id'))
   req['retrieval_mode']=message.get('retrieval_mode','manual')
   req['knowledge_id']=message.get('knowledge_id')
   session.event('source_selected',retrieval_mode=req['retrieval_mode'],knowledge_id=req['knowledge_id'],source=req['source'],locator=req['locator'])
   with contextlib.redirect_stdout(sys.stderr):result=engine.answer(req)
   emit('answer',record=result)
  except Exception as e:
   detail=traceback.format_exc();session.event('error',action='rust_generate',error=str(e),traceback=detail);print(detail,file=sys.stderr,flush=True);emit('error',message=str(e))
 session.event('frontend_disconnected',saved_answers=session.count);return 0
if __name__=='__main__':sys.exit(main())
