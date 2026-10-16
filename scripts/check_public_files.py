"""Conservative file/content scan. Findings print locations, never secret values."""
from pathlib import Path
import argparse,hashlib,json,re,sys
ROOT=Path(__file__).resolve().parents[1]
RULES={
    'service_token':r'(?<![A-Za-z0-9])(?:hf_|ghp_|gho_|github_pat_|sk-proj-)[A-Za-z0-9_\-]{20,}',
    'private_key':r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    'email_address':r'[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}',
    'mobile_number':r'(?<![A-Za-z0-9])1[3-9][0-9]{9}(?![A-Za-z0-9])',
    'identity_number':r'(?<![A-Za-z0-9])[1-9][0-9]{16}[0-9Xx](?![A-Za-z0-9])',
    'personal_home':r'(?:[A-Za-z]:[\\/]Users[\\/]|/ho' r'me/|/Us' r'ers/)[^\s\x22\x27]+',
    'credential_in_url':r'https?://[^\s/]+:[^\s/]+@',
}
FORBIDDEN_DIRS={'models','runs','reports','evaluations','sources','paper','node_modules','target','__pycache__','.venv','venv','.cache','.idea','.vscode'}
FORBIDDEN_SUFFIXES={'.doc','.docx','.xls','.xlsx','.ppt','.pptx','.pdf','.exe','.dll','.safetensors','.gguf','.ckpt','.bin','.pt','.pth','.pem','.key','.pfx','.p12','.zip','.7z','.log','.pyc','.pyo'}
def scan(root,verify_snapshot=False):
    findings=[];files=[]
    for p in root.rglob('*'):
        rel=p.relative_to(root)
        if '.git' in rel.parts:continue
        if p.is_symlink():findings.append({'file':rel.as_posix(),'rule':'symbolic_link'});continue
        if not p.is_file():continue
        files.append(p)
        if (any(x in FORBIDDEN_DIRS for x in rel.parts)
                or rel.parts[:2]==('configs','runtime')
                or (len(rel.parts)>1 and rel.parts[0]=='data' and rel.parts[1]!='synthetic')):
            findings.append({'file':rel.as_posix(),'rule':'private_or_generated_directory'})
        if p.suffix.lower() in FORBIDDEN_SUFFIXES or p.name.startswith('.env') or p.name.startswith('current_'):
            findings.append({'file':rel.as_posix(),'rule':'private_or_binary_file'})
        if p.stat().st_size>5*1024*1024:findings.append({'file':rel.as_posix(),'rule':'large_file'})
        try:text=p.read_text(encoding='utf-8')
        except UnicodeDecodeError:findings.append({'file':rel.as_posix(),'rule':'non_text_file'});continue
        for kind,pattern in RULES.items():
            for match in re.finditer(pattern,text):findings.append({'file':rel.as_posix(),'line':text.count('\n',0,match.start())+1,'rule':kind})
    if verify_snapshot:
        snapshot=root/'PUBLIC_FILES.sha256'
        if not snapshot.exists():findings.append({'file':'PUBLIC_FILES.sha256','rule':'missing_snapshot'})
        else:
            expected={name:digest for line in snapshot.read_text(encoding='utf-8').splitlines() for digest,name in [line.split('  ',1)]}
            actual={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p!=snapshot}
            for name in sorted(set(expected)|set(actual)):
                if actual.get(name)!=expected.get(name):findings.append({'file':name,'rule':'snapshot_mismatch'})
    return {'status':'passed' if not findings else 'failed','scanned_files':len(files),'findings':findings,'scope':'Rules detect listed patterns; not a guarantee for future files or all possible personal information.'}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--verify-snapshot',action='store_true');a=p.parse_args()
    report=scan(ROOT,a.verify_snapshot);print(json.dumps(report,ensure_ascii=False,indent=2));return 0 if report['status']=='passed' else 1
if __name__=='__main__':sys.exit(main())
