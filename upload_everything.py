import os, glob, json, time
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER","0")
from huggingface_hub import HfApi
api=HfApi(); U='abuz8studios'
LOG=open(r'E:\ABU\02_PROJECTS\AHMAD-1\upload2.log','a',buffering=1)
def L(*a): print(time.strftime('%H:%M:%S'),*a,file=LOG)
T=r'E:\ABU\02_PROJECTS\AHMAD-1\train'
jobs=[]  # (repo, local, remote)
def add(repo,local,remote):
    if os.path.isfile(local): jobs.append((repo,local,remote))
W=f'{U}/SUPERBRAIN-1'; LAB=f'{U}/SUPERBRAIN-1-lab'
w=os.path.realpath(os.path.join(T,'soup_g2f_run6'))
add(W,glob.glob(os.path.join(w,'gguf','*q5_k_m.gguf'))[0],'SUPERBRAIN-1-soup_g2f_run6-q5_k_m.gguf')
add(W,os.path.join(w,'verdict.json'),'verdict_soup_g2f_run6.json')
runs=['soup_g2f_run6','run6_math_grpo','run4_a8_quran','g2f_rl','g2f_math','g2f_sft','run7_a10','run7b_a10_gentle','soup50','rl3f','rl1f','g4_morph','g4_rl','g4_ta','g4_sft1','g4_math']
for r in runs:
    p=os.path.realpath(os.path.join(T,r))
    add(LAB,os.path.join(p,'verdict.json'),f'runs/{r}/verdict.json')
    for f in glob.glob(os.path.join(p,'lora','*')):
        if os.path.isfile(f): add(LAB,f,f'runs/{r}/lora/{os.path.basename(f)}')
for r in runs:
    p=os.path.realpath(os.path.join(T,r))
    for g in glob.glob(os.path.join(p,'gguf','*q5_k_m.gguf')): add(LAB,g,f'runs/{r}/{r}-q5_k_m.gguf')
M=f'{U}/MUTHALLATH-1-m0'; md=r'D:\ABU\02_PROJECTS\MUTHALLATH-1'
for f in [md+r'\LEDGER.md',md+r'\m0\ablation.log',md+r'\m0\birth.log']: add(M,f,os.path.basename(f))
for c in ['muthallath0_final','jabarr_finisher','jabarr_tri_ON','jabarr_tri_OFF']:
    for f in glob.glob(md+rf'\m0\checkpoints\{c}\*'):
        if os.path.isfile(f): add(M,f,f'checkpoints/{c}/{os.path.basename(f)}')
F=f'{U}/FAIE-proofs'
for f in glob.glob(r'D:\ABU\NEOCORTEX\*_proof.json'): add(F,f,os.path.basename(f))
Q=f'{U}/Qadir-v1-v2'; h=r'F:\ABUZ8\ollama_qadir_hibernation_20260502_155035'
for f in glob.glob(h+r'\modelfiles\*.Modelfile'): add(Q,f,'modelfiles/'+os.path.basename(f))
big=sorted([f for f in glob.glob(h+r'\ollama_blobs\*') if os.path.getsize(f)>1e9])
names=['qadir_v1_c_bf16.gguf','qadir_v2_humble_bf16.gguf']
for i,f in enumerate(big[:2]): add(Q,f,'blob_'+os.path.basename(f)[:20]+'.gguf')
json.dump([[a,b,c,os.path.getsize(b)] for a,b,c in jobs],open(r'E:\ABU\02_PROJECTS\AHMAD-1\HF_UPLOAD_QUEUE.json','w'),indent=0)
for repo in sorted({j[0] for j in jobs}): api.create_repo(repo,exist_ok=True)
L('queue',len(jobs),'files',round(sum(os.path.getsize(j[1]) for j in jobs)/1e9,1),'GB')
done=[]
for repo,loc,rem in jobs:
    try:
        ex={f.path:getattr(f,'size',None) for f in api.list_repo_tree(repo,recursive=True)}
        if ex.get(rem)==os.path.getsize(loc): L('skip',repo,rem); done.append(rem); continue
        api.upload_file(path_or_fileobj=loc,path_in_repo=rem,repo_id=repo); L('OK',repo,rem); done.append(rem)
    except Exception as e: L('FAIL',repo,rem,repr(e)[:200])
L('ALL DONE',len(done),'/',len(jobs))
