"""Transparent single-GPU loop with exact batch cursor, RNG and wall-time resume."""
import csv
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from .common import *
from .dataset import SpeechDataset, collate, epoch_batches
from .model import EmotionFastSpeech2, loss_terms
from .vocoder import Vocoder
from .synthesize import synthesize


def move(batch, device):
    return {k:v.to(device) for k,v in batch.items()}


def atomic_save(path, payload):
    temporary = Path(str(path)+'.tmp')
    torch.save(payload, temporary); os.replace(temporary, path)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


@torch.inference_mode()
def validate(model, data, batch_size, device):
    model.eval(); totals = {}
    # Aggregate per-utterance means so duration/padding distribution does not bias metrics.
    by_emotion = {str(i):[] for i in range(3)}
    for i in range(len(data)):
        b = move(collate([data[i]]), device)
        values = loss_terms(model(b['phones'], b['emotions'], b), b)
        check(all(torch.isfinite(v) for v in values.values()), 'validation loss 비유한 값')
        for k,v in values.items(): totals[k] = totals.get(k,0)+v.item()
        by_emotion[str(int(b['emotions'][0]))].append(values['mel'].item())
    return {k:v/len(data) for k,v in totals.items()} | {'emotion_mel': {k:sum(v)/len(v) if v else None for k,v in by_emotion.items()}}


def run(args):
    config = read_json(args.config); cfg = config['train']; seed = cfg['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    check(device.type == 'cuda' or args.mode == 'smoke', '본 학습/과적합은 GPU에서 실행하세요.')
    if device.type == 'cuda': torch.cuda.manual_seed_all(seed)
    root, out = Path(args.features), Path(args.output)
    per = 10 if args.mode == 'smoke' else 30 if args.mode == 'overfit' else 0
    data = SpeechDataset(root, 'train', per, seed); val = SpeechDataset(root, 'val')
    vocab, lexicon = read_json(root/'vocab.json'), read_json(root/'lexicon.json')
    assets = read_json(args.assets)
    reconstruction = read_json(Path(args.reconstruction)/'reconstruction.json')
    check(reconstruction['assets'] == fingerprint(assets) and reconstruction['code'] == code_hash(), '동일 코드/보코더의 복원 검사가 필요합니다.')
    check(args.audio_reviewed, '원본/mel 복원 음성을 듣고 --audio-reviewed를 지정하세요.')
    cfg = dict(cfg)
    if args.mode != 'full':
        cfg.update(max_steps=100 if args.mode=='smoke' else 2000, max_hours=2, warmup=100,
                   save_every=50, validate_every=100, early_stop_after=10000)
    contract = {'code': code_hash(), 'features': sha256(root/'report.json'),
                'manifest': sha256(root/'manifest.jsonl'), 'stats': sha256(root/'stats.json'),
                'vocab': fingerprint(vocab), 'lexicon': fingerprint(lexicon), 'assets': fingerprint(assets),
                'model': config['model'], 'train': cfg, 'mode': args.mode, 'ids': [r['id'] for r in data.rows]}
    if args.mode == 'full':
        check(args.pilot_report and args.pilot_audio_reviewed, '과적합 pilot 보고서와 청취 확인이 필요합니다.')
        pilot = read_json(args.pilot_report)
        check(pilot['mode']=='overfit' and pilot['passed'] and pilot['code']==contract['code']
              and pilot['features']==contract['features'] and pilot['assets']==contract['assets'] and pilot['model']==contract['model'], '호환되는 pilot 통과 기록이 필요합니다.')
    if not args.resume:
        check(not out.exists(), '새 실험 폴더 필요; 재개는 --resume 사용')
        out.mkdir(parents=True)
    else:
        check((out/'last.pt').is_file(), 'last.pt 없음')
    model = EmotionFastSpeech2(len(vocab), **config['model']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], betas=(.9,.98), weight_decay=cfg['weight_decay'])
    step=epoch=cursor=bad=0; elapsed=0.; best=float('inf'); first_losses=[]; recent_losses=[]
    initial_embedding = model.emotion_embedding.weight.detach().cpu().clone()
    if args.resume:
        ckpt = torch.load(out/'last.pt', map_location='cpu', weights_only=False)
        check(ckpt['contract']==contract, '코드/데이터/설정이 달라 재개할 수 없습니다.')
        model.load_state_dict(ckpt['model']); optimizer.load_state_dict(ckpt['optimizer'])
        step, epoch, cursor, bad = [ckpt[k] for k in ['step','epoch','cursor','bad']]
        elapsed, best = ckpt['elapsed'], ckpt['best']
        first_losses, recent_losses = ckpt['first_losses'], ckpt['recent_losses']
        initial_embedding = ckpt['initial_embedding']
    write_json(out/'config.json', config); write_json(out/'contract.json', contract)
    import subprocess, platform
    try: commit = subprocess.check_output(['git','rev-parse','HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError): commit = None
    write_json(out/'environment.json', {'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,'git':commit,'device':str(device)})
    vocoder = Vocoder(args.assets, device)
    if args.resume:
        restore_rng(ckpt['rng'])
    prompts = sorted({r['normalized_text'] for r in val.rows})[:2]
    checkpoint_time = time.monotonic(); start_elapsed=elapsed
    autocast = cfg['bf16'] and device.type=='cuda' and torch.cuda.is_bf16_supported()
    def save(path):
        atomic_save(path, {'format_version':1,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
            'contract':contract,'model_config':config['model'],'vocab':vocab,'lexicon':lexicon,
            'stats':data.stats,'step':step,'epoch':epoch,'cursor':cursor,'bad':bad,'best':best,
            'elapsed':start_elapsed+time.monotonic()-checkpoint_time,'rng':rng_state(),
            'first_losses':first_losses,'recent_losses':recent_losses,'initial_embedding':initial_embedding})
    try:
        while step < cfg['max_steps'] and start_elapsed+time.monotonic()-checkpoint_time < cfg['max_hours']*3600:
            model.train(); optimizer.zero_grad(set_to_none=True); total_value=0.
            for _ in range(cfg['accumulation']):
                batches = epoch_batches(data.rows,cfg['batch_size'],seed,epoch)
                if cursor >= len(batches): epoch+=1; cursor=0; batches=epoch_batches(data.rows,cfg['batch_size'],seed,epoch)
                b = move(collate([data[i] for i in batches[cursor]]),device); cursor+=1
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast):
                    predictions = model(b['phones'], b['emotions'], b)
                    terms = loss_terms(predictions,b); loss = sum(terms.values())/cfg['accumulation']
                check(torch.isfinite(loss), '비유한 train loss')
                loss.backward(); total_value+=loss.item()
            gradient = model.emotion_embedding.weight.grad
            check(gradient is not None and torch.isfinite(gradient).all(), '감정 embedding gradient 오류')
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.0, error_if_nonfinite=True)
            rate = cfg['lr']*min((step+1)/cfg['warmup'], (cfg['warmup']/(step+1))**.5)
            for group in optimizer.param_groups: group['lr']=rate
            optimizer.step(); step+=1
            if len(first_losses)<20:first_losses.append(total_value)
            recent_losses=(recent_losses+[total_value])[-20:]
            with (out/'train.jsonl').open('a') as log:
                log.write(__import__('json').dumps({'step':step,'loss':total_value,'lr':rate,'grad_norm':float(grad_norm),
                    'seconds':start_elapsed+time.monotonic()-checkpoint_time, **{k:float(v.detach()) for k,v in terms.items()}})+'\n')
            if step%10==0: print(f'step={step} loss={total_value:.4f} lr={rate:.6f}',flush=True)
            if step%cfg['validate_every']==0:
                metrics=validate(model,val,cfg['batch_size'],device)
                write_json(out/f'validation_{step:06d}.json',metrics)
                if metrics['mel']<best: best=metrics['mel']; bad=0; save(out/'best.pt')
                else: bad+=1
                import soundfile as sf
                failures=[]
                for i,text in enumerate(prompts):
                    for emotion in EMOTIONS:
                        try:
                            y,_=synthesize(model,text,emotion,lexicon,vocab,vocoder,device)
                            sample_dir=out/'samples'/str(step); sample_dir.mkdir(parents=True,exist_ok=True)
                            sf.write(sample_dir/f'{i}_{emotion}.wav',y,SR,subtype='PCM_16')
                        except ValueError as exc: failures.append({'text':text,'emotion':emotion,'error':str(exc)})
                write_json(out/f'synthesis_errors_{step:06d}.json',failures)
            if step%cfg['save_every']==0:
                save(out/'last.pt'); save(out/f'step_{step:06d}.pt')
                regular=sorted(p for p in out.glob('step_*.pt') if int(p.stem.split('_')[1])%5000)
                for old in regular[:-2]: old.unlink()
            if step>=cfg['early_stop_after'] and bad>=cfg['patience']: break
    except KeyboardInterrupt:
        # Do not checkpoint a partially consumed accumulation group; last atomic checkpoint remains resumable.
        print('중단됨. 마지막 완료된 체크포인트에서 --resume 하세요.', flush=True)
        return
    save(out/'last.pt')
    updated=not torch.equal(initial_embedding,model.emotion_embedding.weight.detach().cpu())
    model.eval()
    b=move(collate([data[0]]),device)
    with torch.inference_mode():
        m0=model(b['phones'],torch.zeros(1,dtype=torch.long,device=device),b)['mel']
        m1=model(b['phones'],torch.ones(1,dtype=torch.long,device=device),b)['mel']
    changed=bool(torch.isfinite(m0).all() and torch.isfinite(m1).all() and not torch.allclose(m0,m1))
    dropped=sum(recent_losses)/len(recent_losses)<sum(first_losses)/len(first_losses)*.7
    passed=updated and changed and step>= (500 if args.mode=='overfit' else 100) and (dropped if args.mode=='overfit' else True)
    write_json(out/'run_report.json',{'mode':args.mode,'step':step,'passed':passed,'code':contract['code'],
        'features':contract['features'],'assets':contract['assets'],'model':contract['model'],'emotion_updated':updated,'condition_changes_mel':changed,
        'loss_dropped_30_percent':dropped,'elapsed_seconds':start_elapsed+time.monotonic()-checkpoint_time,
        'note':'Numerical checks do not establish intelligibility or emotion quality; listening required.'})
