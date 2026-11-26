"""Same-text synthesis, held-out losses and blind listening assets (not an emotion classifier)."""
from pathlib import Path
import csv
import random
import shutil
import numpy as np
import torch
import soundfile as sf
from .common import *
from .audio import frame_prosody
from .synthesize import load_checkpoint, synthesize
from .vocoder import Vocoder
from .text import normalize, read_lexicon
from .dataset import SpeechDataset
from .train import validate


def metrics(y):
    frames=max(1,len(y)//HOP)
    f0,_=frame_prosody(y,frames); voiced=f0[f0>0]
    return {'seconds':len(y)/SR,'rms':float(np.sqrt(np.mean(y.astype(float)**2))),
            'f0_median':float(np.median(voiced)) if len(voiced) else None,
            'f0_std':float(np.std(voiced)) if len(voiced) else None,
            'voiced_fraction':float(np.mean(f0>0)), 'clip_fraction':float(np.mean(np.abs(y)>=.999))}


def write_csv(path, rows, keys):
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def run(args):
    texts=[x.strip() for x in Path(args.texts).read_text().splitlines() if x.strip()]
    normalized=[normalize(t) for t in texts]
    check(len(texts)==10 and len(set(normalized))==10,'서로 다른 평가 문장 10개 필요')
    root=Path(args.features)
    seen={r['normalized_text'] for r in read_rows(root/'manifest.jsonl') if r['split']!='test'}
    check(not seen.intersection(normalized),'평가 문장이 train/validation에 포함됨')
    out=Path(args.output); check(not out.exists(),'새 평가 output 경로 필요'); out.mkdir(parents=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model,ckpt=load_checkpoint(args.checkpoint,device)
    check(sha256(root/'report.json')==ckpt['contract']['features'],'평가 데이터 버전 불일치')
    vocoder=Vocoder(args.assets,device)
    check(fingerprint(vocoder.spec)==ckpt['contract']['assets'],'보코더 불일치')
    lexicon=dict(ckpt['lexicon'])
    if args.extra_lexicon:
        for word,phones in read_lexicon(Path(args.extra_lexicon)).items():lexicon.setdefault(word,phones)
    all_rows=[]; blind=[]
    for i,text in enumerate(texts,1):
        for emotion in list(EMOTIONS)+['MeanEmbedding']:
            mean=emotion=='MeanEmbedding'
            y,mel=synthesize(model,text,'Happy' if mean else emotion,lexicon,ckpt['vocab'],vocoder,device,mean)
            name=f'{i:02d}_{emotion}.wav'; sf.write(out/name,y,SR,subtype='PCM_16')
            all_rows.append({'sentence':i,'text':text,'emotion':emotion,'file':name,**metrics(y)})
            if not mean:blind.append({'file':name,'sentence':i,'emotion':emotion})
    write_csv(out/'acoustic_metrics.csv',all_rows,list(all_rows[0]))
    random.Random(42).shuffle(blind); (out/'blind_audio').mkdir()
    key=[]
    for i,row in enumerate(blind,1):
        name=f'clip_{i:03d}.wav'; shutil.copyfile(out/row['file'],out/'blind_audio'/name)
        key.append({'blind_file':name,**row})
    write_json(out/'listening_key.json',key)
    write_csv(out/'listening.csv',[{'file':r['blind_file'],'perceived_emotion':'','naturalness_1_to_5':'','intelligibility_1_to_5':''} for r in key],
              ['file','perceived_emotion','naturalness_1_to_5','intelligibility_1_to_5'])
    scores=validate(model,SpeechDataset(root,'test'),1,device)
    write_json(out/'evaluation.json',{'test_teacher_forced_losses':scores,'checkpoint_sha256':sha256(args.checkpoint),
       'texts_sha256':sha256(args.texts),'metrics':all_rows,'notes':['Mean embedding is not a separately trained unconditioned baseline.',
       'Acoustic differences alone do not establish perceived emotion. Do not tune on test losses.']})
    # Simple standalone SVG; plots do not require matplotlib.
    colors={'Happy':'#e6a000','Sad':'#286bb8','Angry':'#c43b3b'}
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="500" viewBox="0 0 1000 500">',
         '<rect width="1000" height="500" fill="white"/><text x="40" y="30" font-size="20">Same-text F0 median (Hz)</text>']
    values=[r['f0_median'] or 0 for r in all_rows if r['emotion'] in EMOTIONS]; maximum=max(max(values),1)
    for row in all_rows:
        if row['emotion'] not in EMOTIONS:continue
        x=50+(row['sentence']-1)*90+EMOTIONS[row['emotion']]*22; h=(row['f0_median'] or 0)/maximum*340
        svg.append(f'<rect x="{x}" y="{400-h}" width="18" height="{h}" fill="{colors[row["emotion"]]}"/>')
    for i in range(10):svg.append(f'<text x="{65+i*90}" y="425">{i+1}</text>')
    svg.append(f'<text x="40" y="460">Happy: gold | Sad: blue | Angry: red; tallest bar = {maximum:.1f} Hz</text></svg>')
    (out/'f0_comparison.svg').write_text('\n'.join(svg))
    print(f'평가 결과: {out}; blind_audio 30개와 평균 embedding 비교 10개')
