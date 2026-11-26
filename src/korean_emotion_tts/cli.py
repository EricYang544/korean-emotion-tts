"""One explicit command per stage; no implicit downloads or training."""
import argparse
from importlib import import_module


def main():
    parser=argparse.ArgumentParser(prog='ketts')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare',help='원본 검증, 정규화, 분할 및 MFA corpus 생성')
    for name in ['archive','sha256-file','lexicon','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--extra-lexicon')
    p=sub.add_parser('align',help='별도 환경의 MFA 실행')
    for name in ['source','output','acoustic-model','mfa']:p.add_argument('--'+name,required=True)
    p.add_argument('--jobs',type=int,default=4)
    p.add_argument('--per-emotion',type=int,default=0)
    p=sub.add_parser('features',help='TextGrid -> duration/mel/F0/energy')
    for name in ['source','alignments','output']:p.add_argument('--'+name,required=True)
    p=sub.add_parser('register-vocoder',help='공식 보코더 asset 해시·출처 기록')
    for name in ['repo','checkpoint','config','sha256-file','output']:p.add_argument('--'+name,required=True)
    p=sub.add_parser('reconstruct',help='원본 mel -> WAV 복원 확인')
    for name in ['assets','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--wav',nargs='+',required=True)
    p=sub.add_parser('train',help='작은 검사/과적합/본 학습')
    for name in ['features','assets','reconstruction','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--config',default='configs/base.json')
    p.add_argument('--mode',choices=['smoke','overfit','full'],required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--audio-reviewed',action='store_true')
    p.add_argument('--pilot-report');p.add_argument('--pilot-audio-reviewed',action='store_true')
    p=sub.add_parser('synthesize',help='한국어 문장 + 감정 -> WAV')
    for name in ['checkpoint','assets','text','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--emotion',choices=['Happy','Sad','Angry'],required=True)
    p.add_argument('--extra-lexicon');p.add_argument('--mean-emotion',action='store_true')
    p=sub.add_parser('evaluate',help='10개 공통 문장과 test split 평가')
    for name in ['checkpoint','assets','features','texts','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--extra-lexicon')
    args=parser.parse_args()
    module,function={'prepare':('prepare','run'),'align':('prepare','align'),
        'features':('features','run'),'register-vocoder':('vocoder','register'),
        'reconstruct':('vocoder','reconstruct'),'train':('train','run'),
        'synthesize':('synthesize','run'),'evaluate':('evaluate','run')}[args.command]
    getattr(import_module('.'+module,__package__),function)(args)
