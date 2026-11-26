from pathlib import Path
import torch
from .common import *
from .model import EmotionFastSpeech2
from .text import phonemize, read_lexicon
from .vocoder import Vocoder


@torch.inference_mode()
def synthesize(model, text, emotion, lexicon, vocab, vocoder, device, mean=False):
    tokens = phonemize(text, lexicon)
    missing = set(tokens)-vocab.keys()
    check(not missing, f'미학습 음소: {missing}')
    phones = torch.tensor([[vocab[t] for t in tokens]], dtype=torch.long, device=device)
    result = model(phones, torch.tensor([EMOTIONS[emotion]], device=device), mean_emotion=mean)
    mel = result['mel'][0].float().cpu().numpy()
    return vocoder(mel), mel


def load_checkpoint(path, device):
    # Only load trusted checkpoints produced by this project.
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    check(ckpt['format_version'] == 1, '지원하지 않는 체크포인트')
    model = EmotionFastSpeech2(len(ckpt['vocab']), **ckpt['model_config']).to(device)
    model.load_state_dict(ckpt['model'], strict=True); model.eval()
    return model, ckpt


def run(args):
    import soundfile as sf
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, ckpt = load_checkpoint(args.checkpoint, device)
    lexicon = dict(ckpt['lexicon'])
    if args.extra_lexicon:
        for word, phones in read_lexicon(Path(args.extra_lexicon)).items():
            lexicon.setdefault(word, phones)
    vocoder = Vocoder(args.assets, device)
    check(fingerprint(vocoder.spec) == ckpt['contract']['assets'], '학습/합성 보코더 불일치')
    y, mel = synthesize(model, args.text, args.emotion, lexicon, ckpt['vocab'], vocoder, device, args.mean_emotion)
    out = Path(args.output); check(not out.exists(), '기존 출력 보호: 새 WAV 경로 필요')
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, y, SR, subtype='PCM_16')
    write_json(out.with_suffix('.json'), {'text': args.text, 'emotion': args.emotion, 'mean_emotion': args.mean_emotion,
        'checkpoint_sha256': sha256(args.checkpoint), 'frames': len(mel), 'lexicon_sha256': fingerprint(lexicon)})
