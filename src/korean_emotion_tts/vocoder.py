"""Frozen official HiFi-GAN adapter. No teacher code or hidden model downloads."""
from pathlib import Path
import importlib
import json
import sys
from types import SimpleNamespace
import torch
from .common import *


def register(args):
    repo, checkpoint, config = map(lambda p: Path(p).resolve(), (args.repo, args.checkpoint, args.config))
    expected = Path(args.sha256_file).read_text().split()[0]
    check(sha256(checkpoint) == expected, '보코더 checkpoint SHA-256 불일치')
    settings = read_json(config)
    for key, expected_value in {'sampling_rate': SR, 'num_mels': NMELS, 'n_fft': NFFT, 'win_size': NFFT, 'hop_size': HOP, 'fmin': 0, 'fmax': 8000}.items():
        check(settings.get(key) == expected_value, f'보코더 설정 불일치: {key}')
    import subprocess
    remote = subprocess.check_output(['git', '-C', str(repo), 'remote', 'get-url', 'origin'], text=True).strip()
    check(remote.rstrip('/').removesuffix('.git') in {'https://github.com/jik876/hifi-gan', 'git@github.com:jik876/hifi-gan'}, '공식 HiFi-GAN 저장소 필요')
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    check(not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip(), '공식 보코더 소스 변경 감지')
    files = {str(p): sha256(p) for p in [checkpoint, config, repo/'models.py', repo/'utils.py', repo/'env.py']}
    write_json(args.output, {'repo': str(repo), 'checkpoint': str(checkpoint), 'config': str(config),
                           'files': files, 'revision': revision, 'source': remote, 'model': 'UNIVERSAL_V1'})


class Vocoder:
    def __init__(self, assets, device):
        spec = read_json(assets)
        for path, digest in spec['files'].items():
            check(sha256(path) == digest, f'보코더 asset 변경: {path}')
        self.spec, self.device = spec, device
        check('models' not in sys.modules and 'utils' not in sys.modules, '보코더 모듈 이름 충돌')
        sys.path.insert(0, spec['repo'])
        try:
            Generator = importlib.import_module('models').Generator
            self.model = Generator(SimpleNamespace(**read_json(spec['config'])))
        finally:
            sys.path.pop(0)
            for name in ['models', 'utils', 'env']:
                sys.modules.pop(name, None)
        checkpoint = torch.load(spec['checkpoint'], map_location='cpu', weights_only=True)
        self.model.load_state_dict(checkpoint['generator'], strict=True)
        self.model.remove_weight_norm()
        self.model.to(device).eval().requires_grad_(False)

    @torch.inference_mode()
    def __call__(self, mel):
        value = torch.as_tensor(mel, dtype=torch.float32, device=self.device)
        check(value.ndim == 2 and value.shape[1] == NMELS and torch.isfinite(value).all(), '잘못된 mel')
        y = self.model(value.T.unsqueeze(0)).flatten().float().cpu().numpy()
        import numpy as np
        check(len(y) == len(mel)*HOP and np.isfinite(y).all(), '보코더 출력 오류')
        return y


def reconstruct(args):
    import soundfile as sf
    from .audio import load_audio, mel_spectrogram
    out = Path(args.output); check(not out.exists(), '새 output 경로 필요'); out.mkdir(parents=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocoder = Vocoder(args.assets, device)
    records = []
    for i, path in enumerate(args.wav):
        y = load_audio(path); mel = mel_spectrogram(y); restored = vocoder(mel)
        sf.write(out/f'{i:02d}_original.wav', y, SR, subtype='PCM_16')
        sf.write(out/f'{i:02d}_restored.wav', restored, SR, subtype='PCM_16')
        records.append({'source': str(path), 'sha256': sha256(path), 'frames': len(mel)})
    write_json(out/'reconstruction.json', {'files': records, 'assets': fingerprint(vocoder.spec), 'code': code_hash(), 'listening_required': True})
