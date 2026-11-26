"""Small shared I/O helpers; no machine-specific paths."""
import hashlib
import json
import os
from pathlib import Path

EMOTIONS = {'Happy': 0, 'Sad': 1, 'Angry': 2}
SR, HOP, NFFT, NMELS = 22050, 256, 1024, 80


def sha256(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def code_hash():
    return fingerprint({p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))})


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temp, path)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def write_rows(path, rows):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')


def check(condition, message):
    if not condition:
        raise ValueError(message)
