"""Read-only source audit. Does not initialize Git, commit, upload or push."""
from pathlib import Path
import ast
import json
import subprocess
import sys
import tomllib
root=Path(__file__).resolve().parents[1]
for folder in ['src','tests','scripts']:
    for p in (root/folder).rglob('*.py'):
        ast.parse(p.read_text(),filename=str(p))
for p in (root/'configs').glob('*.json'):json.loads(p.read_text())
with (root/'pyproject.toml').open('rb') as f:tomllib.load(f)
for p in (root/'src').rglob('*.py'):
    text=p.read_text()
    if 'teacher_emotion_0031' in text or '/content/drive' in text:
        raise SystemExit(f'Legacy coupling found: {p}')
print('Source syntax, JSON/TOML and independent project paths: OK')
if (root/'.git').exists():
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=root).decode().split('\0')
    for name in filter(None,tracked):
        p=root/name
        if p.suffix.lower() in {'.zip','.wav','.pth','.pt','.ckpt','.pem','.key','.nemo','.npz','.npy'} or p.name.startswith('.env'):
            raise SystemExit(f'Private/large artifact tracked: {name}')
        if p.is_file() and p.stat().st_size>5_000_000:raise SystemExit(f'Large tracked file: {name}')
    print('Tracked artifact audit: OK')
else:
    print('Git is not initialized; tracked-file audit deferred. Nothing was uploaded.')
