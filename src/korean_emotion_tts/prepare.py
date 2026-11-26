"""Create an immutable source corpus; explicit OOV review before MFA."""
from collections import defaultdict, Counter
from pathlib import Path, PurePosixPath
import random
from zipfile import ZipFile
from .common import *
from .inspect_data import inspect_archive, member_path
from .text import normalize, read_lexicon


def split_rows(rows, seed=42):
    groups = defaultdict(list)
    for row in rows:
        groups[row['normalized_text']].append(row)
    buckets = defaultdict(list)
    for text, group in groups.items():
        buckets[tuple(sorted({r['emotion'] for r in group}))].append(text)
    assignments = {}
    rng = random.Random(seed)
    for signature in sorted(buckets):
        keys = sorted(buckets[signature]); rng.shuffle(keys)
        n = len(keys)
        nval = ntest = max(1, round(n * .05)) if n >= 20 else 0
        for i, key in enumerate(keys):
            assignments[key] = 'test' if i < ntest else 'val' if i < ntest+nval else 'train'
    return [dict(row, split=assignments[row['normalized_text']]) for row in rows]


def run(args):
    out = Path(args.output)
    check(not out.exists(), '출력 폴더가 이미 있습니다. 새 경로를 지정하세요.')
    report = inspect_archive(args.archive, args.sha256_file)
    lexicon = read_lexicon(Path(args.lexicon))
    if args.extra_lexicon:
        for w, phones in read_lexicon(Path(args.extra_lexicon)).items():
            lexicon.setdefault(w, phones)
    out.mkdir(parents=True)
    corpus = out / 'corpus' / '0031'; corpus.mkdir(parents=True)
    rows, rejected = [], []
    with ZipFile(args.archive) as z:
        manifest = next(n for n in z.namelist() if PurePosixPath(n).name == 'manifest.jsonl')
        root = PurePosixPath(manifest).parent
        import json
        for line in z.read(manifest).decode().splitlines():
            source = json.loads(line)
            uid = source['id']
            check(uid and all(c.isalnum() or c in '_-' for c in uid), '안전하지 않은 ID')
            try:
                normalized = normalize(source['text'])
                check(uid not in report['silent_ids'], '완전 무음')
            except ValueError as exc:
                rejected.append({'id': uid, 'reason': str(exc)}); continue
            audio = corpus / (uid + '.wav')
            audio.write_bytes(z.read(member_path(root, source['audio_filepath'])))
            (corpus / (uid + '.lab')).write_text(normalized + '\n', encoding='utf-8')
            rows.append(dict(id=uid, text=source['text'], normalized_text=normalized,
                             emotion=source['emotion'], emotion_id=source['emotion_id'],
                             audio=str(audio.relative_to(out)), duration=source['duration']))
    rows = split_rows(rows)
    words = {w for r in rows for w in r['normalized_text'].split()}
    oov = sorted(words - lexicon.keys())
    (out / 'oov.txt').write_text('\n'.join(oov) + ('\n' if oov else ''), encoding='utf-8')
    (out / 'dictionary.dict').write_text(''.join(w+'\t'+' '.join(lexicon[w])+'\n' for w in sorted(lexicon)), encoding='utf-8')
    write_json(out / 'lexicon.json', lexicon)
    write_rows(out / 'manifest.jsonl', rows)
    write_json(out / 'rejected.json', rejected)
    write_json(out / 'source_report.json', report)
    write_json(out / 'contract.json', {'archive_sha256': report['sha256'], 'seed': 42,
               'lexicon_sha256': sha256(out/'lexicon.json'), 'manifest_sha256': sha256(out/'manifest.jsonl'),
               'dictionary_sha256': sha256(out/'dictionary.dict'), 'oov_count': len(oov), 'code_sha256': code_hash()})
    print(f'{len(rows)} records; OOV {len(oov)}; rejected {len(rejected)}; {out}')
    if oov:
        print('OOV를 같은 한국어 MFA G2P로 생성한 뒤 --extra-lexicon으로 새 source 폴더를 준비하세요. 정렬은 아직 실행하지 않습니다.')


def align(args):
    import subprocess
    check(args.per_emotion >= 0 and args.jobs > 0, 'per-emotion >= 0, jobs > 0 필요')
    source = Path(args.source).resolve()
    contract = read_json(source/'contract.json')
    check(contract['oov_count'] == 0, 'OOV가 남아 있습니다.')
    check(sha256(source/'dictionary.dict') == contract['dictionary_sha256'], '사전 변경 감지')
    check(not Path(args.output).exists(), '새 alignment 출력 경로가 필요합니다.')
    # Read the installed tool's version/help first; fail rather than silently use a changed interface.
    version = subprocess.check_output([args.mfa, 'version'], text=True).strip()
    check('3.' in version, f'MFA 3.x 환경 필요: {version}')
    corpus = source/'corpus'
    if args.per_emotion:
        import shutil
        corpus = Path(str(Path(args.output).resolve())+'_sample_input')
        check(not corpus.exists(), '기존 sample corpus 보호: 새 output 필요')
        (corpus/'0031').mkdir(parents=True)
        rows = read_rows(source/'manifest.jsonl')
        for emotion in EMOTIONS:
            choices = sorted([r for r in rows if r['emotion']==emotion and r['split']=='train'], key=lambda r:r['id'])
            check(len(choices)>=args.per_emotion, '샘플 부족')
            random.Random(42).shuffle(choices)
            for row in choices[:args.per_emotion]:
                for suffix in ['.wav','.lab']:
                    shutil.copyfile(source/'corpus'/'0031'/(row['id']+suffix),corpus/'0031'/(row['id']+suffix))
    command = [args.mfa, 'align', str(corpus), str(source/'dictionary.dict'),
               args.acoustic_model, str(Path(args.output).resolve()), '--output_format', 'long_textgrid',
               '--num_jobs', str(args.jobs), '--clean', '--no_tokenization']
    subprocess.run(command, check=True)
    write_json(Path(args.output)/'alignment_provenance.json', {'command': command, 'mfa_version': version,
               'source_contract': contract, 'per_emotion': args.per_emotion, 'acoustic_sha256': sha256(args.acoustic_model)})
