"""Extract per-utterance arrays with resumable fingerprints; fit statistics on train only."""
from pathlib import Path
import numpy as np
from .common import *
from .audio import load_audio, mel_spectrogram, frame_prosody
from .alignment import read_textgrid, intervals_to_durations


def run(args):
    source, aligned, out = map(Path, (args.source, args.alignments, args.output))
    contract = read_json(source/'contract.json')
    check(contract['oov_count'] == 0, 'OOV 미해결')
    check(read_json(aligned/'alignment_provenance.json').get('per_emotion', 0) == 0, '샘플 정렬은 경계 청취 검사용입니다. 전체 정렬 후 특징을 추출하세요.')
    check(contract == read_json(aligned/'alignment_provenance.json')['source_contract'], '정렬 source 불일치')
    for name, key in [('lexicon.json','lexicon_sha256'), ('manifest.jsonl','manifest_sha256')]:
        check(sha256(source/name) == contract[key], f'{name} 변경 감지')
    lexicon = read_json(source/'lexicon.json')
    grids = {}
    for p in aligned.rglob('*.TextGrid'):
        check(p.stem not in grids, f'중복 TextGrid: {p.stem}')
        grids[p.stem] = p
    identity = {'source': contract, 'grids': {k: sha256(v) for k, v in sorted(grids.items())}, 'code': code_hash()}
    out.mkdir(parents=True, exist_ok=True)
    if (out/'feature_contract.json').exists():
        check(read_json(out/'feature_contract.json') == identity, '캐시 조건 변경: 새 output 경로 필요')
    else:
        write_json(out/'feature_contract.json', identity)
    (out/'arrays').mkdir(exist_ok=True)
    vocab = {'<pad>': 0, '<pause>': 1}
    for phone in sorted({p for v in lexicon.values() for p in v}):
        check(phone not in vocab, '사전에 예약 음소 포함')
        vocab[phone] = len(vocab)
    accepted, rejected = [], []
    for row in read_rows(source/'manifest.jsonl'):
        uid = row['id']; dest = out/'arrays'/f'{uid}.npz'
        try:
            check(uid in grids, 'TextGrid 없음')
            if not dest.exists():
                y = load_audio(source/row['audio'])
                mel = mel_spectrogram(y)
                words, phones = read_textgrid(grids[uid])
                tokens, durations = intervals_to_durations(words, phones, row['normalized_text'], lexicon, len(mel))
                f0, rms = frame_prosody(y, len(mel))
                pitch, voiced, energy = [], [], []
                offset = 0
                for duration in durations:
                    values = f0[offset:offset+duration]; positive = values[values > 0]
                    voiced.append(float(len(positive) > 0))
                    pitch.append(float(np.log(positive).mean()) if len(positive) else 0.0)
                    energy.append(float(np.log(np.maximum(rms[offset:offset+duration], 1e-6)).mean()) if duration else 0.0)
                    offset += duration
                with dest.with_suffix('.tmp').open('wb') as f:
                    np.savez_compressed(f, phones=np.array([vocab[t] for t in tokens], dtype=np.int64),
                        durations=np.array(durations, dtype=np.int64), mel=mel.astype(np.float32),
                        pitch=np.array(pitch, np.float32), voiced=np.array(voiced, np.float32), energy=np.array(energy, np.float32))
                dest.with_suffix('.tmp').replace(dest)
            with np.load(dest, allow_pickle=False) as a:
                check(all(np.isfinite(a[k]).all() for k in a.files), '비유한 특징')
                check(a['durations'].sum() == len(a['mel']), '캐시 duration 오류')
                length = len(a['mel'])
            accepted.append(dict(row, feature=str(dest.relative_to(out)), frames=length, feature_sha256=sha256(dest)))
        except (ValueError, KeyError, OSError) as exc:
            rejected.append({'id': uid, 'emotion': row['emotion'], 'reason': str(exc)})
    write_json(out/'rejected.json', rejected)
    check(accepted, '전처리 통과 데이터 없음')
    for split in ('train', 'val', 'test'):
        check({r['emotion'] for r in accepted if r['split'] == split} == set(EMOTIONS), f'{split}에 세 감정 필요')
    p_values, e_values = [], []
    for row in accepted:
        if row['split'] != 'train':
            continue
        with np.load(out/row['feature']) as a:
            p_values.extend(a['pitch'][(a['voiced'] > 0) & (a['durations'] > 0)].tolist())
            e_values.extend(a['energy'][a['durations'] > 0].tolist())
    check(p_values and e_values, '통계 데이터 없음')
    stats = {name: {'mean': float(np.mean(values)), 'std': max(float(np.std(values)), 1e-5)} for name, values in [('pitch',p_values), ('energy',e_values)]}
    write_json(out/'stats.json', stats); write_json(out/'vocab.json', vocab); write_json(out/'lexicon.json', lexicon)
    write_rows(out/'manifest.jsonl', accepted)
    write_json(out/'report.json', {'accepted': len(accepted), 'rejected': len(rejected),
        'counts': {s: {e: sum(r['split']==s and r['emotion']==e for r in accepted) for e in EMOTIONS} for s in ['train','val','test']},
        'fingerprint': fingerprint(identity), 'feature_sha256': fingerprint({r['id']:r['feature_sha256'] for r in accepted})})
    print(read_json(out/'report.json'))
