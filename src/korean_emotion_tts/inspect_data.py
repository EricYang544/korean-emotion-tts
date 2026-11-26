"""ZIP을 풀지 않고 manifest와 모든 WAV를 검증한다. 외부 의존성 없음."""
import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import unicodedata
import wave
from zipfile import ZipFile

EMOTIONS = {"Happy": 0, "Sad": 1, "Angry": 2}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def member_path(root, relative):
    path = PurePosixPath(relative)
    require(not path.is_absolute() and '..' not in path.parts and '\\' not in relative,
            f'안전하지 않은 경로: {relative}')
    return str(root / path)


def inspect_archive(archive, sha256_file):
    archive = Path(archive)
    expected = Path(sha256_file).read_text().split()[0].lower()
    with archive.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    require(actual == expected, 'ZIP SHA-256 불일치')
    counts, seconds = Counter(), Counter()
    text_groups = defaultdict(list)
    ids, paths = set(), set()
    silent, clipped = [], []
    from array import array
    import sys

    with ZipFile(archive) as bundle:
        names = bundle.namelist()
        require(len(names) == len(set(names)), 'ZIP 중복 member')
        manifests = [n for n in names if PurePosixPath(n).name == 'manifest.jsonl']
        require(len(manifests) == 1, 'manifest.jsonl은 정확히 하나여야 함')
        root = PurePosixPath(manifests[0]).parent
        rows = [json.loads(line) for line in bundle.read(manifests[0]).decode('utf-8-sig').splitlines() if line.strip()]
        require(bool(rows), '빈 manifest')
        for row in rows:
            uid, emotion = row['id'], row['emotion']
            require(uid not in ids, f'중복 ID: {uid}')
            ids.add(uid)
            require(emotion in EMOTIONS and row['emotion_id'] == EMOTIONS[emotion], f'감정 ID 불일치: {uid}')
            require(str(row['speaker_identity']) == '0031', f'다른 화자: {uid}')
            text = ' '.join(unicodedata.normalize('NFC', row['text']).split())
            require(bool(text), f'빈 전사: {uid}')
            wav_path = member_path(root, row['audio_filepath'])
            require(wav_path not in paths, f'중복 WAV: {uid}')
            paths.add(wav_path)
            label = json.loads(bundle.read(member_path(root, row['annotation_filepath'])).decode('utf-8-sig'))
            require(label['화자정보']['Emotion'] == emotion, f'라벨 감정 불일치: {uid}')
            require(str(label['기본정보']['NumberOfSpeaker']).zfill(4) == '0031', f'라벨 화자 불일치: {uid}')
            transcription = label['전사정보']
            label_text = transcription.get('TransLabelText') or transcription.get('OrgLabelText')
            require(' '.join(unicodedata.normalize('NFC', label_text).split()) == text, f'라벨 전사 불일치: {uid}')
            with wave.open(io.BytesIO(bundle.read(wav_path)), 'rb') as wav:
                require((wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getcomptype()) == (22050, 1, 2, 'NONE'), f'WAV 규격 불일치: {uid}')
                frames = wav.getnframes()
                pcm = wav.readframes(frames)
                require(frames > 0 and len(pcm) == frames * 2, f'잘린/빈 WAV: {uid}')
            duration = frames / 22050
            require(row['sample_rate'] == 22050 and abs(duration - float(row['duration'])) <= 1 / 22050, f'길이·샘플레이트 불일치: {uid}')
            samples = array('h', pcm)
            if sys.byteorder != 'little':
                samples.byteswap()
            peak = max(abs(v) for v in samples)
            if peak == 0:
                silent.append(uid)
            if any(v >= 32767 or v <= -32768 for v in samples):
                clipped.append(uid)
            counts[emotion] += 1
            seconds[emotion] += duration
            text_groups[text].append(uid)
        require(set(counts) == set(EMOTIONS), '세 감정이 모두 필요함')
    return {'archive': archive.name, 'sha256': actual, 'records': len(ids),
            'counts': dict(counts), 'seconds': dict(seconds),
            'hours': sum(seconds.values()) / 3600, 'unique_texts': len(text_groups),
            'duplicate_text_groups': [v for v in text_groups.values() if len(v) > 1],
            'silent_ids': silent, 'full_scale_sample_ids': clipped,
            'note': 'Full-scale samples are review flags, not automatic rejection. This is source validation, not alignment or feature extraction.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--sha256-file', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    report = inspect_archive(args.archive, args.sha256_file)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
