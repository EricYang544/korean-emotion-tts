import unittest
from pathlib import PurePosixPath
from korean_emotion_tts.inspect_data import member_path, require


class ValidationTests(unittest.TestCase):
    def test_reject_unsafe_paths(self):
        for path in ['../secret', '/absolute', 'wav/../../secret', 'wav\\secret']:
            with self.subTest(path=path), self.assertRaises(ValueError):
                member_path(PurePosixPath('source'), path)

    def test_relative_path(self):
        self.assertEqual(member_path(PurePosixPath('source'), 'wav/a.wav'), 'source/wav/a.wav')

    def test_validation_is_not_python_assert(self):
        with self.assertRaisesRegex(ValueError, 'invalid'):
            require(False, 'invalid')


class ArchiveTests(unittest.TestCase):
    def test_valid_archive_and_hash_rejection(self):
        import hashlib
        import io
        import json
        from pathlib import Path
        import tempfile
        import wave
        from zipfile import ZipFile
        from korean_emotion_tts.inspect_data import inspect_archive, EMOTIONS
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / 'source.zip'
            sidecar = Path(tmp) / 'source.zip.sha256'
            audio = io.BytesIO()
            with wave.open(audio, 'wb') as wav:
                wav.setparams((1, 2, 22050, 0, 'NONE', 'not compressed'))
                wav.writeframes(b'\x01\x00' * 100)
            rows = []
            with ZipFile(archive, 'w') as z:
                for emotion, index in EMOTIONS.items():
                    row = dict(id=str(index), emotion=emotion, emotion_id=index,
                               speaker_identity='0031', text='안녕하세요',
                               audio_filepath=f'wav/{index}.wav',
                               annotation_filepath=f'labels/{index}.json',
                               duration=100 / 22050, sample_rate=22050)
                    rows.append(row)
                    z.writestr(f'source/wav/{index}.wav', audio.getvalue())
                    z.writestr(f'source/labels/{index}.json', json.dumps({
                        '화자정보': {'Emotion': emotion},
                        '기본정보': {'NumberOfSpeaker': '0031'},
                        '전사정보': {'TransLabelText': '안녕하세요'}}))
                z.writestr('source/manifest.jsonl', '\n'.join(map(json.dumps, rows)))
            sidecar.write_text(hashlib.sha256(archive.read_bytes()).hexdigest())
            result = inspect_archive(archive, sidecar)
            self.assertEqual(result['records'], 3)
            self.assertEqual(len(result['duplicate_text_groups']), 1)
            self.assertEqual(result['silent_ids'], [])
            sidecar.write_text('0' * 64)
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                inspect_archive(archive, sidecar)
