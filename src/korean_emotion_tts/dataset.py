import random
from pathlib import Path
import numpy as np
import torch
from .common import read_rows, read_json, check, sha256


class SpeechDataset:
    def __init__(self, root, split, per_emotion=0, seed=42):
        self.root = Path(root)
        self.stats = read_json(self.root/'stats.json')
        self.rows = [r for r in read_rows(self.root/'manifest.jsonl') if r['split'] == split]
        if per_emotion:
            check(split == 'train', '파일럿 선택은 train에서만 가능')
            selected = []; rng = random.Random(seed)
            for emotion in range(3):
                rows = sorted([r for r in self.rows if r['emotion_id'] == emotion], key=lambda r:r['id'])
                check(len(rows) >= per_emotion, '파일럿 표본 부족')
                rng.shuffle(rows); selected.extend(rows[:per_emotion])
            self.rows = sorted(selected, key=lambda r:r['id'])
        check(bool(self.rows), f'빈 split: {split}')
        for row in self.rows:
            check(sha256(self.root/row['feature']) == row['feature_sha256'], f"특징 변조/손상: {row['id']}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with np.load(self.root/row['feature'], allow_pickle=False) as data:
            item = {k: torch.from_numpy(data[k].copy()) for k in data.files}
        for key in ['pitch', 'energy']:
            item[key] = (item[key]-self.stats[key]['mean'])/self.stats[key]['std']
        item['pitch'] *= item['voiced']
        item['energy'] *= item['durations'].gt(0)
        item['emotion'] = row['emotion_id']
        return item


def collate(items):
    batch = {k: torch.nn.utils.rnn.pad_sequence([x[k] for x in items], batch_first=True)
             for k in ['phones', 'durations', 'pitch', 'voiced', 'energy', 'mel']}
    batch['emotions'] = torch.tensor([x['emotion'] for x in items], dtype=torch.long)
    batch['mel_lengths'] = torch.tensor([len(x['mel']) for x in items], dtype=torch.long)
    return batch


def epoch_batches(rows, batch_size, seed, epoch):
    # Deterministic shuffled length buckets; checkpoint stores epoch + batch cursor.
    rng = random.Random(seed+epoch)
    indices = list(range(len(rows))); rng.shuffle(indices)
    batches = []
    for start in range(0, len(indices), batch_size*50):
        bucket = sorted(indices[start:start+batch_size*50], key=lambda i: rows[i]['frames'])
        batches.extend(bucket[j:j+batch_size] for j in range(0, len(bucket), batch_size))
    rng.shuffle(batches)
    return batches
