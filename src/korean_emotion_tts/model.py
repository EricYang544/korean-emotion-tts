"""Our small phoneme-level FastSpeech2: [B,N] phones -> [B,T,80] log-mel."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def positions(length, hidden, device):
    index = torch.arange(length, device=device).float().unsqueeze(1)
    scale = torch.exp(torch.arange(0, hidden, 2, device=device).float() * (-math.log(10000)/hidden))
    out = torch.zeros(length, hidden, device=device)
    out[:, 0::2] = torch.sin(index*scale); out[:, 1::2] = torch.cos(index*scale)
    return out


class FFTBlock(nn.Module):
    def __init__(self, hidden, heads, ff_hidden, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden); self.norm2 = nn.LayerNorm(hidden)
        self.conv1 = nn.Conv1d(hidden, ff_hidden, 9, padding=4)
        self.conv2 = nn.Conv1d(ff_hidden, hidden, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding):
        attention, _ = self.attention(x, x, x, key_padding_mask=padding, need_weights=False)
        x = self.norm1(x+self.dropout(attention)).masked_fill(padding[...,None], 0)
        f = self.conv1(x.transpose(1,2)).transpose(1,2)
        f = self.dropout(F.relu(f)).masked_fill(padding[...,None], 0)
        f = self.conv2(f.transpose(1,2)).transpose(1,2)
        return self.norm2(x+self.dropout(f)).masked_fill(padding[...,None], 0)


class Predictor(nn.Module):
    def __init__(self, hidden, dropout):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(hidden, hidden, 3, padding=1) for _ in range(2)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(2)])
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden, 1)

    def forward(self, x, padding):
        for conv, norm in zip(self.convs, self.norms):
            x = self.dropout(norm(F.relu(conv(x.transpose(1,2)).transpose(1,2))))
            x = x.masked_fill(padding[...,None], 0)
        return self.linear(x).squeeze(-1).masked_fill(padding, 0)


def length_regulate(x, durations):
    if (durations < 0).any() or (durations.sum(1) <= 0).any():
        raise ValueError('duration must be nonnegative with a positive utterance total')
    sequences = [torch.repeat_interleave(h, d.long(), dim=0) for h, d in zip(x, durations)]
    return nn.utils.rnn.pad_sequence(sequences, batch_first=True), durations.sum(1).long()


class EmotionFastSpeech2(nn.Module):
    def __init__(self, vocab_size, hidden=256, heads=2, layers=4, ff_hidden=1024, dropout=.1, max_frames=4400):
        super().__init__()
        if hidden % 2 or hidden % heads:
            raise ValueError('hidden must be divisible by 2 and heads')
        self.hidden, self.max_frames = hidden, max_frames
        self.phone_embedding = nn.Embedding(vocab_size, hidden, padding_idx=0)
        self.emotion_embedding = nn.Embedding(3, hidden)
        self.encoder = nn.ModuleList([FFTBlock(hidden, heads, ff_hidden, dropout) for _ in range(layers)])
        self.decoder = nn.ModuleList([FFTBlock(hidden, heads, ff_hidden, dropout) for _ in range(layers)])
        self.predictors = nn.ModuleDict({k: Predictor(hidden, dropout) for k in ['duration', 'pitch', 'voiced', 'energy']})
        self.pitch_projection = nn.Linear(2, hidden)
        self.energy_projection = nn.Linear(1, hidden)
        self.mel_projection = nn.Linear(hidden, 80)

    def forward(self, phones, emotions, targets=None, mean_emotion=False):
        padding = phones.eq(0)
        if padding.all(1).any():
            raise ValueError('empty phone input')
        x = self.phone_embedding(phones) + positions(phones.shape[1], self.hidden, phones.device)
        x = x.masked_fill(padding[...,None], 0)
        for block in self.encoder:
            x = block(x, padding)
        condition = (self.emotion_embedding.weight.mean(0).expand(len(phones), -1)
                     if mean_emotion else self.emotion_embedding(emotions))
        x = (x + condition[:,None,:]).masked_fill(padding[...,None], 0)
        predictions = {k: module(x, padding) for k, module in self.predictors.items()}
        if targets is None:
            raw = torch.expm1(predictions['duration'].float())
            if not torch.isfinite(raw).all():
                raise ValueError('nonfinite duration')
            durations = raw.round().clamp_min(0).long()
            durations = torch.where(phones > 1, durations.clamp_min(1), durations).masked_fill(padding, 0)
            if (durations.sum(1) > self.max_frames).any():
                raise ValueError('predicted duration exceeds maximum; inspect model')
            voiced = (predictions['voiced'] > 0).float()
            pitch, energy = predictions['pitch']*voiced, predictions['energy']
        else:
            durations = targets['durations'].masked_fill(padding, 0)
            pitch, voiced, energy = targets['pitch'], targets['voiced'], targets['energy']
        x = x + self.pitch_projection(torch.stack([pitch, voiced], -1)) + self.energy_projection(energy[...,None])
        x = x.masked_fill(padding[...,None], 0)
        expanded, lengths = length_regulate(x, durations)
        mel_padding = torch.arange(expanded.shape[1], device=x.device)[None,:] >= lengths[:,None]
        expanded = (expanded + positions(expanded.shape[1], self.hidden, x.device)).masked_fill(mel_padding[...,None], 0)
        for block in self.decoder:
            expanded = block(expanded, mel_padding)
        predictions.update(mel=self.mel_projection(expanded).masked_fill(mel_padding[...,None], 0),
                           mel_lengths=lengths, used_durations=durations)
        return predictions


def loss_terms(prediction, target):
    phone_mask = target['phones'].ne(0)
    active = phone_mask & target['durations'].gt(0)
    voiced = active & target['voiced'].gt(0)
    def mean(value, mask):
        return (value*mask).sum()/mask.sum().clamp_min(1)
    frame_mask = torch.arange(target['mel'].shape[1], device=phone_mask.device)[None,:] < target['mel_lengths'][:,None]
    if prediction['mel'].shape != target['mel'].shape:
        raise ValueError('prediction/target mel shape mismatch')
    return {
        'mel': mean((prediction['mel']-target['mel']).abs().mean(-1), frame_mask),
        'duration': mean((prediction['duration']-torch.log1p(target['durations'].float()))**2, phone_mask),
        'pitch': mean((prediction['pitch']-target['pitch'])**2, voiced),
        'voiced': mean(F.binary_cross_entropy_with_logits(prediction['voiced'], target['voiced'], reduction='none'), active),
        'energy': mean((prediction['energy']-target['energy'])**2, active),
    }
