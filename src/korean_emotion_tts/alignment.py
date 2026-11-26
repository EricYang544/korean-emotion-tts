"""Map MFA intervals to a deterministic phone/pause stream without fake durations."""
from .common import SR, HOP, check
from .text import phonemize


def intervals_to_durations(words, phones, text, lexicon, frames):
    expected_words = text.split()
    words = [e for e in words if e[2].strip()]
    check([w[2] for w in words] == expected_words, 'MFA 단어 tier와 정규화 문장 불일치')
    tokens, ends = ['<pause>'], [words[0][0]]
    previous_end = 0.0
    for wi, (start, end, word) in enumerate(words):
        check(start >= previous_end - 1e-5 and end > start, '겹친/빈 단어 구간')
        entries = [p for p in phones if start <= (p[0]+p[1])/2 < end and p[2] not in {'', 'sil', 'sp'}]
        check([p[2] for p in entries] == lexicon[word], f'음소 불일치: {word}')
        check(all(p[1] > p[0] for p in entries), '빈 음소 구간')
        for i, phone in enumerate(entries):
            tokens.append(phone[2])
            boundary = (phone[1]+entries[i+1][0])/2 if i+1 < len(entries) else end
            ends.append(boundary)
        tokens.append('<pause>')
        ends.append(words[wi+1][0] if wi+1 < len(words) else frames*HOP/SR)
        previous_end = end
    check(tokens == phonemize(text, lexicon), '학습/추론 음소 불일치')
    boundaries = [0] + [max(0, min(frames, round(t*SR/HOP))) for t in ends]
    boundaries[-1] = frames
    duration = [b-a for a, b in zip(boundaries, boundaries[1:])]
    check(all(d >= 0 for d in duration) and sum(duration) == frames, 'duration 불일치')
    check(all(d > 0 for p, d in zip(tokens, duration) if p != '<pause>'), '0 frame 음소; 정렬 검토 필요')
    return tokens, duration


def read_textgrid(path):
    from praatio import textgrid
    tg = textgrid.openTextgrid(str(path), includeEmptyIntervals=True)
    def tier(suffix):
        matches = [n for n in tg.tierNames if n == suffix or n.endswith(' - '+suffix)]
        check(len(matches) == 1, f'{suffix} tier가 하나여야 함: {tg.tierNames}')
        return list(tg.getTier(matches[0]).entries)
    return tier('words'), tier('phones')
