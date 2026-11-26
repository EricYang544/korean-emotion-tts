"""One deterministic Korean lexicon front end for both alignment and synthesis."""
import re
import unicodedata
from .common import check


def normalize(text):
    text = unicodedata.normalize('NFC', text)
    text = ''.join(' ' if unicodedata.category(c).startswith(('P', 'Z')) else c for c in text)
    text = ' '.join(text.split())
    check(bool(text) and re.fullmatch('[가-힣 ]+', text) is not None,
          f'발음 검토 필요: 숫자·영문·기호는 한글 발음으로 명시해주세요: {text}')
    return text


def read_lexicon(path):
    """Read MFA plain/probabilistic dictionary, pick highest probability then lexical tie."""
    candidates = {}
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        fields = line.split()
        if len(fields) < 2 or fields[0].startswith('#'):
            continue
        word, rest = fields[0], fields[1:]
        probabilities = []
        while rest:
            try:
                probabilities.append(float(rest[0]))
                rest = rest[1:]
            except ValueError:
                break
        if rest and not any(p in {'spn', '<unk>', '<eps>', 'sil', 'sp'} for p in rest):
            score = probabilities[0] if probabilities else 1.0
            candidates.setdefault(word, []).append((score, tuple(rest)))
    return {w: list(sorted(v, key=lambda x: (-x[0], x[1]))[0][1]) for w, v in candidates.items()}


def phonemize(text, lexicon):
    words = normalize(text).split()
    missing = sorted(set(words) - lexicon.keys())
    check(not missing, f'OOV 단어: {missing}; 같은 MFA G2P로 발음을 준비하세요.')
    tokens = ['<pause>']
    for word in words:
        tokens.extend(lexicon[word])
        tokens.append('<pause>')
    return tokens
