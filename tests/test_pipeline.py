import tempfile
from pathlib import Path
import unittest
from korean_emotion_tts.text import normalize, phonemize, read_lexicon
from korean_emotion_tts.prepare import split_rows
from korean_emotion_tts.alignment import intervals_to_durations


class TextAndSplitTests(unittest.TestCase):
    def test_normalization_rejects_ambiguous_text(self):
        self.assertEqual(normalize('  오늘은, 좋아요! '), '오늘은 좋아요')
        for text in ['', '오늘 12시', 'AI 공부', 'ㅋㅋㅋ']:
            with self.assertRaises(ValueError): normalize(text)

    def test_same_sentence_never_crosses_splits(self):
        rows=[dict(id=f'{e}-{i}',emotion=e,normalized_text=f'문장{i}') for e in ['Happy','Sad','Angry'] for i in range(100)]
        assigned=split_rows(rows)
        for i in range(100):
            self.assertEqual(len({r['split'] for r in assigned if r['normalized_text']==f'문장{i}'}),1)
        self.assertEqual(assigned,split_rows(rows))
        self.assertEqual({r['split'] for r in assigned},{'train','val','test'})

    def test_dictionary_probability_and_oov(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'dictionary.dict'
            path.write_text('가\t0.2\tk a\n가\t0.8\tk ɑ\n나\tn a\n')
            lex=read_lexicon(path)
            self.assertEqual(phonemize('가 나',lex),['<pause>','k','ɑ','<pause>','n','a','<pause>'])
            with self.assertRaisesRegex(ValueError,'OOV'):phonemize('다',lex)


class AlignmentTests(unittest.TestCase):
    def test_duration_partition_and_optional_pause(self):
        words=[(.1,.3,'가'),(.3,.5,'나')]
        phones=[(.1,.2,'k'),(.2,.3,'a'),(.3,.4,'n'),(.4,.5,'a')]
        tokens,d=intervals_to_durations(words,phones,'가 나',{'가':['k','a'],'나':['n','a']},60)
        self.assertEqual(sum(d),60)
        self.assertEqual(d[3],0)
        self.assertEqual(tokens[3],'<pause>')
        self.assertTrue(all(x>=0 for x in d))

    def test_wrong_phones_are_rejected_not_repaired(self):
        with self.assertRaisesRegex(ValueError,'음소 불일치'):
            intervals_to_durations([(0,.5,'가')],[(0,.5,'wrong')],'가',{'가':['k','a']},50)
