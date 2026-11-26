"""Runtime tests. Missing dependencies are explicit skips locally, not claimed passes."""
import importlib.util
import unittest
AVAILABLE=all(importlib.util.find_spec(x) is not None for x in ['torch','numpy','librosa','soundfile'])


@unittest.skipUnless(AVAILABLE,'Requires training environment: torch/numpy/librosa/soundfile')
class NumericalTests(unittest.TestCase):
    def setUp(self):
        import torch
        from korean_emotion_tts.model import EmotionFastSpeech2
        torch.manual_seed(42)
        self.model=EmotionFastSpeech2(8,hidden=16,heads=2,layers=1,ff_hidden=32,dropout=0)
        self.batch={'phones':torch.tensor([[1,2,3,1],[1,4,1,0]]),
          'emotions':torch.tensor([0,2]),'durations':torch.tensor([[0,2,2,1],[1,2,0,0]]),
          'pitch':torch.zeros(2,4),'energy':torch.zeros(2,4),'voiced':torch.tensor([[0.,1,1,0],[0,1,0,0]]),
          'mel':torch.randn(2,5,80),'mel_lengths':torch.tensor([5,3])}

    def test_condition_gradient_and_influence(self):
        import torch
        from korean_emotion_tts.model import loss_terms
        b=self.batch; result=self.model(b['phones'],b['emotions'],b)
        sum(loss_terms(result,b).values()).backward()
        self.assertGreater(float(self.model.emotion_embedding.weight.grad.abs().sum()),0)
        other=self.model(b['phones'],torch.tensor([1,1]),b)
        self.assertFalse(torch.allclose(result['mel'],other['mel']))

    def test_padding_does_not_change_loss(self):
        import torch
        from korean_emotion_tts.model import loss_terms
        b=self.batch; result=self.model(b['phones'],b['emotions'],b)
        before=loss_terms(result,b)
        b['mel'][1,3:]=999
        b['pitch'][1,3]=999; b['energy'][1,3]=999
        after=loss_terms(result,b)
        for k in before:self.assertTrue(torch.allclose(before[k],after[k]),k)

    def test_length_regulator(self):
        import torch
        from korean_emotion_tts.model import length_regulate
        x=torch.tensor([[[1.],[2.],[3.]]]); out,length=length_regulate(x,torch.tensor([[2,0,1]]))
        self.assertEqual(out.flatten().tolist(),[1.,1.,3.])
        self.assertEqual(length.item(),3)
        with self.assertRaises(ValueError):length_regulate(x,torch.zeros(1,3,dtype=torch.long))

    def test_checkpoint_resume_next_update(self):
        import tempfile
        from pathlib import Path
        import torch
        from korean_emotion_tts.train import atomic_save,rng_state,restore_rng
        from korean_emotion_tts.model import loss_terms
        b=self.batch; optimizer=torch.optim.AdamW(self.model.parameters(),lr=.001)
        def step():
            optimizer.zero_grad(); sum(loss_terms(self.model(b['phones'],b['emotions'],b),b).values()).backward();optimizer.step()
        step()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'checkpoint.pt'
            atomic_save(p,{'model':self.model.state_dict(),'optimizer':optimizer.state_dict(),'rng':rng_state()})
            step(); expected={k:v.clone() for k,v in self.model.state_dict().items()}
            state=torch.load(p,weights_only=False);self.model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);restore_rng(state['rng'])
            step()
            for k,v in self.model.state_dict().items():self.assertTrue(torch.equal(v,expected[k]),k)

    def test_mel_geometry_and_silence_floor(self):
        import numpy as np
        from korean_emotion_tts.audio import mel_spectrogram
        mel=mel_spectrogram(np.zeros(22050,dtype=np.float32))
        self.assertEqual(mel.shape,(22050//256,80))
        self.assertTrue(np.isfinite(mel).all())

    def test_inference_does_not_need_targets(self):
        import torch
        self.model.eval()
        with torch.no_grad():
            self.model.predictors['duration'].linear.weight.zero_()
            self.model.predictors['duration'].linear.bias.fill_(1.)
            result=self.model(self.batch['phones'],self.batch['emotions'])
        self.assertEqual(result['mel'].shape[-1],80)
        self.assertTrue(torch.isfinite(result['mel']).all())
