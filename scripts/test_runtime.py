"""Run in the training environment: missing numerical dependencies is an error here."""
import importlib
import unittest
for name in ['torch','numpy','scipy','soundfile','librosa','pyworld','praatio','matplotlib']:
    importlib.import_module(name)
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover('tests'))
if not result.wasSuccessful() or result.skipped:raise SystemExit(1)
