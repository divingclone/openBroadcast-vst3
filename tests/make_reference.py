import sys
import argparse
from pathlib import Path
import numpy as np
p = argparse.ArgumentParser()
p.add_argument('--source', type=Path, default=Path('../openbroadcast'))
a = p.parse_args()
sys.path.insert(0, str((a.source/'src').resolve()))
from openbroadcast.audio.denoiser import Denoiser
out = Path('tests/generated')
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(728)
n = 48000 * 3
t = np.arange(n) / 48000
audio = (.025 * np.sin(2*np.pi*173*t) + .01*np.sin(2*np.pi*1301*t) + rng.normal(0,.008,n)).astype(np.float32)
audio[:9600] = 0
audio[48000:96000] *= .02
audio[110000:110480] += .2
audio.tofile(out/'input.f32')
for name in ('denoiser', 'dereverb'):
    for intensity in (0., .37, 1.):
        engine = Denoiser(model_file=f'{name}_48k.onnx', intensity=intensity, threads=1)
        result = np.concatenate([engine.process(audio[i:i+480]) for i in range(0,n,480)])
        result.tofile(out/f'{name}_{intensity:.2f}.f32')
        print(name, intensity, float(np.max(np.abs(result))))
