"""Small real-model Studio Voice fixtures from the upstream Python engine."""
import argparse
import sys
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--source', type=Path, default=Path('../openbroadcast'))
p.add_argument('--gpu', type=int)
p.add_argument('--provider', choices=('dml', 'cuda', 'coreml'), default='dml', help='Backend used with --gpu')
a = p.parse_args()
sys.path.insert(0, str((a.source/'src').resolve()))
from openbroadcast.audio.studio import StudioVoice

out = Path('tests/generated')
out.mkdir(parents=True, exist_ok=True)
n = 3840 * 6
t = np.arange(n) / 48000
rng = np.random.default_rng(390)
audio = (.04*np.sin(2*np.pi*173*t) + .015*np.sin(2*np.pi*346*t) + rng.normal(0,.004,n)).astype(np.float32)
audio[:4800] = 0
audio[-3840:] = 0
audio.tofile(out/'studio_input.f32')
provider, device = ('cpu', 0) if a.gpu is None else (a.provider, a.gpu)
engine = StudioVoice(provider=provider, device_id=device, threads=1)
result = np.concatenate([engine.process(audio[i:i+480]) for i in range(0,n,480)])
# The scalar native interface buffers one full 3840-sample block, whereas the
# Python interface can return a hop from the block supplied in the same call.
np.concatenate([np.zeros(480,np.float32),result[:-480]]).tofile(out/f'studio_{provider}.f32')
print(provider, 'peak', float(np.max(np.abs(result))))
