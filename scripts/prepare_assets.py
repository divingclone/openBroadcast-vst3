"""Copy local OpenBroadcast assets and export its NumPy coefficients losslessly."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--source', type=Path, default=Path('../openbroadcast'))
p.add_argument('--output', type=Path, default=Path('assets'))
a = p.parse_args()
source = a.source / 'src/openbroadcast/audio'
a.output.mkdir(parents=True, exist_ok=True)
(a.output / 'models').mkdir(exist_ok=True)
with np.load(source / 'data/dsp_48k.npz', allow_pickle=False) as data:
    with (a.output / 'dsp.bin').open('wb') as f:
        f.write(b'OBDSP001')
        for key in ('analysis_window', 'synthesis_window', 'mel'):
            value = np.asarray(data[key], dtype='<f4')
            f.write(struct.pack('<I', value.size))
            f.write(value.tobytes())
for name in ('denoiser_48k.onnx', 'dereverb_48k.onnx', 'studio_voice_48k.onnx'):
    shutil.copy2(source / 'models' / name, a.output / 'models' / name)
manifest = {str(f.relative_to(a.output)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in [a.output / 'dsp.bin', *sorted((a.output / 'models').glob('*.onnx'))]}
(a.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
