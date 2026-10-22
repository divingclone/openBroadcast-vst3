"""Test real ORT CPU fallback after simulated HIP discovery and missing MIGraphX.

This is not an AMD GPU execution test. No fake libraries enter release bundles.
"""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

executable, resources, fake_hip = map(Path, sys.argv[1:])
with tempfile.TemporaryDirectory(prefix='ob-amd-fallback-') as folder:
    root = Path(folder)
    for name in ('models', 'dsp.bin'):
        (root / name).symlink_to(resources / name, target_is_directory=(name == 'models'))
    # Copy cores/workers instead of symlinking: $ORIGIN must resolve here,
    # without either GPU provider. Both runtime versions must still run CPU.
    for backend in ('cuda', 'migraphx'):
        runtime = root / 'runtime' / backend
        runtime.mkdir(parents=True)
        for name in ('libonnxruntime.so', 'libonnxruntime_providers_shared.so', 'ob_inference_worker'):
            shutil.copy2(resources / 'runtime' / backend / name, runtime / name)
    env = dict(os.environ, OPENBROADCAST_EXPECT_BACKEND='CPU', LD_LIBRARY_PATH=str(fake_hip))
    env.pop('OB_TEST_HIP_MODE', None)
    for flag in ('--device-only', '--studio-device'):
        result = subprocess.run([str(executable), str(root), flag], env=env,
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                timeout=120)
        print(result.stdout)
        assert result.returncode == 0, 'Plugin failed instead of using CPU'
        assert 'AMD Test' in result.stdout and '(MIGraphX) initialization failed' in result.stdout
        assert 'Auto device: CPU' in result.stdout and 'DEVICE TESTS PASSED' in result.stdout
        if flag == '--studio-device':
            assert 'Studio Voice' in result.stdout
    # If the default worker itself is unavailable, CPU can use the independent
    # AMD runtime. The ZIP must not depend on just one functioning core runtime.
    (root / 'runtime/cuda/ob_inference_worker').rename(root / 'runtime/cuda/worker.disabled')
    result = subprocess.run([str(executable), str(root), '--device-only'], env=env,
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=120)
    print(result.stdout)
    assert result.returncode == 0 and 'Auto device: CPU' in result.stdout
    assert 'DEVICE TESTS PASSED' in result.stdout
print('AMD FALLBACK TESTS PASSED (simulated HIP, real CPU inference)')
