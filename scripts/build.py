"""Build a native macOS/Linux VST3 ZIP (never installs the plugin).

Run: python3 scripts/build.py --test
Requires Python >= 3.12, CMake >= 3.24, a C++17 compiler, Git and curl. Linux also needs JUCE's
ALSA, X11, Xext, Xinerama, Xrandr, Xcursor, FreeType, Fontconfig and OpenGL
development packages. Use xvfb-run for tests on a headless Linux machine.
Existing assets are reused; --source prepares them from a local OpenBroadcast
checkout and requires NumPy. macOS needs Xcode command-line tools.
The Linux x64 ZIP includes isolated CUDA and MIGraphX workers sharing one set
of models. GPU acceleration requires compatible system CUDA/cuDNN or ROCm/MIGraphX libraries.
"""
import argparse
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent.parent
RUNTIMES = {
    ('Linux', 'x86_64'): ('1.24.4', 'linux-x64-gpu', 'c5f804ff5d239b436fa59e9f2fb288a39f7eb9552f6a636c8b71e792e91a8808'),
    ('Linux', 'aarch64'): ('1.24.4', 'linux-aarch64', '866109a9248d057671a039b9d725be4bd86888e3754140e6701ec621be9d4d7e'),
    # 1.23.2 is the last selected SDK with an official Intel + Apple Silicon archive.
    ('Darwin', 'arm64'): ('1.23.2', 'osx-universal2', '49ae8e3a66ccb18d98ad3fe7f5906b6d7887df8a5edd40f49eb2b14e20885809'),
    ('Darwin', 'x86_64'): ('1.23.2', 'osx-universal2', '49ae8e3a66ccb18d98ad3fe7f5906b6d7887df8a5edd40f49eb2b14e20885809'),
}
AMD_WHEEL = 'onnxruntime_migraphx-1.27.1-cp312-cp312-manylinux_2_34_x86_64.whl'
AMD_URL = 'https://files.pythonhosted.org/packages/9c/b7/7946bd996b95c859162ccd76c7e1e3f01e3627e5bf06a3ae1bf288fe89bd/' + AMD_WHEEL
AMD_SHA256 = '089c992442b6bff9f7786173ce6a31db111b104ae66be34d5467593951d905a4'

def run(*args, **kwargs):
    subprocess.run([str(x) for x in args], check=True, cwd=ROOT, **kwargs)

def repository(name, url, tag, revision):
    destination = ROOT / '.deps' / name
    if not destination.exists():
        run('git', 'clone', '--depth', '1', '--branch', tag, url, destination)
    actual = subprocess.check_output(['git', '-C', str(destination), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != revision:
        raise RuntimeError(f'Unexpected {name} revision: {actual}')

def sdk(system, arch):
    version, target, digest = RUNTIMES[system, arch]
    name = f'onnxruntime-{target}-{version}'
    archive = ROOT / '.deps' / f'{name}.tgz'
    if not archive.exists():
        partial = archive.with_suffix('.download')
        run('curl', '-fL', '--retry', '3', f'https://github.com/microsoft/onnxruntime/releases/download/v{version}/{name}.tgz', '-o', partial)
        partial.replace(archive)
    with archive.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != digest:
        raise RuntimeError(f'Checksum mismatch: {archive}')
    destination = ROOT / '.deps' / name
    if not destination.exists():
        with tarfile.open(archive) as tar:
            # Python >= 3.12 validates paths and symlinks using the data filter.
            tar.extractall(ROOT / '.deps', filter='data')
    return destination

def amd_sdk(headers):
    # Extract only native C API libraries, never the Python binding. ORT 1.27.1
    # supports the 1.24 C API used by this plugin. Keep all provider libraries
    # from the same wheel; CUDA and MIGraphX runtimes must not be mixed.
    archive = ROOT / '.deps' / AMD_WHEEL
    if not archive.exists():
        partial = archive.with_suffix('.download')
        run('curl', '-fL', '--retry', '3', AMD_URL, '-o', partial)
        partial.replace(archive)
    with archive.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != AMD_SHA256:
            raise RuntimeError(f'Checksum mismatch: {archive}')
    destination = ROOT / '.deps/onnxruntime-linux-x64-migraphx-1.27.1'
    libraries = destination / 'lib'
    libraries.mkdir(parents=True, exist_ok=True)
    shutil.copytree(headers / 'include', destination / 'include', dirs_exist_ok=True)
    with zipfile.ZipFile(archive) as wheel:
        for name in ('libonnxruntime.so.1.27.1', 'libonnxruntime_providers_shared.so',
                     'libonnxruntime_providers_migraphx.so'):
            (libraries / name).write_bytes(wheel.read('onnxruntime/capi/' + name))
        for name in ('LICENSE', 'ThirdPartyNotices.txt'):
            (destination / name).write_bytes(wheel.read('onnxruntime/' + name))
    shutil.copy2(libraries / 'libonnxruntime.so.1.27.1', libraries / 'libonnxruntime.so')
    return destination

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', type=Path, help='Local OpenBroadcast checkout for assets/reference fixtures')
    parser.add_argument('--build-dir', type=Path)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--universal', action='store_true', help='Build both Intel and Apple Silicon on macOS')
    args = parser.parse_args()
    system, arch = platform.system(), platform.machine()
    if (system, arch) not in RUNTIMES:
        parser.error('Use scripts/build.ps1 on Windows; this script supports native 64-bit macOS/Linux.')
    if args.universal and system != 'Darwin':
        parser.error('--universal requires macOS')
    (ROOT / '.deps').mkdir(exist_ok=True)
    repository('JUCE', 'https://github.com/juce-framework/JUCE.git', '8.0.12', '29396c22c93392d6738e021b83196283d6e4d850')
    repository('kissfft', 'https://github.com/mborgerding/kissfft.git', '131.1.0', '8f47a67f595a6641c566087bf5277034be64f24d')
    runtime = sdk(system, arch)
    amd = amd_sdk(runtime) if (system, arch) == ('Linux', 'x86_64') else None
    if args.source:
        run(sys.executable, 'scripts/prepare_assets.py', '--source', args.source.resolve())
    if not (ROOT / 'assets/models/studio_voice_48k.onnx').is_file():
        parser.error('Models are missing; supply --source /path/to/openbroadcast (requires NumPy).')
    licenses = ROOT / 'assets/licenses'
    licenses.mkdir(exist_ok=True)
    shutil.copy2(ROOT / '.deps/JUCE/LICENSE.md', licenses / 'JUCE.md')
    shutil.copy2(ROOT / '.deps/kissfft/LICENSES/BSD-3-Clause', licenses / 'KissFFT.txt')
    os_label = 'macOS' if system == 'Darwin' else 'Linux'
    arch_label = 'universal2' if args.universal else {'x86_64': 'x64', 'aarch64': 'arm64'}.get(arch, arch)
    build = (args.build_dir or ROOT / f'build-{os_label}-{arch_label}').resolve()
    options = ['-DCMAKE_BUILD_TYPE=Release', f'-DORT_DIR={runtime}', f'-DAMD_ORT_DIR={amd or ""}']
    if system == 'Darwin':
        options += ['-DCMAKE_OSX_DEPLOYMENT_TARGET=14.0', f'-DCMAKE_OSX_ARCHITECTURES={"arm64;x86_64" if args.universal else arch}']
    run('cmake', '-S', ROOT, '-B', build, *options)
    run('cmake', '--build', build, '--config', 'Release', '--parallel', str(args.jobs))
    bundle = build / 'OpenBroadcast_artefacts/Release/VST3/OpenBroadcast.vst3'
    # Refresh resources even when only licenses/assets changed and no relink ran.
    run('cmake', f'-DDESTINATION={bundle}/Contents/Resources', f'-DASSETS={ROOT}/assets',
        f'-DRUNTIME={runtime}/lib', f'-DORT_SDK={runtime}', f'-DPLATFORM={system}',
        f'-DCUDA_WORKER={build}/workers/cuda/ob_inference_worker' if system == 'Linux' else '-DCUDA_WORKER=',
        f'-DAMD_WORKER={build}/workers/migraphx/ob_inference_worker' if amd else '-DAMD_WORKER=',
        f'-DAMD_ORT_SDK={amd or ""}', '-P', ROOT / 'cmake/BundleResources.cmake')
    if args.test:
        if args.source:
            run(sys.executable, 'tests/make_reference.py', '--source', args.source.resolve())
            run(sys.executable, 'tests/make_studio_reference.py', '--source', args.source.resolve())
        run('ctest', '--test-dir', build, '-C', 'Release', '--output-on-failure')
    destination = ROOT / 'dist' / f'OpenBroadcast-VST3-{os_label}-{arch_label}.zip'
    destination.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='openbroadcast-package-') as staging:
        staged = Path(staging)
        shutil.copytree(bundle, staged / bundle.name, symlinks=False)
        shutil.copy2(ROOT / 'README.md', staged / 'README.md')
        if system == 'Darwin':
            # Sign AFTER flattening runtime symlinks, so ZIP extraction preserves
            # the exact files sealed in the signature. This is not notarization.
            staged_bundle = staged / bundle.name
            for library in sorted((staged_bundle / 'Contents/Resources/runtime').glob('*.dylib')):
                run('codesign', '--force', '--sign', '-', library)
            run('codesign', '--force', '--sign', '-', staged_bundle)
            run('codesign', '--verify', '--deep', '--strict', staged_bundle)
        with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for file in sorted(staged.rglob('*')):
                if file.is_file():
                    archive.write(file, file.relative_to(staged))
    print(f'Built: {destination}')

if __name__ == '__main__':
    main()
