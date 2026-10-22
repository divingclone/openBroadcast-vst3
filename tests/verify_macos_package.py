"""Validate the delivered ZIP, then load its relocated bundle in a VST3 host."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent.parent


def run(*args):
    print('+', *map(str, args), flush=True)
    try:
        return subprocess.check_output(list(map(str, args)), text=True,
                                       stderr=subprocess.STDOUT, timeout=600)
    except subprocess.CalledProcessError as error:
        print(error.output, flush=True)
        raise


def main():
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise RuntimeError('Run this validation on a macOS Apple Silicon runner.')
    arch = 'arm64'
    build = ROOT / 'build-macOS-arm64'
    package = ROOT / 'dist/OpenBroadcast-VST3-macOS-arm64.zip'
    with package.open('rb') as stream:
        print('ZIP SHA256:', hashlib.file_digest(stream, 'sha256').hexdigest(), flush=True)
    with tempfile.TemporaryDirectory(prefix='OpenBroadcast relocated ') as temporary:
        directory = Path(temporary)
        # ditto preserves Unix permissions just like macOS Archive Utility.
        print(run('ditto', '-x', '-k', package, directory))
        bundle = directory / 'OpenBroadcast.vst3'
        resources = bundle / 'Contents/Resources'
        binary = bundle / 'Contents/MacOS/OpenBroadcast'
        if not os.access(binary, os.X_OK):
            raise RuntimeError('Extracted plugin lost executable permissions.')
        if run('lipo', '-archs', binary).strip().split() != [arch]:
            raise RuntimeError('Plugin architecture differs from the native runner.')
        libraries = sorted((resources / 'runtime').glob('*.dylib'))
        if not libraries:
            raise RuntimeError('Bundled ONNX Runtime is missing.')
        for library in libraries:
            if run('lipo', '-archs', library).split() != [arch]:
                raise RuntimeError(f'Runtime must contain only {arch}: {library}')
        for executable in [binary, *libraries]:
            dependencies = run('otool', '-L', executable)
            print(dependencies)
            for line in dependencies.splitlines():
                # Only indented lines describe dependencies, not file headers.
                if not line[:1].isspace():
                    continue
                name = line.strip().split(' (', 1)[0]
                if not name.startswith(('@', '/System/Library/', '/usr/lib/')):
                    raise RuntimeError(f'Non-portable dependency: {name}')
        print(run('codesign', '--verify', '--deep', '--strict', '--verbose=2', bundle))
        manifest = json.loads((resources / 'manifest.json').read_text())
        for name, expected in manifest.items():
            with (resources / name.replace('\\', '/')).open('rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            if actual != expected:
                raise RuntimeError(f'Asset checksum mismatch: {name}')
        with zipfile.ZipFile(package) as archive:
            models = [name for name in archive.namelist() if name.endswith('.onnx')]
            if len(models) != 3:
                raise RuntimeError('Package must contain exactly one copy of each model.')
        # These tests clear OPENBROADCAST_VST_ASSETS and discover the runtime
        # relative to the relocated plugin, exercising the manual installation.
        for options in ([], ['--studio']):
            print(run(build / 'host_smoke', bundle, *options), flush=True)
        print(run(build / 'plugin_test', resources, '--studio-device'), flush=True)
    print(f'MACOS PACKAGE PASSED: {arch}', flush=True)


if __name__ == '__main__':
    main()
