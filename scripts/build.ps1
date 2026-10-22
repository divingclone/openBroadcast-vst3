param(
    [string]$OpenBroadcast = (Join-Path $PSScriptRoot '../../openbroadcast'),
    [string]$Python = '',
    [switch]$Test
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$OpenBroadcast = (Resolve-Path -LiteralPath $OpenBroadcast).Path
New-Item -ItemType Directory -Force -Path '.deps','assets/runtime','assets/licenses' | Out-Null
function Check-Exit([string]$Step) { if ($LASTEXITCODE -ne 0) { throw "$Step failed ($LASTEXITCODE)" } }
function Package([string]$Name, [string]$Url, [string]$Hash) {
    $zip = Join-Path $root ".deps/$Name.zip"
    if (!(Test-Path -LiteralPath $zip) -or (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash -ne $Hash) {
        Invoke-WebRequest -Uri $Url -OutFile $zip
    }
    if ((Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash -ne $Hash) { throw "Checksum mismatch: $Name" }
    if (!(Test-Path -LiteralPath ".deps/$Name")) { Expand-Archive -LiteralPath $zip -DestinationPath ".deps/$Name" }
}
function Repository([string]$Name, [string]$Url, [string]$Tag, [string]$Commit) {
    if (!(Test-Path -LiteralPath ".deps/$Name")) { & git clone --depth 1 --branch $Tag $Url ".deps/$Name"; Check-Exit "Clone $Name" }
    $actual = & git -C ".deps/$Name" rev-parse HEAD
    Check-Exit "Verify $Name"
    if ($actual -ne $Commit) { throw "Unexpected revision in .deps/$Name; expected $Commit" }
}
Repository 'JUCE' 'https://github.com/juce-framework/JUCE.git' '8.0.12' '29396c22c93392d6738e021b83196283d6e4d850'
Repository 'kissfft' 'https://github.com/mborgerding/kissfft.git' '131.1.0' '8f47a67f595a6641c566087bf5277034be64f24d'
Package 'ort' 'https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime.directml/1.24.4/microsoft.ml.onnxruntime.directml.1.24.4.nupkg' '57E9F11B73437BEF7A309496135D4C1F96B1A8E9DDBA60013FA27BFC1D788681'
Package 'directml' 'https://api.nuget.org/v3-flatcontainer/microsoft.ai.directml/1.15.4/microsoft.ai.directml.1.15.4.nupkg' '4E7CB7DDCE8CF837A7A75DC029209B520CA0101470FCDF275C1F49736A3615B9'
Copy-Item -LiteralPath '.deps/directml/bin/x64-win/DirectML.dll' -Destination '.deps/ort/runtimes/win-x64/native/DirectML.dll'
Copy-Item -Path '.deps/ort/runtimes/win-x64/native/*.dll' -Destination 'assets/runtime/'
Copy-Item -LiteralPath '.deps/JUCE/LICENSE.md' -Destination 'assets/licenses/JUCE.md'
Copy-Item -LiteralPath '.deps/kissfft/LICENSES/BSD-3-Clause' -Destination 'assets/licenses/KissFFT.txt'
Copy-Item -LiteralPath '.deps/ort/LICENSE' -Destination 'assets/licenses/ONNXRuntime.txt'
Copy-Item -LiteralPath '.deps/ort/ThirdPartyNotices.txt' -Destination 'assets/licenses/ONNXRuntime-ThirdPartyNotices.txt'
Copy-Item -LiteralPath '.deps/directml/LICENSE.txt' -Destination 'assets/licenses/DirectML.txt'
Copy-Item -LiteralPath '.deps/directml/LICENSE-CODE.txt' -Destination 'assets/licenses/DirectML-Code.txt'
if (!$Python) {
    $localPython = Join-Path $OpenBroadcast '.venv/Scripts/python.exe'
    if (Test-Path -LiteralPath $localPython) { $Python = $localPython } else { $Python = (Get-Command python -ErrorAction Stop).Source }
}
& $Python scripts/prepare_assets.py --source $OpenBroadcast
Check-Exit 'Prepare assets (requires NumPy)'
$cmake = Get-Command cmake -ErrorAction SilentlyContinue
if ($cmake) { $cmake = $cmake.Source } else {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    $vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.CMake.Project -property installationPath
    if (!$vs) { throw 'Install Visual Studio C++ desktop tools, CMake, and Windows SDK.' }
    $cmake = Join-Path $vs 'Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe'
}
& $cmake -S . -B build -G 'Visual Studio 17 2022' -A x64
Check-Exit 'CMake configure'
& $cmake --build build --config Release --parallel 6
Check-Exit 'Build'
if ($Test) {
    & $Python tests/make_reference.py --source $OpenBroadcast
    Check-Exit 'Python references'
    & $Python tests/make_studio_reference.py --source $OpenBroadcast
    Check-Exit 'Studio Voice Python reference'
    & (Join-Path (Split-Path $cmake) 'ctest.exe') --test-dir build -C Release --output-on-failure
    Check-Exit 'Tests'
}
New-Item -ItemType Directory -Force -Path 'dist' | Out-Null
$bundle = Join-Path $root 'build/OpenBroadcast_artefacts/Release/VST3/OpenBroadcast.vst3'
# Copy only this named bundle; unrelated user artifacts in dist remain untouched.
Copy-Item -LiteralPath $bundle -Destination 'dist/' -Recurse -Force
Copy-Item -LiteralPath 'README.md' -Destination 'dist/'
# Remove only obsolete files from earlier releases; retain bundled licenses.
foreach ($obsolete in @('install.ps1','THIRD_PARTY.md','VALIDATION.md')) {
    $obsoletePath = Join-Path $root "dist/$obsolete"
    if (Test-Path -LiteralPath $obsoletePath -PathType Leaf) { Remove-Item -LiteralPath $obsoletePath }
}
Compress-Archive -Path 'dist/OpenBroadcast.vst3','dist/README.md' -DestinationPath 'dist/OpenBroadcast-VST3-Windows-x64.zip' -Force
Write-Host "Built: $root/dist/OpenBroadcast.vst3"
