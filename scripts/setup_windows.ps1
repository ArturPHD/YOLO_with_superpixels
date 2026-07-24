param(
    [string]$Python = "",
    [ValidateSet("auto", "cpu", "cu118", "cu121", "cu124", "cu126")]
    [string]$Torch = "auto",
    [string]$TorchVersion = "2.13.0",
    [string]$FlashAttnWheel = "",
    [string]$SpitRevision = "9ad9cb81130ddbefac894094e37699807811f8ea",
    [switch]$NoConstraints,
    [switch]$SkipTests,
    [switch]$RunSmoke,
    [switch]$RunNeckSmoke,
    [switch]$RunAuxSmoke,
    [ValidateSet("slic", "torch_slic", "gpu_slic", "spit", "spit_dense_adjacency", "spit_adjacency", "spit_voronoi", "pseudo_voronoi", "torch_grid", "gpu_grid", "torch_voronoi", "gpu_voronoi", "color_affinity", "gpu_affinity", "gradient_affinity", "texture_affinity", "color_gradient_affinity", "opencv_slic", "opencv_slico", "opencv_mslic", "opencv_lsc", "opencv_seeds")]
    [string]$SmokeAlgorithm = "torch_slic"
)

$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$VenvDir = Join-Path $Root ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$ThirdPartyDir = Join-Path $Root "third_party"
$SpitDir = Join-Path $ThirdPartyDir "SPiT"
$ConstraintsFile = Join-Path $Root "constraints-verified-windows-py310-cu126.txt"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Invoke-External {
    param(
        [string]$FilePath,
        [string[]]$Arguments
    )
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $($Arguments -join ' ')"
    }
}

function Resolve-PythonExe {
    if ($Python) {
        $resolved = & $Python -c "import sys; print(sys.executable)"
        if ($LASTEXITCODE -eq 0 -and $resolved) {
            return $resolved.Trim()
        }
        throw "Could not run Python command: $Python"
    }

    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($version in @("3.10", "3.11")) {
            $resolved = & py "-$version" -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $resolved) {
                return $resolved.Trim()
            }
        }
    }

    foreach ($command in @("python", "python3")) {
        if (Get-Command $command -ErrorAction SilentlyContinue) {
            $resolved = & $command -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $resolved) {
                return $resolved.Trim()
            }
        }
    }

    throw "Python 3.10 or 3.11 was not found. Install Python 3.10/3.11 first."
}

function Install-Torch {
    if ($FlashAttnWheel) {
        Write-Warning "flash-attn wheels are not installed by this Windows script. Use scripts/setup_linux.sh on Linux with --flash-attn-wheel."
    }

    if ($Torch -eq "auto") {
        if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
            $resolvedTorch = "cu126"
        } else {
            $resolvedTorch = "cpu"
        }
    } else {
        $resolvedTorch = $Torch
    }

    $index = if ($resolvedTorch -eq "cpu") {
        "https://download.pytorch.org/whl/cpu"
    } else {
        "https://download.pytorch.org/whl/$resolvedTorch"
    }

    $currentTorch = & $VenvPython -c "import torch; print(torch.__version__)" 2>$null
    if ($LASTEXITCODE -eq 0 -and $currentTorch) {
        $currentTorch = $currentTorch.Trim()
        $wantedTag = if ($resolvedTorch -eq "cpu") { "+cpu" } else { "+$resolvedTorch" }
        $versionOk = !$TorchVersion -or $currentTorch.StartsWith($TorchVersion)
        if ($versionOk -and $currentTorch.Contains($wantedTag)) {
            Write-Step "Using existing torch $currentTorch"
            return
        }
    }

    Write-Step "Installing torch/torchvision from $index"
    if ($TorchVersion) {
        Invoke-External $VenvPython @("-m", "pip", "install", "--upgrade", "--force-reinstall", "torch==$TorchVersion", "torchvision", "--index-url", $index)
    } else {
        Invoke-External $VenvPython @("-m", "pip", "install", "--upgrade", "--force-reinstall", "torch", "torchvision", "--index-url", $index)
    }
}

function Link-SpitCheckout {
    Write-Step "Linking local SPiT checkout"
    & $VenvPython -m pip show spit *> $null
    if ($LASTEXITCODE -eq 0) {
        Invoke-External $VenvPython @("-m", "pip", "uninstall", "-y", "spit")
    }

    $eggInfo = Join-Path $SpitDir "spit.egg-info"
    if (Test-Path $eggInfo) {
        Remove-Item -LiteralPath $eggInfo -Recurse -Force
    }

    $sitePackages = & $VenvPython -c "import site; print(next(p for p in site.getsitepackages() if p.lower().endswith('site-packages')))"
    if ($LASTEXITCODE -ne 0 -or !$sitePackages) {
        throw "Could not resolve site-packages path."
    }
    $spitPathForPth = $SpitDir -replace "\\", "/"
    $pthFile = Join-Path $sitePackages.Trim() "yolo_superpixels_spit.pth"
    Set-Content -LiteralPath $pthFile -Value $spitPathForPth -Encoding ASCII
    $venvPthFile = Join-Path $VenvDir "yolo_superpixels_spit.pth"
    Set-Content -LiteralPath $venvPthFile -Value $spitPathForPth -Encoding ASCII
    Write-Host "SPiT checkout linked via $pthFile"
}

Set-Location $Root

Write-Step "Resolving Python"
$PythonExe = Resolve-PythonExe
Invoke-External $PythonExe @("--version")

Write-Step "Creating virtual environment at $VenvDir"
Invoke-External $PythonExe @("-m", "venv", $VenvDir)

Write-Step "Upgrading pip tooling"
Invoke-External $VenvPython @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")

Install-Torch

Write-Step "Installing project dependencies"
$projectInstallArgs = @("-m", "pip", "install", "-e", ".[ultralytics,slic,opencv,dev,spit]")
if (!$NoConstraints -and (Test-Path $ConstraintsFile)) {
    $projectInstallArgs = @("-m", "pip", "install", "-c", $ConstraintsFile, "-e", ".[ultralytics,slic,opencv,dev,spit]")
}
Invoke-External $VenvPython $projectInstallArgs

Write-Step "Preparing SPiT checkout"
New-Item -ItemType Directory -Force -Path $ThirdPartyDir | Out-Null
if (!(Test-Path (Join-Path $SpitDir ".git"))) {
    if (!(Get-Command git -ErrorAction SilentlyContinue)) {
        throw "git was not found. Install Git or clone https://github.com/dsb-ifi/SPiT.git into third_party/SPiT manually."
    }
    Invoke-External "git" @("clone", "https://github.com/dsb-ifi/SPiT.git", $SpitDir)
} else {
    Write-Host "SPiT already exists at $SpitDir"
}
Invoke-External "git" @("-C", $SpitDir, "fetch", "origin", $SpitRevision)
Invoke-External "git" @("-C", $SpitDir, "checkout", "--detach", $SpitRevision)

Link-SpitCheckout

$env:YOLO_CONFIG_DIR = "$Root"

Write-Step "Checking imports"
$checkImports = @"
import sys
import torch
import torchvision
import ultralytics
import skimage
import cv2
import cupy
from spit.tokenizer.densesp import DenseSPEdgeEmbedder
from yolo_superpixels import AuxiliaryLossConfig, NeckInjectionConfig, SuperpixelAuxiliaryYOLO, SuperpixelNeckYOLO, SuperpixelYOLO
print("python", sys.version.split()[0])
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
print("torchvision", torchvision.__version__)
print("ultralytics", ultralytics.__version__)
print("skimage", skimage.__version__)
print("opencv", cv2.__version__, "ximgproc", hasattr(cv2, "ximgproc"))
print("cupy", cupy.__version__)
print("spit", DenseSPEdgeEmbedder.__name__)
print("yolo_superpixels", SuperpixelYOLO.__name__)
print("yolo_superpixels_neck", SuperpixelNeckYOLO.__name__, NeckInjectionConfig.__name__)
print("yolo_superpixels_auxiliary", SuperpixelAuxiliaryYOLO.__name__, AuxiliaryLossConfig.__name__)
"@
$checkFile = Join-Path ([System.IO.Path]::GetTempPath()) "yolo_superpixels_import_check.py"
Set-Content -LiteralPath $checkFile -Value $checkImports -Encoding UTF8
try {
    Invoke-External $VenvPython @($checkFile)
} finally {
    Remove-Item -LiteralPath $checkFile -ErrorAction SilentlyContinue
}

if (!$SkipTests) {
    Write-Step "Running tests"
    $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
    Invoke-External $VenvPython @("-B", "-m", "pytest", "-q", "-p", "no:cacheprovider")
}

if ($RunSmoke) {
    Write-Step "Running one-image YOLO12 smoke training"
    Invoke-External $VenvPython @("examples\smoke_train_one_image.py", "--algorithm", $SmokeAlgorithm, "--model", "yolo12n.pt", "--imgsz", "96")
}

if ($RunNeckSmoke) {
    Write-Step "Running one-image YOLO12 neck-injection smoke training"
    $device = if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { "0" } else { "cpu" }
    Invoke-External $VenvPython @(
        "examples\smoke_train_neck_injection.py",
        "--model", "yolo12n.pt",
        "--device", $device,
        "--imgsz", "64",
        "--fusion", "add", "concat", "cross_attention",
        "--token-provider", "region",
        "--region-algorithm", "torch_slic"
    )
}

if ($RunAuxSmoke) {
    Write-Step "Running one-image YOLO12 auxiliary-loss smoke training"
    $device = if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { "0" } else { "cpu" }
    Invoke-External $VenvPython @(
        "examples\smoke_train_auxiliary_loss.py",
        "--model", "yolo12n.pt",
        "--device", $device,
        "--imgsz", "64",
        "--algorithm", "torch_slic"
    )
}

Write-Step "Done"
Write-Host "Activate with:" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\activate"
