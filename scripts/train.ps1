# YOLO-seg egitim scripti — Windows RTX 3070
# Kullanim:
#   cd object-detection
#   .\.venv\Scripts\Activate.ps1
#   .\scripts\train.ps1 smoke          # 3 epoch test
#   .\scripts\train.ps1 damage         # tam hasar egitimi
#   .\scripts\train.ps1 parts          # tam parca egitimi
#   .\scripts\train.ps1 all            # hasar + parca + weights kopyala

param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("smoke", "damage", "parts", "all")]
    [string]$Mode
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$Yolo = Join-Path $Root ".venv\Scripts\yolo.exe"
if (-not (Test-Path $Yolo)) {
    throw "venv bulunamadi. Once: python -m venv .venv && pip install torch ... && pip install -r requirements.txt"
}

$DamageData = "datasets/cardd-seg/data.yaml"
$PartsData  = "carparts-seg.yaml"

function Invoke-DamageSmoke {
    & $Yolo train model=yolo26n-seg.pt data=$DamageData imgsz=640 device=0 batch=4 epochs=3 name=damage_smoke
}

function Invoke-PartsSmoke {
    & $Yolo train model=yolo26n-seg.pt data=$PartsData imgsz=640 device=0 batch=4 epochs=3 name=parts_smoke
}

function Invoke-DamageFull {
    & $Yolo train model=yolo26s-seg.pt data=$DamageData imgsz=640 device=0 batch=8 epochs=100 patience=15 cos_lr=True name=damage_v2
}

function Invoke-PartsFull {
    & $Yolo train model=yolo26s-seg.pt data=$PartsData imgsz=640 device=0 batch=8 epochs=100 patience=15 name=parts_v1
}

function Copy-Weights {
    $Weights = Join-Path $Root "weights"
    New-Item -ItemType Directory -Force -Path $Weights | Out-Null
    Copy-Item (Join-Path $Root "runs\segment\damage_v2\weights\best.pt") (Join-Path $Weights "car-damage-seg-v2.pt") -Force
    Copy-Item (Join-Path $Root "runs\segment\parts_v1\weights\best.pt")   (Join-Path $Weights "car-parts-seg.pt") -Force
    Write-Host "Agirliklar kopyalandi: weights\car-damage-seg-v2.pt, weights\car-parts-seg.pt"
}

switch ($Mode) {
    "smoke"  { Invoke-DamageSmoke; Invoke-PartsSmoke }
    "damage" { Invoke-DamageFull }
    "parts"  { Invoke-PartsFull }
    "all"    { Invoke-DamageFull; Invoke-PartsFull; Copy-Weights }
}
