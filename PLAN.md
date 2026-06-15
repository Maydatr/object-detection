# Object Detection — Uygulama Plani

> Windows PC (RTX 3070): gelistirme + egitim + UI testi — tek makine
> Repo: https://github.com/Maydatr/object-detection

---

## Proje Ozeti

Arac hasar tespiti PoC. Iki uygulama:

| Dosya | Amac |
|-------|------|
| `main.py` | Canli kamera/video, tek model, tikla-kilitle (v1) |
| `main_v2.py` | Coklu foto -> parca+hasar -> kaporta kontrol raporu (v2) |

v2 mimarisi: **parca segmentasyonu** + **hasar segmentasyonu** -> overlap-ratio ile panele atama -> sedan SVG semasi.

---

## Dosya Yapisi

```
object-detection/
├── main.py                 # v1 (dokunma)
├── main_v2.py              # v2 kaporta raporu
├── backends.py             # UltralyticsBackend
├── mask_intersection.py    # overlap-ratio atama
├── session.py              # VehicleSession (1 arac = 1 oturum)
├── panel_config.yaml       # parca -> sema eslemesi
├── assets/sedan_schema.svg
├── scripts/convert_cardd.py
├── datasets/
│   ├── car-damage-v6/      # v1 Roboflow dataset (data.yaml repoda)
│   └── cardd-seg/          # v2 hasar dataset (data.yaml repoda)
└── weights/
    ├── car-damage-seg.pt       # v1 (mevcut)
    ├── car-parts-seg.pt        # v2 parca modeli (egitim sonrasi)
    └── car-damage-seg-v2.pt    # v2 hasar modeli (egitim sonrasi)
```

---

## Makine Rolleri

### Windows PC (RTX 3070) — Tek ortam

- Kod, egitim, inference ve UI (`main.py` / `main_v2.py`)
- CUDA `device=0`, batch 8 (OOM olursa 4)
- Egitim bitince `best.pt` -> `weights/` altina kopyala

---

## Adim 1: Ortam (Windows)

```powershell
cd C:\Users\mayda\OneDrive\Belgeler\dev\rnd-projects\object-detection
.\.venv\Scripts\Activate.ps1
```

CUDA kontrol:
```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## Adim 2: CarDD Dataset Hazirlik (PC)

CarDD zip'i PC'ye kopyala ve ac:
```
C:\datasets\CarDD_release\CarDD_COCO\
  annotations\instances_{train,val,test}2017.json
  train2017\  val2017\  test2017\
```

`scripts/convert_cardd.py` path'leri (zaten ayarli):
```python
CARDD_ANNOTATIONS = Path(r"C:\datasets\CarDD_release\CarDD_COCO\annotations")
CARDD_IMAGES_ROOT = Path(r"C:\datasets\CarDD_release\CarDD_COCO")
```

Donusum:
```bash
python scripts/convert_cardd.py
```

`datasets/cardd-seg/data.yaml` path (zaten ayarli):
```yaml
path: C:/Users/mayda/OneDrive/Belgeler/dev/rnd-projects/object-detection/datasets/cardd-seg
```

---

## Adim 3: Model Egitimi (RTX 3070)

Script ile (onerilen):
```powershell
.\.venv\Scripts\Activate.ps1
.\scripts\train.ps1 smoke    # 3 epoch test
.\scripts\train.ps1 all      # tam egitim + weights kopyala
```

Veya manuel:

### Smoke test (3 epoch, her ikisi icin)

```bash
# Hasar modeli (CarDD)
yolo train model=yolo26n-seg.pt data=datasets/cardd-seg/data.yaml \
  imgsz=768 device=0 batch=16 epochs=3 name=damage_smoke

# Parca modeli (otomatik indirir)
yolo train model=yolo26n-seg.pt data=carparts-seg.yaml \
  imgsz=768 device=0 batch=16 epochs=3 name=parts_smoke
```

### Tam egitim

```bash
# Hasar — CarDD, 6 sinif
yolo train model=yolo26s-seg.pt data=datasets/cardd-seg/data.yaml \
  imgsz=768 device=0 batch=16 epochs=100 patience=15 cos_lr=True \
  name=damage_v2

# Parca — Carparts-Seg, 23 sinif
yolo train model=yolo26s-seg.pt data=carparts-seg.yaml \
  imgsz=768 device=0 batch=16 epochs=100 patience=15 \
  name=parts_v1
```

Yetmezse: `yolo26m-seg.pt` dene.

### Agirliklari kopyala

```bash
cp runs/segment/damage_v2/weights/best.pt weights/car-damage-seg-v2.pt
cp runs/segment/parts_v1/weights/best.pt weights/car-parts-seg.pt
```

Mac'e USB / scp / Drive ile `weights/` klasorunu tasi.

---

## Adim 4: v2 Uygulama (Windows)

```powershell
cd object-detection
.\.venv\Scripts\Activate.ps1
python main_v2.py
```

### UI akisi

1. **Yeni Arac Oturumu** — tek arac baslat
2. **Dosya / Klasor / Kamera** ile foto ekle (serbest sayi)
3. **Fotograflari Isle** — iki model + overlap-ratio
4. **Raporu Olustur** — sedan semasi renklendirilir
5. **PNG / PDF Kaydet** — opsiyonel export
6. **Oturumu Sifirla** — sonraki araca gec

### Renk kodlari

| Renk | Anlam |
|------|-------|
| Yesil `#4CAF50` | Hasar yok (panel goruldu) |
| Gri `#B0BEC5` | Veri yok (panel hic gorulmedi) |
| Turuncu `#FF9800` | Hafif: scratch, dent |
| Kirmizi `#F44336` | Agir: crack, cam, far, lastik |

---

## Adim 5: Overlap-Ratio Mantigi (referans)

```
overlap_ratio = |hasar_maskesi ∩ parca_maskesi| / |hasar_maskesi|
```

- IoU degil — kucuk hasar buyuk panelde haksiz dusmesin
- `min_ratio` varsayilan: 0.10 (UI'dan ayarlanabilir)
- Ayni panel birden fazla fotoda: en agir hasar + en yuksek guven

Severity: `tire_flat > crack > glass_shatter > dent > lamp_broken > scratch`

---

## Egitim Parametreleri Karsilastirma

| Parametre | RTX 3070 (Windows) |
|-----------|---------------------|
| device | 0 |
| batch | 8 (OOM: 4) |
| imgsz | 640 |
| epochs | 100 |
| patience | 15 |

---

## Git Notlari

Repoya **gitmeyen** seyler (`.gitignore`):
- `.venv/`, `runs/`, `weights/*.pt`
- Dataset gorselleri ve label `.txt` dosyalari
- `yolo26n-seg.pt` gibi base modeller

Repoda **olan** seyler:
- Tum Python kaynak kodu
- `data.yaml`, `panel_config.yaml`, SVG sema
- `scripts/convert_cardd.py`, `PLAN.md`

---

## Yapilacaklar Checklist

- [x] Windows ortam + CUDA torch
- [x] CarDD ac + convert_cardd.py
- [x] data.yaml path
- [ ] Smoke test (3 epoch x 2 model)
- [ ] Tam egitim (100 epoch x 2 model)
- [ ] `best.pt` -> `weights/car-parts-seg.pt` + `weights/car-damage-seg-v2.pt`
- [ ] `python main_v2.py` ile uctan uca test
- [ ] Ornek arac fotolariyla rapor PNG/PDF export dene

---

## Sorun Giderme

| Sorun | Cozum |
|-------|-------|
| `Model eksik` uyarisi | `weights/` altinda iki `.pt` oldugunu dogrula |
| CUDA OOM | `batch` dusur (8), `imgsz` 640 yap |
| CarDD label bulunamadi | `convert_cardd.py` + symlink/images path kontrol |
| Panel `unknown` cok fazla | `min overlap-ratio` dusur veya parca modeli kalitesini artir |
| Egitim cok yavas | Task Manager'da GPU kullanimini kontrol et, `device=0` oldugundan emin ol |

---

*Son guncelleme: 2026-06-15*
