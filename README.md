# Akilli Odak Tanima PoC

Mac uzerinde gercek zamanli nesne tespiti ve takibi yapan, kullanicinin video uzerinde bir nesneye tiklayarak ona kilitlenebildigi PyQt6 tabanli bir Proof of Concept uygulamasi.

Bu proje, **arac hasar tespiti** sisteminin ilk adimidir. Su anki fazda genel nesne tanima ve takip altyapisi kurulmustur; hasar siniflari (cizik, gocuk, kirik cam vb.) sonraki fazda ozel egitilmis model ile eklenecektir.

---

## Icindekiler

- [Proje Ozeti](#proje-ozeti)
- [Ozellikler](#ozellikler)
- [Teknoloji Yigini](#teknoloji-yigini)
- [Proje Yapisi](#proje-yapisi)
- [Mimari](#mimari)
- [Nasil Calisir](#nasil-calisir)
- [Kurulum](#kurulum)
- [Kullanim](#kullanim)
- [Arayuz Bilesenleri](#arayuz-bilesenleri)
- [Parametreler](#parametreler)
- [Odak Sistemi (Tikla-Kilitle)](#odak-sistemi-tikla-kilitle)
- [Tespit ve Takip Motoru](#tespit-ve-takip-motoru)
- [Performans](#performans)
- [macOS Notlari](#macos-notlari)
- [Sinirlamalar](#sinirlamalar)
- [Sorun Giderme](#sorun-giderme)
- [Yol Haritasi](#yol-haritasi)

---

## Proje Ozeti

Amac: Kamera, video dosyasi veya RTSP akisindan gelen goruntulerde nesneleri tespit etmek, takip etmek ve kullanicinin sectigi bir nesneye odaklanarak detayli bilgi sunmak.

Neden bu yaklasim?

- Ekranda her nesneye kalin kutu cizmek yerine **tek bir odak nesne** vurgulanir; demo daha temiz ve profesyonel gorunur.
- Tikla-kilitle mekanigi, ileride hasar bolgesi secimi icin ayni UX desenini kullanmamiza olanak tanir.
- Takip ID'si, sure ve hareket yonu gibi metrikler ekibe "bu sadece bir kutu cizen script degil" mesajini verir.

Su anki model **YOLO26n** (COCO veri seti, 80 sinif: arac, insan, bisiklet vb.) kullanir. Hasar tespiti bu fazda kapsam disidir.

---

## Ozellikler

| Ozellik | Aciklama |
|---------|----------|
| Coklu kaynak | Webcam, video dosyasi (mp4/mov/avi/mkv), RTSP stream |
| Coklu model | YOLO26n, DETR-R50, D-FINE-M, RF-DETR-Base; Grid modunda 2-3 model karsilastirma |
| Gercek zamanli tespit | Secili model ile frame bazli nesne algilama |
| Kalici takip | ByteTrack ile nesnelere benzersiz track ID atama |
| Tikla-kilitle | Video uzerinde tiklanan nesneye odaklanma |
| Detay paneli | Sinif, guven, boyut, konum, renk, sure, hareket yonu |
| Akilli overlay | Odak nesne vurgulu; digerleri soluk ince kutu |
| Cihaz secimi | CUDA / Apple MPS / CPU otomatik fallback |
| Frame kaydetme | Tespit iceren frameleri `./output` klasorune kaydetme |
| macOS kamera destegi | AVFoundation ile dogru kamera index eslesmesi |

---

## Teknoloji Yigini

| Katman | Teknoloji | Rol |
|--------|-----------|-----|
| UI | PyQt6 | Masaustu arayuz, video paneli, kontrol kartlari |
| Capture | capture_engine.py | Video dongusu, odak yonetimi, callback'ler |
| Backend | backends.py | 4 model backend, ByteTrack, Supervision annotator |
| Gorsel isleme | OpenCV | Video yakalama, cizim, renk analizi |
| Tespit | Ultralytics / Transformers / RF-DETR | Nesne tespiti |
| Takip | ByteTrack (supervision) | Frame'ler arasi ID tutarliligi |
| Derin ogrenme | PyTorch | Model inference, GPU/MPS destegi |
| macOS entegrasyonu | pyobjc AVFoundation | Kamera cihaz isimleri |

---

## Proje Yapisi

```
object-detection/
├── main.py              # PyQt6 arayuzu, tiklanabilir video, odak paneli
├── capture_engine.py    # Video dongusu, odak yonetimi, callback'ler
├── backends.py          # Model backend'leri (YOLO, DETR, D-FINE, RF-DETR)
├── sources.py           # macOS kamera listeleme (AVFoundation)
├── requirements.txt     # Python bagimliliklari
├── yolo26n.pt           # Onceden indirilmis YOLO model agirliklari
├── output/              # (opsiyonel) Kaydedilen frameler
└── README.md
```

### Dosya Sorumluluklari

**`main.py`**
- PyQt6 ana pencere (sol kontrol / orta video / sag inspector)
- Kaynak secimi (Kamera / Video / RTSP)
- Tekil ve Grid gorunum modlari
- Parametre kontrolleri (confidence, skip frames)
- `ClickableVideoLabel`: video uzerinde tiklama algilama
- Widget -> frame koordinat donusumu (letterbox hesabi)
- Odak Nesne paneli ve Kilidi birak butonu
- `CaptureBridge`: CaptureEngine callback'lerini PyQt sinyallerine cevirir

**`capture_engine.py`**
- `threading.Thread` uzerinde video okuma dongusu
- Her N frame'de backend `infer -> update -> annotate` cagrisi
- Odak kilitleme / birakma mantigi
- Callback'ler: `on_panel`, `on_focus`, `on_status`, `on_error`, `on_device`, `on_finished`

**`backends.py`**
- `UltralyticsBackend`, `TransformersBackend`, `RFDetrBackend`
- `MODEL_REGISTRY`: gosterim adi -> backend factory
- ByteTrack takip + Supervision annotator'lar
- `resolve_device()`: cuda -> mps -> cpu secimi

**`sources.py`**
- macOS'ta AVFoundation ile kamera isimlerini OpenCV index sirasiyla eslestirir
- Fallback: `system_profiler` veya jenerik `Kamera 0..N`

---

## Mimari

```
┌─────────────────────────────────────────────────────────────────┐
│                    main.py (PyQt6 UI Thread)                     │
│  ┌──────────────┐  ┌─────────────┐  ┌────────────────────────┐ │
│  │ Video Panel  │  │ Odak Panel  │  │ Kaynak / Param / Log    │ │
│  │ (tiklanabilir)│  │ (detaylar)  │  │ kontrolleri             │ │
│  └──────┬───────┘  └──────▲──────┘  └────────────────────────┘ │
│         │ tiklama          │ focus_info (queued signal)          │
│         ▼                  │                                       │
│  ┌──────────────────────────────────────────────────────────────┐│
│  │              CaptureBridge (QObject sinyalleri)               ││
│  └──────────────────────────┬───────────────────────────────────┘│
└─────────────────────────────┼───────────────────────────────────┘
                              │ callback'ler
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              capture_engine.py (Worker Thread)                   │
│  VideoCapture → infer → ByteTrack → annotate → callback emit    │
└─────────────────────────────┬───────────────────────────────────┘
                              ▼
                    ┌──────────────────┐
                    │   backends.py    │
                    │ 4 model backend  │
                    └──────────────────┘
```

### Veri Akisi

```
VideoCapture.read()
       │
       ▼
  [her N frame]
       │
       ▼
backend.infer(frame)  ──►  sv.Detections
       │
       ▼
backend.update(detections)  ──►  tracker_id atanmis detections
       │
       ├──► _resolve_focus_request()   (tiklanan noktadaki en kucuk kutu)
       ├──► _publish_focus_info()    (detay dict olustur)
       │
       ▼
backend.annotate(frame, detections, focus_id)
       │
       ├──► on_panel  ──► CaptureBridge.panel_ready  ──► UI video guncelleme
       └──► on_focus  ──► CaptureBridge.focus_info  ──► UI odak paneli
```

---

## Nasil Calisir

1. Kullanici kaynak secer (kamera, video veya RTSP) ve **Baslat**'a basar.
2. `CaptureEngine` arka planda modeli yukler ve video akisini okumaya baslar.
3. Her `skip_frames` kadar frame'de secili backend `infer + ByteTrack` calisir; her nesneye bir `track_id` atanir.
4. Kullanici video uzerinde bir nesneye tiklar.
5. Tiklanan koordinat frame uzayina cevrilir; o noktayi iceren en kucuk kutuya sahip nesnenin `track_id`'si odak olarak kilitlenir.
6. Odak nesne mavi kalin kutu ile vurgulanir; yan panelde sinif, guven, boyut, renk, sure ve hareket yonu gosterilir.
7. Odak nesne tespitlerde gorunmezse panel "Odak kayboldu" durumunu gosterir.
8. **Kilidi birak** ile manuel olarak odak sifirlanabilir.

---

## Kurulum

### Gereksinimler

- macOS (birincil hedef platform; Linux/Windows da calisabilir)
- Python 3.11 veya uzeri
- Webcam veya test icin video dosyasi
- ~2 GB disk (PyTorch + model)

### Adimlar

```bash
# 1. Repoyu klonla veya proje dizinine git
cd object-detection

# 2. Sanal ortam olustur
python3 -m venv .venv
source .venv/bin/activate

# 3. Bagimliliklari kur
pip install -r requirements.txt

# 4. Model (ilk calistirmada otomatik indirilir; yoksa manuel)
# yolo26n.pt dosyasi proje kokunde olmali
```

### Bagimliliklar (`requirements.txt`)

```
ultralytics>=8.4.61      # YOLO modeli
opencv-python>=4.9       # Video yakalama ve gorsel isleme
PyQt6>=6.7               # Masaustu arayuz
supervision>=0.27.0      # Detections + ByteTrack + annotator
transformers>=4.50       # Meta DETR, USTC D-FINE (HuggingFace)
timm>=1.0                # DETR backbone
pillow>=10               # Gorsel on-isleme
rfdetr>=1.0              # Roboflow RF-DETR
pyobjc-framework-AVFoundation>=10.0   # macOS kamera isimleri (sadece darwin)
```

PyTorch, `ultralytics` kurulumu sirasinda otomatik gelir.

Cross-vendor modeller (DETR, D-FINE, RF-DETR) ilk secimde HuggingFace Hub / Roboflow uzerinden otomatik indirilir.

---

## Kullanim

```bash
source .venv/bin/activate
python main.py
```

### Hizli Baslangic

1. **Kaynak** kartindan `Kamera`, `Video` veya `RTSP` sec.
2. Kamera modunda listeden cihazi sec; video modunda **Gozat** ile dosya sec.
3. **Baslat**'a tikla; model yuklenir ve akis baslar.
4. Video uzerinde bir nesneye tikla; odak kilitlenir.
5. **Odak Nesne** panelinde detaylari izle.
6. **Kilidi birak** ile odagi serbest birak veya **Durdur** ile akisi kapat.

### Kaynak Ornekleri

| Tip | Ornek |
|-----|-------|
| Webcam | Dropdown'dan `[0] FaceTime HD Camera` sec |
| Video dosyasi | `/Users/you/Downloads/traffic.mp4` |
| RTSP | `rtsp://admin:password@192.168.1.100:554/stream1` |

---

## Arayuz Bilesenleri

### Sol Panel — Video

- Canli video akisi
- FPS overlay (sol ust kose)
- Odak nesne: mavi kalin kutu + kompakt etiket (sinif, skor, ID)
- Diger nesneler: soluk ince gri kutu
- Tiklanabilir alan (letterbox offset hesabi ile dogru koordinat)

### Sag Panel — Kontroller

| Kart | Icerik |
|------|--------|
| Kaynak | Kamera / Video / RTSP secimi |
| Parametreler | Confidence slider, Skip frames, Frame kaydetme |
| Odak Nesne | Sinif, guven, ID, boyut, en-boy, konum, renk, sure, yon |
| Canli Durum | FPS ve toplam tespit sayisi |
| Baslat / Durdur | Akis kontrolu |
| Log | Durum mesajlari ve hatalar |

### Odak Paneli Alanlari

| Alan | Aciklama |
|------|----------|
| Sinif | COCO sinif adi (ornegin `car`, `person`) |
| Guven | Tespit guven skoru (0-100%) |
| Takip ID | ByteTrack tarafindan atanan kalici ID |
| Boyut | Bounding box genislik x yukseklik (piksel) |
| En/Boy | Genislik / yukseklik orani |
| Konum | Kutunun merkez koordinati (x, y) |
| Renk | Nesne ROI'sindeki baskin renk (RGB + renk karesi) |
| Sure | Nesnenin ekranda goruldugu toplam sure (saniye) |
| Yon | Hareket yonu: yukari / asagi / sola / saga / sabit |

---

## Parametreler

### Confidence (0.05 - 0.95)

Tespit esigi. Dusuk deger = daha fazla tespit ama daha fazla yanlis pozitif. Varsayilan: `0.25`.

### Skip Frames (1 - 10)

Her kac frame'de bir inference yapilacagi. Deger arttikca FPS artar, guncellik azalir.

| Deger | Etki |
|-------|------|
| 1 | Her frame inference (en guncel, en yavas) |
| 2-3 | Dengeli (onerilen) |
| 5+ | Hizli ama gecikmeli tespit |

### Frame Kaydetme

Isaretlendiginde tespit iceren frameler `./output/frame_YYYYMMDD_HHMMSS_ffffff.jpg` olarak kaydedilir.

---

## Odak Sistemi (Tikla-Kilitle)

### Koordinat Donusumu

Video `KeepAspectRatio` ile olceklenir; kenarlarda letterbox boslugu olusabilir. Tiklanan widget koordinati su formulle frame koordinatina cevrilir:

```
scale = min(widget_w / frame_w, widget_h / frame_h)
offset_x = (widget_w - frame_w * scale) / 2
offset_y = (widget_h - frame_h * scale) / 2
frame_x = (widget_x - offset_x) * frame_w / (frame_w * scale)
frame_y = (widget_y - offset_y) * frame_h / (frame_h * scale)
```

### Odak Secim Algoritmasi

1. Tiklanan `(x, y)` noktasini iceren tum tespit kutulari bulunur.
2. Birden fazla aday varsa **en kucuk alanli** kutu secilir (ic ice kutularda icteki nesne).
3. Secilen kutunun `track_id`'si odak olarak kilitlenir.

### Odak Kaybi

Odak nesne tespitlerde bulunamazsa panel "Odak kayboldu" durumunu gosterir. Kilit manuel olarak **Kilidi birak** ile sifirlanabilir.

### Hareket Yonu

Onceki ve guncel merkez pozisyonu karsilastirilir. Piksel farki esik degerinin altindaysa `sabit`; aksi halde `sol` / `sag` / `yukari` / `asagi` veya kombinasyonu.

---

## Tespit ve Takip Motoru

### Model

Tum modeller ayni boru hattindan gecer: `infer -> sv.Detections -> ByteTrack -> annotate`. UI'dan birden fazla model secilerek tutarli karsilastirma yapilabilir.

| Model | Firma | Backend | Agirlik |
|-------|-------|---------|---------|
| YOLO26n | Ultralytics | `UltralyticsBackend` | `yolo26n.pt` (yerel) |
| DETR-R50 | Meta | `TransformersBackend` | `facebook/detr-resnet-50` (HF Hub) |
| D-FINE-M | USTC | `TransformersBackend` | `ustc-community/dfine-medium-coco` (HF Hub) |
| RF-DETR-Base | Roboflow | `RFDetrBackend` | `rf-detr-base.pt` (otomatik indirme) |

- **COCO 80 sinif**: person, car, truck, bus, bicycle, dog, chair vb.
- Yerel agirliklar proje kokunde; cross-vendor modeller ilk secimde otomatik indirilir.

### Takip

```python
detections = backend.infer(frame, conf=0.25)
detections = backend.update(detections)  # sv.ByteTrack
```

ByteTrack, dusuk guvenli tespitleri de kullanarak ID kopmalari azaltir.

### Baskin Renk

Odak nesnenin bounding box ROI'sindeki piksellerin ortalama BGR degeri hesaplanir ve RGB olarak panele yansitilir.

### Cihaz Secimi (`resolve_device`)

```
1. CUDA  (NVIDIA GPU varsa)
2. MPS   (Apple Silicon GPU)
3. CPU   (fallback)
```

Cihaz bilgisi sag ustte `DEVICE: MPS` badge'i ile gosterilir.

---

## Performans

| Ortam | Skip=2 | Skip=3 |
|-------|--------|--------|
| Apple M1/M2/M3 (MPS) | ~15-25 FPS | ~25-35 FPS |
| CPU only | ~5-10 FPS | ~8-15 FPS |
| NVIDIA GPU (CUDA) | ~30-60 FPS | ~40-80 FPS |

Performans; kaynak cozunurlugu, sahne karmasikligi ve tespit sayisina gore degisir.

**Oneriler:**
- Demo icin `skip_frames = 2` veya `3` kullan
- Confidence'i gereksiz yere dusurme (false positive artar)
- RTSP'de ag gecikmesi FPS'i dusurebilir

---

## macOS Notlari

### Kamera Izni

Ilk calistirmada macOS kamera izni isteyebilir:

`System Settings` -> `Privacy & Security` -> `Camera` -> Terminal veya IDE'ye izin ver.

Izinsiz calistirmada "Kaynak acilamadi" hatasi alinir.

### iPhone / Continuity Camera

Continuity kamerasi ilk frame'i gec verebilir. Uygulama 5 saniye bekler; hala goruntu gelmezse cihazi uyandirin ve tekrar deneyin.

### Kamera Index Eslesmesi

`sources.py`, AVFoundation API'si ile OpenCV'nin kullandigi kamera sirasini birebir eslestirir. Dropdown'daki `[0]`, `[1]` index'leri OpenCV `VideoCapture(0)` ile uyumludur.

---

## Sinirlamalar

- **Hasar tespiti yok**: Su an sadece COCO genel siniflari tespit edilir (arac, insan vb.). Cizik, gocuk, kirik cam siniflari yoktur.
- **Egitim verisi**: 10-15 etiketsiz foto ile model egitilemez; hasar fazinda yuzlerce-binlerce etiketli gorsel gerekir.
- **Tek odak**: Ayni anda yalnizca bir nesneye odaklanilabilir.
- **Platform**: Birincil hedef macOS; diger platformlarda kamera isim listeleme jenerik kalir.
- **Gercek zamanli sinir**: Yuksek cozunurluklu RTSP akislarda gecikme olabilir.

---

## Sorun Giderme

| Sorun | Cozum |
|-------|-------|
| "Kaynak acilamadi" | Kamera iznini kontrol et; baska uygulama kamerayi kullaniyor olabilir |
| Model yuklenmiyor | `pip install ultralytics torch` tekrar calistir; internet baglantisi gerekli |
| Dusuk FPS | `skip_frames` degerini 3-5'e cikar |
| Tikla-kilitle calismiyor | Nesnenin kutusu gorunur olmali; confidence'i dusur veya nesneye tam kutunun uzerine tikla |
| RTSP baglanamiyor | URL, kullanici/sifre ve ag erisimini kontrol et |
| iPhone kamerasi bos | Cihazi uyandir, 5 sn bekle, tekrar Baslat |
| MPS hatasi | PyTorch MPS destegi icin guncel surum gerekli; CPU'ya duser |

---

## Yol Haritasi

### Faz 1 — Akilli Odak Tanima (mevcut)

- [x] YOLO + ByteTrack pipeline
- [x] Tikla-kilitle UX
- [x] Detayli odak paneli
- [x] Kamera / video / RTSP destegi

### Faz 2 — Hasar Tespiti

1. **Veri toplama**: Etiketli hasar gorselleri (cizik, gocuk, kirik cam, kirik far, ezilme)
   - Public dataset: [Roboflow Car Damage](https://universe.roboflow.com), CarDD
   - Kendi veriniz: minimum 500-1000 etiketli gorsel / sinif onerilir
2. **Model egitimi**: YOLO26 uzerinde fine-tune
3. **Arayuz entegrasyonu**: Model secici (genel / hasar), hasar tipi ve siddet paneli
4. **Foto modu**: Tek fotograf yukleme ve statik analiz (sigorta ekspertiz senaryosu)

### Faz 3 — Uretim Hazirligi

- REST API katmani
- Batch foto analizi
- Rapor ciktisi (PDF/JSON)
- Edge deployment (ONNX / TensorRT)

---

## Lisans ve Katki

Bu proje dahili R&D / PoC amaclidir. Katki ve sorular icin proje sahibiyle iletisime gecin.
