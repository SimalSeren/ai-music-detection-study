# AI Music Detection Study

Bu proje, yapay zeka ile uretilmis muziklerin tespiti problemine artefact-odakli ve kontrollu bir deney duzeniyle yaklasir.

Problem dogrudan `real vs AI music` olarak kurulmaz. Bunun yerine:

- gercek muzik verisi kullanilir
- ayni veri reconstruction / degradation sureclerinden gecirilir
- modelin muzigin icerigini degil, bu sureclerin biraktigi izleri ogrenmesi hedeflenir

## Veri ve temsil

- Gercek veri: `FMA-small`
- Standart format: `16 kHz`, `10 saniye`
- Ozellik: `STFT log-amplitude spectrogram`
- Cikti formatı: `.npy`

Kullanilan fake family ornekleri:

- `resample_8k`
- `quantize_8bit`
- `smoothed_noise`
- `Griffin-Lim` varyantlari

## Arastirma protokolu

Projede daha guvenilir bir deney zemini icin su yapilar aktiflestirilmistir:

- leakage-free `track-level split`
- `global_manifest.csv`
- `split_manifest.csv`
- track-level evaluation
- validation-only threshold secimi
- calibration / temperature scaling
- negative controls

Bu sayede ayni track'in turevleri farkli split'lere dagilmadan, daha temiz bir degerlendirme yapilabilir.

## Modeller

Mevcut model ailesi:

- `SimpleCNN`
- `ResNetSpectrogram`
- `ArtifactNet`
- `SpectrogramTransformer`
- `Attention MIL`

`SpectrogramTransformer`, Audio Spectrogram Transformer (AST) fikrine yakin, spectrogram patch'leri uzerinde self-attention kullanan transformer tabanli bir baseline olarak eklenmistir.

`ArtifactNet`, ham spectrogram ile artefact-vurgulu ikinci kolu birlestiren ozgun bir modeldir.

## Uygulama

Mevcut en iyi checkpoint ile calisan demo uygulamasi:

- `app/best_model_app.py`

Calistirma:

```powershell
streamlit run app/best_model_app.py
```

Uygulama:

- ses dosyasi yuklemeyi
- best model checkpoint'i ile tahmin almayi
- track-level fake olasiligi gormeyi
- spectrogram ve crop-level skor ozetini gorsellestirmeyi

destekler.

## Egitim

Crop-based egitim:

```powershell
python src/train_cropped_baseline.py `
  --model resnet `
  --train-split data/splits_labeled/train.txt `
  --val-split data/splits_labeled/val.txt `
  --test-split data/splits_labeled/test.txt `
  --output-root results_resnet
```

Transformer baseline:

```powershell
python src/train_cropped_baseline.py `
  --model transformer `
  --train-split data/splits_labeled/train.txt `
  --val-split data/splits_labeled/val.txt `
  --test-split data/splits_labeled/test.txt `
  --output-root results_transformer
```

Clip-level attention modeli:

```powershell
python src/train_clip_attention.py `
  --train-split data/splits_labeled/train.txt `
  --val-split data/splits_labeled/val.txt `
  --test-split data/splits_labeled/test.txt `
  --output-root results_attention
```

## Pilot sonuclar

Kucuk olcekli pilot benchmark'ta:

- `ResNet` same-family ayarda yuksek performans vermistir
- mixed-family egitimde performans dusse de guclu kalmistir
- `ArtifactNet` mixed-family ayarda ek bir kiyas saglamistir
- `SpectrogramTransformer` ilk pilot kosuda benchmark'a eklenmistir
- leave-one-family-out OOD testlerinde genelleme problemi devam etmistir

Ana arastirma mesaji:

**Ayni dagilimda yuksek basari alinabilse de, held-out fake family testlerinde performans ciddi bicimde dusmektedir.**

Bu da modelin genel bir fake kavramindan cok family-spesifik artefact'lari ogrendigini gostermektedir.

## Sonuc dosyalari

Ana ozetler:

- `results_reports_today/benchmark_summary.csv`
- `results_reports_today/ood_benchmark_summary.csv`
- `results_reports_today/ood_benchmark_matrix_balanced_accuracy.csv`

Ana ozet grafikler:

- `results_reports_today/summary_plots/model_comparison_bars.png`
- `results_reports_today/summary_plots/same_mixed_ood_balanced_accuracy.png`
- `results_reports_today/summary_plots/ood_heatmap_balanced_accuracy.png`
- `results_reports_today/summary_plots/ood_family_comparison.png`

## Sonraki adimlar

- `200-track` benchmark hattini tamamlamak
- `Griffin-Lim` family uretimini daha hafif ayarlarla tekrar kurmak
- `Attention MIL` modelini tam benchmark'a eklemek
- daha fazla fake family ile egitimi genisletmek
- multi-seed evaluation yapmak
