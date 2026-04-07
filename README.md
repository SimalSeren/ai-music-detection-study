# ai-music-detection-study

Bu proje, yapay zeka ile uretilmis muziklerin tespiti icin artefact-odakli bir arastirma hattidir.

Problem dogrudan `real vs AI music` olarak kurulmaz. Bunun yerine gercek muzikten kontrollu bozulmalar ve reconstruction family'leri uretilir; modellerin muzigin icerigini degil, uretim surecinin biraktigi izleri ogrenmesi hedeflenir.

## Mevcut arastirma durumu

- Gercek veri: `FMA-small`
- Standart format: `16 kHz`, `10 saniye`
- Ozellik: STFT amplitude spectrogram + dB scaling
- Split protokolu: track-level, leakage'siz
- Fake family destegi:
  - `griffinlim_mel32`
  - `griffinlim_mel8`
  - `resample_8k`
  - `quantize_8bit`
  - `smoothed_noise`
- Baseline modeller:
  - `SimpleCNN`
  - `ResNetSpectrogram`
  - `ArtifactNet`
  - `Attention MIL`

## Dizin yapisi

- `src/build_spectrogram_dataset.py`
  - Gercek seslerden spectrogram uretir.
- `src/generate_griffinlim_recons.py`
  - Parametrik Griffin-Lim family'leri uretir.
- `src/generate_degradation_recons.py`
  - Codec gerektirmeyen degradation family'leri uretir.
- `src/build_fake_spectrogram_dataset.py`
  - Fake family audio'larini spectrogram datasetine cevirir.
- `src/create_labeled_split.py`
  - Belge v1.3 uyumlu `global_manifest`, `split_manifest` ve labeled split uretir.
- `src/create_ood_splits.py`
  - Leave-one-family-out OOD split setleri uretir.
- `src/evaluate_track_protocol.py`
  - Validation'dan threshold ve calibration ogrenip testte donduran track-level evaluation contract.
- `src/run_negative_controls.py`
  - Stem overlap, manifest butunlugu ve same-source control ozeti uretir.
- `src/random_label_sanity_check.py`
  - Random-label sanity check icin split uretir.
- `src/train_cropped_baseline.py`
  - `simplecnn`, `resnet`, `artifactnet` icin crop-based egitim.
- `src/train_clip_attention.py`
  - Attention MIL ile clip-level egitim.
- `src/plot_evaluation_suite.py`
  - ROC, PR, histogram, threshold sweep, family bar chart uretir.
- `src/build_benchmark_report.py`
  - Standart benchmark tablosu uretir.
- `src/build_ood_benchmark_report.py`
  - Held-out family OOD matrisi uretir.
- `scripts/run_research_suite.ps1`
  - Tum ID benchmark hattini calistirir.
- `scripts/run_ood_matrix.ps1`
  - Held-out family OOD deneylerini toplu calistirir.

## Standart pipeline

### 1. Gercek spectrogramlari uret

```powershell
python src/build_spectrogram_dataset.py `
  --input-list data/fma_subset_1000.txt `
  --output-dir data/processed/fma_specs
```

### 2. Fake family'leri uret

```powershell
python src/generate_griffinlim_recons.py `
  --input-list data/fma_subset_1000.txt `
  --output-root data/reconstructed `
  --family-name griffinlim_mel32 `
  --n-iter 32
```

```powershell
python src/generate_griffinlim_recons.py `
  --input-list data/fma_subset_1000.txt `
  --output-root data/reconstructed `
  --family-name griffinlim_mel8 `
  --n-iter 8
```

```powershell
python src/generate_degradation_recons.py `
  --input-list data/fma_subset_1000.txt `
  --output-root data/reconstructed `
  --families resample_8k quantize_8bit smoothed_noise
```

### 3. Fake spectrogramlari olustur

```powershell
python src/build_fake_spectrogram_dataset.py `
  --input-dir data/reconstructed `
  --output-dir data/processed/fake_specs
```

### 4. Leakage'siz split ve OOD split uret

```powershell
python src/create_labeled_split.py `
  --real-dir data/processed/fma_specs `
  --fake-dir data/processed/fake_specs `
  --output-dir data/splits_labeled
```

Bu adim su dosyalari uretir:

- `data/splits_labeled/global_manifest.csv`
- `data/splits_labeled/split_manifest.csv`
- `data/splits_labeled/manifest.csv`
- `data/splits_labeled/train.txt`
- `data/splits_labeled/val.txt`
- `data/splits_labeled/test.txt`

```powershell
python src/create_ood_splits.py `
  --manifest-path data/splits_labeled/manifest.csv `
  --output-dir data/splits_labeled/ood
```

### 5. Modelleri egit

```powershell
python src/train_cropped_baseline.py `
  --model simplecnn `
  --train-split data/splits_labeled/train.txt `
  --val-split data/splits_labeled/val.txt `
  --test-split data/splits_labeled/test.txt `
  --output-root results_simplecnn
```

```powershell
python src/train_cropped_baseline.py `
  --model resnet `
  --train-split data/splits_labeled/train.txt `
  --val-split data/splits_labeled/val.txt `
  --test-split data/splits_labeled/test.txt `
  --output-root results_resnet
```

```powershell
python src/train_cropped_baseline.py `
  --model artifactnet `
  --train-split data/splits_labeled/train.txt `
  --val-split data/splits_labeled/val.txt `
  --test-split data/splits_labeled/test.txt `
  --output-root results_artifactnet
```

```powershell
python src/train_clip_attention.py `
  --train-split data/splits_labeled/train.txt `
  --val-split data/splits_labeled/val.txt `
  --test-split data/splits_labeled/test.txt `
  --output-root results_attention
```

### 6. Gorsellestirme ve benchmark

```powershell
python src/plot_evaluation_suite.py --experiment-dir results_artifactnet
```

```powershell
python src/build_benchmark_report.py `
  --experiment-dirs results_simplecnn results_resnet results_artifactnet results_attention `
  --output-dir results_reports
```

### 6b. Belge v1.3 evaluation contract

```powershell
python src/evaluate_track_protocol.py `
  --val-predictions results_resnet/logs/val_best_clip_predictions.csv `
  --test-predictions results_resnet/logs/test_clip_predictions.csv `
  --global-manifest data/splits_labeled/global_manifest.csv `
  --split-manifest data/splits_labeled/split_manifest.csv `
  --output-dir results_resnet/protocol_eval
```

```powershell
python src/run_negative_controls.py `
  --global-manifest data/splits_labeled/global_manifest.csv `
  --split-manifest data/splits_labeled/split_manifest.csv `
  --test-predictions results_resnet/protocol_eval/predictions_track_test.csv `
  --output-dir results_resnet/negative_controls
```

```powershell
python src/random_label_sanity_check.py `
  --input-split data/splits_labeled/train.txt `
  --output-split data/splits_labeled/train_random_labels.txt
```

### 7. Tum arastirma hattini tek komutla calistir

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_research_suite.ps1 -PythonExe python
```

### 8. OOD held-out family matrisini calistir

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_ood_matrix.ps1 -PythonExe python
```

## Beklenen ciktilar

Her deney klasorunde:

- `logs/training_history.csv`
- `logs/test_clip_predictions.csv`
- `logs/test_family_metrics.csv`
- `metrics/test_clip_metrics.json`
- `figures/test_roc_pr.png`
- `figures/test_score_hist.png`
- `figures/test_family_bars.png`
- `figures/test_confusion_matrix.png`
- `figures/test_threshold_sweep.png`

Rapor klasorunde:

- `benchmark_summary.csv`
- `benchmark_summary.md`
- `ood_benchmark_summary.csv`
- `ood_benchmark_summary.md`
- `ood_benchmark_matrix_balanced_accuracy.csv`

## Arastirma hedefi

Bu repo artik setup asamasini gecmistir. Ana hedef:

1. Tek fake family ezberini kiran modeller gelistirmek
2. Coklu fake family ile egitim yapip OOD performansi olcmek
3. Paper ile kiyaslanabilir metrik ve gorsel raporlama uretmek
4. `ArtifactNet` ve `Attention MIL` gibi ozgun modellerin katkisinin net etkisini gostermek

## Belge v1.3 uyum durumu

Bu repoda su protokol maddeleri aktiflestirilmistir:

- track-level leakage-free split
- global manifest + split manifest
- validation-only threshold secimi
- validation-only temperature scaling
- track-level ana raporlama
- same-source / overlap / integrity negatif kontrol girisleri
- random-label sanity check girisi

Bir sonraki faz:

- same-source auxiliary control deneylerini otomatiklestirmek
- PR-AUC'yi ana checkpoint metriği yapmak
- Sprint 1 resmi benchmark kosularini tamamlamak
