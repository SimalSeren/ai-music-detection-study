## 3. Hafta İlerleme Özeti

### Bu hafta yapılanlar

- Proje, baseline deney yapısından çıkarılıp daha kontrollü bir araştırma protokolüne taşındı
- Leakage-free, track-level split mantığı kuruldu
- Aynı track’in `real` ve ondan türetilen tüm `fake` örneklerinin aynı split içinde kalması sağlandı
- Manifest tabanlı veri yönetimi eklendi:
  - `global_manifest.csv`
  - `split_manifest.csv`
  - `manifest.csv`
- Track-level evaluation altyapısı eklendi
- Validation üzerinden threshold öğrenme ve testte sabit threshold kullanımı eklendi
- Calibration / temperature scaling desteği eklendi
- Negatif kontrol ve sanity-check altyapısı eklendi:
  - split overlap kontrolü
  - manifest integrity kontrolü
  - same-source control
- Baseline model yapısı genişletildi:
  - `ResNet` tabanlı spectrogram modeli eklendi
  - `ArtifactNet` adlı artefact odaklı özgün model eklendi
  - `Attention MIL / clip-level` model altyapısı eklendi
- Benchmark özetleme ve OOD raporlama scriptleri eklendi
- Spectrogram çıkarım hattı daha stabil olacak şekilde yeniden düzenlendi
- `740` gerçek track için spectrogram üretimi tamamlandı
- `100-track` ölçeğinde yeni pilot benchmark paketi kuruldu
- Bu paket üzerinde:
  - same-family deney
  - mixed-family deney
  - leave-one-family-out OOD deneyleri
  çalıştırıldı
- Sonuçlar için özet tablo ve özet grafikler üretildi
- `200-track` genişletme başlatıldı
- `resample_8k` ve `quantize_8bit` family’leri `200-track` paketinde üretildi
- `Griffin-Lim` family’sini daha büyük ölçekte yeniden üretme denemeleri yapıldı ancak bu adım CPU darboğazı nedeniyle tamamlanamadı (Çok zaman aldığı için yarım kalıyor)



### Kurulan yeni deney düzeni

Bu hafta yalnızca yeni model denemesi yapılmadı; deney düzeni de güçlendirildi.

Eklenen ana bileşenler:

- `global_manifest.csv`
  - tüm örneklerin merkezi kaydı
  - `real / derived`
  - `family`
  - `spectrogram path`
  - `frame count`
  - `patch count`
- `split_manifest.csv`
  - track/stem bazlı split atamaları
- track-level evaluation
  - ana karar seviyesinin crop yerine track olması
- validation-only threshold
  - test setine threshold sızıntısının engellenmesi
- calibration
  - validation tabanlı temperature scaling
- negative controls
  - split overlap
  - manifest bütünlüğü
  - same-source false positive kontrolü

Bu düzenleme ile sonuçlar daha güvenilir hale getirilmiş, kıyas yapılabilir deney zemini oluşturulmuştur.



### Veri paketi ve genişletme

Bu hafta aktif olarak kullanılan yeni pilot paket:

- `100` gerçek track
- fake family’ler:
  - `resample_8k`
  - `quantize_8bit`

Bu veriyle üç tür deney çalıştırıldı:

- same-family training/testing
- mixed-family training/testing
- leave-one-family-out OOD testing

Ayrıca veri ölçeğini büyütmek için ek çalışma başlatıldı:

- `200-track` subset oluşturuldu
- `200-track` real spectrogram paketi hazırlandı
- `resample_8k` fake family üretildi
- `quantize_8bit` fake family üretildi

Bu genişletme, veri çeşitliliğini artırma ve daha sağlam benchmark kurma yönünde atılmış ilk adımdır.



### ResNet same-family pilot sonucu

`100-track` same-family pilotta `ResNet` modeli çalıştırılmıştır.

**Sonuç**
- Accuracy: `1.0000`
- Balanced Accuracy: `1.0000`
- Precision: `1.0000`
- Recall: `1.0000`
- F1-score: `1.0000`

**Yorum**
Aynı family içinde model çok yüksek performans göstermektedir.
Bu sonuç, yeni protokol altında da modelin gördüğü dağılımı çok iyi ayırabildiğini göstermektedir.
Ancak bu sonuç tek başına genelleme anlamına gelmemektedir.


### ResNet mixed-family pilot sonucu

Model, `resample_8k + quantize_8bit` family’leri ile birlikte eğitilmiştir.

**Sonuç**
- Accuracy: `0.8667`
- Balanced Accuracy: `0.8000`
- Precision: `0.8333`
- Recall: `1.0000`
- F1-score: `0.9091`
- AUROC: `0.9550`

**Yorum**
Mixed-family eğitim, same-family kadar kolay değildir ancak model yine güçlü performans göstermektedir.
Bu, fake çeşitliliği arttığında modelin aynı dağılım içinde daha esnek davranabildiğini göstermektedir.
Ancak real sınıfında hâlâ hata görülmektedir.



### ArtifactNet mixed-family pilot sonucu

ArtifactNet, ham spectrogram ve residual benzeri artefact bilgilerini birlikte kullanan özgün modeldir.

**Sonuç**
- Accuracy: `0.7333`
- Balanced Accuracy: `0.7750`
- Precision: `0.9286`
- Recall: `0.6500`
- F1-score: `0.7647`
- AUROC: `0.9200`

**Yorum**
ArtifactNet bu küçük pilot ölçekte ResNet’i geçememiştir.
Ancak bu sonuç, modelin tamamen başarısız olduğu değil; mevcut veri ölçeğinde ek model karmaşıklığının henüz belirgin avantaj vermediği anlamına gelmektedir.



### Genelleme testi (OOD / leave-one-family-out)

Bu hafta genelleme testi daha kontrollü biçimde tekrar kurulmuştur.

Test protokolü:
- model bir fake family üzerinde eğitilir
- hiç görmediği başka bir fake family üzerinde test edilir

Bu yaklaşım, önceki “aynı dağılım yüksek başarı / farklı dağılım çöküş” gözlemini daha düzenli bir protokol altında yeniden üretmektedir.


### ResNet OOD sonucu 1

Train family: `quantize_8bit`  
Test family: `resample_8k`

**Sonuç**
- Accuracy: `0.5000`
- Balanced Accuracy: `0.5000`
- Precision: `0.0000`
- Recall: `0.0000`
- F1-score: `0.0000`
- AUROC: `0.0900`

**Yorum**
Model, held-out family üzerinde neredeyse tamamen çökmüştür.
Bu durum, modelin genel “fake” kavramı yerine eğitim family’sine özgü artefact’ları öğrendiğini göstermektedir.


### ResNet OOD sonucu 2

Train family: `resample_8k`  
Test family: `quantize_8bit`

**Sonuç**
- Accuracy: `0.4500`
- Balanced Accuracy: `0.4500`
- Precision: `0.4286`
- Recall: `0.3000`
- F1-score: `0.3529`
- AUROC: `0.2500`

**Yorum**
İki family arasında sınırlı transfer olsa da performans hâlâ düşüktür.
Bu da genelleme probleminin yeni protokol altında da devam ettiğini doğrulamaktadır.


### ArtifactNet OOD sonuçları

ArtifactNet modeli için de held-out-family deneyleri çalıştırılmıştır.

**Sonuçlar**

Train: `quantize_8bit` → Test: `resample_8k`
- Accuracy: `0.3000`
- Balanced Accuracy: `0.3000`
- F1-score: `0.0000`

Train: `resample_8k` → Test: `quantize_8bit`
- Accuracy: `0.5000`
- Balanced Accuracy: `0.5000`
- F1-score: `0.0000`

**Yorum**
ArtifactNet, bu küçük pilot ölçekte OOD problemine çözüm getirememiştir.
Bu hafta elde edilen en kritik araştırma sonucu şudur:

**model mimarisi büyütülse bile, held-out family genelleme problemi hâlâ açık şekilde devam etmektedir.**


### Negatif kontrol sonuçları

Yeni protokolde deney güvenilirliğini artırmak için negatif kontroller de çalıştırılmıştır.

**Kontrol sonuçları**
- split overlap: `0`
- duplicate sample id: `0`
- missing spectrogram path: `0`
- invalid origin row: `0`
- same-source false positive rate: `0.0`

**Yorum**
Bu sonuçlar, veri bölünmesinde bariz leakage olmadığını ve yeni protokolün teknik olarak daha güvenilir çalıştığını göstermektedir.

---

### Üretilen raporlama ve görselleştirme çıktıları

Bu hafta yalnızca metrikler değil, raporlama için özet tablolar ve grafikler de üretildi.

**Ana özet tablolar**
- `benchmark_summary.csv`
- `ood_benchmark_summary.csv`
- `ood_benchmark_matrix_balanced_accuracy.csv`

**Ana özet grafikler**
- model comparison bar chart
- same-family vs mixed-family vs OOD karşılaştırma grafiği
- OOD heatmap
- held-out family comparison grafiği

Bu çıktılar sayesinde sonuçlar artık yalnızca klasör bazlı değil, slayt ve rapor için daha düzenli biçimde sunulabilir hale gelmiştir.



### 200-track genişletme durumu

Bu hafta ayrıca veri ölçeğini artırmak için `200-track` genişletme başlatılmıştır.

**Tamamlananlar**
- `200-track` subset oluşturuldu
- `200-track` real spectrogram paketi hazırlandı
- `resample_8k` fake family üretildi
- `quantize_8bit` fake family üretildi

**Tamamlanamayan kısım**
- `Griffin-Lim` family üretimi CPU darboğazı nedeniyle tamamlanamadı

**Yorum**
Dolayısıyla `200-track` benchmark hattı henüz tam olarak koşturulmamıştır.
Ancak veri altyapısı hazırlanmış ve genişleme süreci başlatılmıştır


### Mevcut durum

Şu anda proje:

- leakage-free track-level split altyapısına sahiptir
- manifest tabanlı deney yönetimine sahiptir
- track-level evaluation ve calibration desteğine sahiptir
- ResNet ve ArtifactNet pilot modellerine sahiptir
- same-family, mixed-family ve held-out-family OOD pilot sonuçlarını üretmiştir
- genelleme problemini yeni protokol altında yeniden göstermiştir
- özet tablo ve görsel üretim altyapısına sahiptir

### Sınırlılıklar

- Pilot benchmark şu anda `100-track` ölçeğindedir
- Kullanılan fake family’ler henüz daha çok degradation tabanlıdır
- `Griffin-Lim` geniş ölçekte yeniden üretilememiştir
- Neural decoder tabanlı gerçek AI-generated music verisi henüz eklenmemiştir
- Attention MIL / clip-level model henüz tam benchmark’a sokulmamıştır
- Çoklu seed tekrarı henüz tamamlanmamıştır

### Sonraki adımlar

- `200-track` benchmark hattını tamamlamak
- `Griffin-Lim` family üretimini daha hafif ayarlarla tekrar kurmak
- `Attention MIL` modelini benchmark’a dahil etmek
- Daha fazla fake family ile mixed-family eğitimi genişletmek
- Leave-one-family-out OOD matrisini büyütmek
- Çoklu seed ile sonuçları `mean ± std` formatında raporlamak
