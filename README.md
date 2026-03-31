# 2. Hafta İlerleme Özeti

## Bu hafta yapılanlar

- Veri seti 100 → 500 → 1000 örneğe genişletildi  
- Spectrogram üretim pipeline’ı büyük veri ile yeniden çalıştırıldı  
- Griffin-Lim ile sahte veri üretimi geniş veri üzerinde tekrar yapıldı  
- Real/fake etiketli veri seti güncellendi  
- Cropped spectrogram yaklaşımı ile model yeniden eğitildi  
- Eğitim sürecine ek metrikler eklendi:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix  
- Eğitim sonuçları CSV olarak kaydedildi  
- Eğitim eğrileri (loss ve accuracy) üretildi  
- En iyi model kaydedildi  
- Noise tabanlı farklı bir fake veri üretildi  
- Model farklı veri dağılımı üzerinde test edildi (generalization test)

## Eğitim sonuçları (aynı dağılım)

Epoch 1/15 | Train Loss: 0.2085 | Train Acc: 0.9150 | Val Loss: 0.0544 | Val Acc: 0.9806 | Prec: 0.9758 | Rec: 0.9867 | F1: 0.9812
Confusion Matrix:
 [[1323   35]
 [  19 1409]]
--- Best model saved! ---
Epoch 2/15 | Train Loss: 0.0343 | Train Acc: 0.9891 | Val Loss: 0.0260 | Val Acc: 0.9903 | Prec: 0.9930 | Rec: 0.9881 | F1: 0.9905
Confusion Matrix:
 [[1348   10]
 [  17 1411]]
--- Best model saved! ---
Epoch 3/15 | Train Loss: 0.0287 | Train Acc: 0.9900 | Val Loss: 0.0705 | Val Acc: 0.9885 | Prec: 0.9979 | Rec: 0.9797 | F1: 0.9887
Confusion Matrix:
 [[1355    3]
 [  29 1399]]
Epoch 4/15 | Train Loss: 0.0152 | Train Acc: 0.9948 | Val Loss: 0.0090 | Val Acc: 0.9964 | Prec: 0.9986 | Rec: 0.9944 | F1: 0.9965
Confusion Matrix:
 [[1356    2]
 [   8 1420]]
--- Best model saved! ---
Epoch 5/15 | Train Loss: 0.0098 | Train Acc: 0.9966 | Val Loss: 0.0095 | Val Acc: 0.9953 | Prec: 0.9910 | Rec: 1.0000 | F1: 0.9955
Confusion Matrix:
 [[1345   13]
 [   0 1428]]
Epoch 6/15 | Train Loss: 0.0093 | Train Acc: 0.9969 | Val Loss: 0.0088 | Val Acc: 0.9982 | Prec: 0.9993 | Rec: 0.9972 | F1: 0.9982
Confusion Matrix:
 [[1357    1]
 [   4 1424]]
--- Best model saved! ---
Epoch 7/15 | Train Loss: 0.0054 | Train Acc: 0.9982 | Val Loss: 0.0139 | Val Acc: 0.9932 | Prec: 1.0000 | Rec: 0.9867 | F1: 0.9933
Confusion Matrix:
 [[1358    0]
 [  19 1409]]
Epoch 8/15 | Train Loss: 0.0083 | Train Acc: 0.9967 | Val Loss: 0.0044 | Val Acc: 0.9982 | Prec: 1.0000 | Rec: 0.9965 | F1: 0.9982
Confusion Matrix:
 [[1358    0]
 [   5 1423]]
Epoch 9/15 | Train Loss: 0.0051 | Train Acc: 0.9982 | Val Loss: 0.0026 | Val Acc: 0.9993 | Prec: 1.0000 | Rec: 0.9986 | F1: 0.9993
Confusion Matrix:
 [[1358    0]
 [   2 1426]]
--- Best model saved! ---
Epoch 10/15 | Train Loss: 0.0102 | Train Acc: 0.9973 | Val Loss: 0.0060 | Val Acc: 0.9989 | Prec: 1.0000 | Rec: 0.9979 | F1: 0.9989
Confusion Matrix:
 [[1358    0]
 [   3 1425]]
Epoch 11/15 | Train Loss: 0.0030 | Train Acc: 0.9990 | Val Loss: 0.0021 | Val Acc: 0.9993 | Prec: 1.0000 | Rec: 0.9986 | F1: 0.9993
Confusion Matrix:
 [[1358    0]
 [   2 1426]]
Epoch 12/15 | Train Loss: 0.0045 | Train Acc: 0.9987 | Val Loss: 0.0055 | Val Acc: 0.9978 | Prec: 0.9993 | Rec: 0.9965 | F1: 0.9979
Confusion Matrix:
 [[1357    1]
 [   5 1423]]
Epoch 13/15 | Train Loss: 0.0034 | Train Acc: 0.9990 | Val Loss: 0.0028 | Val Acc: 0.9996 | Prec: 1.0000 | Rec: 0.9993 | F1: 0.9996
Confusion Matrix:
 [[1358    0]
 [   1 1427]]
--- Best model saved! ---
Epoch 14/15 | Train Loss: 0.0120 | Train Acc: 0.9966 | Val Loss: 0.0128 | Val Acc: 0.9935 | Prec: 0.9986 | Rec: 0.9888 | F1: 0.9937
Confusion Matrix:
 [[1356    2]
 [  16 1412]]
Epoch 15/15 | Train Loss: 0.0027 | Train Acc: 0.9988 | Val Loss: 0.0045 | Val Acc: 0.9986 | Prec: 0.9993 | Rec: 0.9979 | F1: 0.9986
Confusion Matrix:
 [[1357    1]
 [   3 1425]]
Training history saved to results\logs\training_history.csv
Training curves saved to results\figures\training_curves.png


### En iyi validation sonucu
- Epoch: 14  
- Accuracy: 1.0000  
- Precision: 1.0000  
- Recall: 1.0000  
- F1-score: 1.0000  

### Confusion Matrix

```text
[[1358    0]
 [   0 1428]]
```

### Yorum

Model eğitim dağılımında (Griffin-Lim) çok yüksek başarı elde etmiştir.  
Bu durum modelin mevcut veri üzerinde real ve fake ayrımını iyi öğrendiğini gösterir.  
Ancak bu sonuç tek başına modelin genelleme yapabildiğini göstermez.

## Genelleme testi (Generalization Test)

Model Griffin-Lim ile eğitilip, farklı bir fake dağılımı olan noise tabanlı veri üzerinde test edilmiştir.

### Sonuç

- Accuracy: 0.4999  
- Precision: 0.0000  
- Recall: 0.0000  
- F1-score: 0.0000  

### Confusion Matrix

```text
[[13955     3]
 [13958     0]]
```

### Yorum

Model, eğitim sırasında gördüğü dağılım dışında tamamen başarısız olmuştur.  
Neredeyse tüm örnekleri “gerçek” olarak sınıflandırmıştır.  

Bu durum modelin genel bir sahte müzik kavramı öğrenmediğini,  
belirli bir üretim yöntemine (Griffin-Lim) ait artefact’ları öğrendiğini göstermektedir.

## Görseller

- Eğitim eğrileri (loss & accuracy)  (Yeni Eklenen)
- Real vs Fake spectrogram karşılaştırması  
- Ortalama spectrogram karşılaştırması  

## Mevcut durum

Şu anda proje:

- Çalışan veri hazırlama pipeline’ına sahiptir  
- Spectrogram üretimi tamamlanmıştır  
- Sahte veri üretimi çalışmaktadır  
- CNN tabanlı baseline model eğitilmiştir  
- Metrik ve grafik üretimi yapılmaktadır  
- Farklı veri dağılımında başarısızlık (generalization problemi) gösterilmiştir  

## Sınırlılıklar

- Fake veri üretimi henüz basit yöntemlerle yapılmıştır (Griffin-Lim, noise)  
- Model mimarisi basit CNN’dir  
- Neural decoder tabanlı gerçek AI üretimleri kullanılmamıştır  
- Değerlendirme spectrogram crop seviyesinde yapılmıştır  

## Sonraki adımlar

- Daha güçlü model (paper’a yakın CNN)  
- Farklı fake türleriyle birlikte eğitim  
- Veri augmentasyonu  
- Daha gerçekçi AI-generated müzik verileri  
- Manipülasyon testleri  

