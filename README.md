# ai-music-detection-study
# Bu haftanın özeti 

## Referans çalışma
AI-Generated Music Detection and its Challenges

## Bu hafta yapılanlar
- Referans makale incelendi.
- FMA-small veri seti indirildi ve klasör yapısı hazırlandı.
- Ses dosyalarının Python ile yüklenmesi test edildi.
- STFT tabanlı spectrogram üretiö hattı kuruldu.
- FMA üzerinden 100 örnekten oluşan bir çalışma alt kümesi oluşturuldu.
- Gerçek spectrogram verileri .npy olarak kaydedildi.
- Griffin-Lim yöntemi ile sahte benzeri reconstruction üretildi.
- Sahte spectrogram veri seti oluşturuldu.
- Real/fake etiketli train/validation/test ayrımı yapıldı.
- PyTorch Dataset ve DataLoader sınıfları oluşturuldu.
- Basit CNN modeli kuruldu.
- Cropped spectrogram yaklaşımı ile baseline eğitim çalıştırıldı, başarılı oldu.

## Kullanılan mevcut yöntem
- Gerçek veri: FMA-small
- Sahte veri: Griffin-Lim reconstruction
- Özellik çıkarma: STFT amplitude spectrogram (dB ölçekli)
- Model: Basit CNN
- Eğitim yaklaşımı: cropped spectrogram pencereleri

## Eğitim sonucu
- Epoch 1/5 | Train Loss: 0.6976 | Train Acc: 0.6415 | Val Loss: 0.4100 | Val Acc: 0.8036
- Epoch 2/5 | Train Loss: 0.2941 | Train Acc: 0.8888 | Val Loss: 0.1249 | Val Acc: 0.9893
- Epoch 3/5 | Train Loss: 0.0866 | Train Acc: 0.9763 | Val Loss: 0.0703 | Val Acc: 0.9643
- Epoch 4/5 | Train Loss: 0.0632 | Train Acc: 0.9826 | Val Loss: 0.0992 | Val Acc: 0.9821
- Epoch 5/5 | Train Loss: 0.0440 | Train Acc: 0.9906 | Val Loss: 0.0667 | Val Acc: 0.9607
(tüm train verisi 5 kez modelden geçecek şekilde, şu an basit veriler üzerine çalıştığımız için doğruluk oranı yüksek)


## Şu anki durum
Çalışan ilk baseline kuruldu. Veri hazırlama, spectrogram çıkarma, sahte veri üretme ve CNN eğitimi aşamaları başarıyla tamamlandı.

## Sonraki adımlar (referans alınan çalışma için)
- Referans makaledeki CNN mimarisine daha yakın model kurmak.
- Precision, recall, F1 ve confusion matrix eklemek.
- Daha güçlü reconstruction yöntemleri denemek.
- Manipülasyon testleri yapmak.




