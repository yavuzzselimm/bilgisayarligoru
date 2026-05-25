# Bilgisayarlı Görme Projesi

Bu proje, görüntü işleme ve bilgisayarlı görme (Computer Vision) teknikleri kullanılarak geliştirilmiş 3 farklı bölümden oluşmaktadır. Proje tamamen Python ve OpenCV kütüphanesi kullanılarak yazılmıştır.

## 🛠️ Kurulum

Projedeki kodları çalıştırabilmek için Python yüklü olmalı ve gerekli kütüphaneler kurulmalıdır. Terminal üzerinden aşağıdaki komut ile kütüphaneleri kurabilirsiniz:

```bash
pip install opencv-python numpy
```

---

## 📂 Bölümler ve Çalıştırılmaları

### 1️⃣ Bölüm 1: Evrak Tarama ve Düzeltme (Document Scanner)
Bir zemin üzerinde çekilmiş belge veya evrak fotoğrafını bulur, arka plandan ayırır (maskeleme), eğriliğini (perspektif) düzeltir ve kontrastını artırarak taranmış temiz bir belge formuna getirir.

**📁 Klasör:** `bolum1_evrak_tarama`

**Nasıl Çalıştırılır?**
Terminalde ilgili klasöre giderek şu komutu çalıştırabilirsiniz:
```bash
python evrak_tarama.py --input test_images/test_belge.jpg
```
*Not: Sistem çalışırken aşamaları görmek isterseniz komutun sonuna `--debug` ekleyebilirsiniz.*

---

### 2️⃣ Bölüm 2: Optik Okuyucu (OMR System)
Pembe renkli klasik bir Türk optik formunu (Çözüm Optik Okuyucu) okuyabilen sistemdir. Gerçek optik formlar üzerindeki işaretlenmiş baloncukları, kurşun kalem izlerinden tespit eder ve öğrenci numarası ile cevapları çıkararak puanlama yapar. 

Özellikler:
- Ayarlanabilir soru sayısı (1-60 arası).
- Öğrenci numarasını okuma.
- Sentetik (otomatik) doldurulmuş test kağıdı üretebilme.
- Sonuçları `.csv` excel formatında ve `.txt` rapor formatında kaydetme.

**📁 Klasör:** `bolum2_optik_okuyucu`

**Nasıl Çalıştırılır?**
Kendi sentetik test verisini (1 cevap anahtarı ve 10 öğrenci) oluşturup tam otomatik test yapmak için:
```bash
python optik_okuyucu.py --test --questions 40 --num-students 10
```

Elinizdeki gerçek bir optik formu veya cevap anahtarını test etmek için:
```bash
python optik_okuyucu.py -a cevap_anahtari.png -s ogrenci_fotograflari_klasoru/ -q 40
```

---

### 3️⃣ Bölüm 3: Çok Taneli Madde Sayma (Particle Counting)
Beyaz bir zemin üzerinde bulunan mısır, pirinç, fındık gibi çok taneli objeleri tespit eder ve sayar. Birbirine yapışık objeleri ayırabilmek için **Watershed Segmentasyonu** kullanır.

**📁 Klasör:** `bolum3_tane_sayma`

**Nasıl Çalıştırılır?**
İstediğiniz bir resimdeki objeleri saydırmak için:
```bash
python tane_sayma.py --input /resmin/yolu.jpg
```
*İpucu: Resimdeki ufak tozları veya gölgeleri obje olarak saymasını istemiyorsanız minimum alan (min-area) filtresini kullanabilirsiniz. (Örn: Mısır taneleri için `--min-area 1000`)*
```bash
python tane_sayma.py --input /resmin/yolu.jpg --min-area 1000
```
Sistem saydığı objeleri numaralandırarak orijinal resmin yanına `..._sayilmis.jpg` olarak kaydedecektir.

---

