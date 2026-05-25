"""
BÖLÜM 1: Doküman Tarama ve Düzeltme
=====================================
Kamera ile çekilmiş bir evrakın:
1. Arka planının maskelenmesi (evrak dışı bölge ayrılır)
2. Kontrastının iyileştirilmesi (zemin beyazlaştırma)
3. Köşe noktalarının belirlenmesi ve trapezoid düzeltme

Kullanım:
    python dokuman_tarama.py --input gorsel.jpg [--output sonuc.jpg] [--debug]
"""

import cv2
import numpy as np
import argparse
import os
import sys

# Modül yolunu ayarla
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    resize_image, show_image, show_images_side_by_side,
    save_image, order_points, four_point_transform
)


def find_document_contour(image, debug=False):
    """
    Görüntüdeki en büyük dikdörtgensel konturu (evrak) bulur.
    
    İşlem adımları:
    1. Grayscale dönüşüm
    2. GaussianBlur ile gürültü azaltma
    3. Canny edge detection
    4. Kontur bulma ve filtreleme
    5. En büyük 4 köşeli kontur seçimi
    
    Args:
        image: Giriş görüntüsü (BGR)
        debug: True ise ara adımları gösterir
    
    Returns:
        4 köşe noktası (4x2 numpy array) veya None
    """
    # Grayscale dönüşüm
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Gürültü azaltma
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Kenar tespiti - birden fazla parametre deneriz
    # Otsu ile otomatik eşik belirleme
    v = np.median(blurred)
    lower = int(max(0, (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))
    edged = cv2.Canny(blurred, lower, upper)
    
    # Kenarları güçlendirmek için dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edged = cv2.dilate(edged, kernel, iterations=1)
    
    if debug:
        show_image("1. Kenar Tespiti (Canny)", edged)
    
    # Konturları bul
    contours, _ = cv2.findContours(
        edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Konturları alana göre büyükten küçüğe sırala
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    doc_contour = None
    
    for contour in contours:
        # Konturu yaklaşık poligona dönüştür
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        # 4 köşeli mi? (dikdörtgen/yamuk)
        if len(approx) == 4:
            # Minimum alan kontrolü (görüntünün en az %5'i olmalı)
            area = cv2.contourArea(approx)
            img_area = image.shape[0] * image.shape[1]
            if area > img_area * 0.05:
                doc_contour = approx
                break
    
    if doc_contour is None:
        print("[!] Uyarı: 4 köşeli evrak konturu bulunamadı.")
        print("    Alternatif yöntem deneniyor (adaptive threshold)...")
        
        # Alternatif: Adaptive threshold ile dene
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Morphological kapama
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close, iterations=3)
        
        contours2, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours2 = sorted(contours2, key=cv2.contourArea, reverse=True)[:5]
        
        for contour in contours2:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                area = cv2.contourArea(approx)
                img_area = image.shape[0] * image.shape[1]
                if area > img_area * 0.05:
                    doc_contour = approx
                    break
    
    if doc_contour is not None and debug:
        debug_img = image.copy()
        cv2.drawContours(debug_img, [doc_contour], -1, (0, 255, 0), 3)
        # Köşe noktalarını işaretle
        for point in doc_contour.reshape(4, 2):
            cv2.circle(debug_img, tuple(point.astype(int)), 10, (0, 0, 255), -1)
        show_image("2. Tespit Edilen Evrak Konturu", debug_img)
    
    return doc_contour


def apply_mask(image, contour):
    """
    Evrak konturunun dışındaki bölgeyi maskeler.
    Sadece evrak alanı kalır, dış alan siyah/beyaz yapılır.
    
    Args:
        image: Giriş görüntüsü
        contour: Evrak konturu (4 köşe)
    
    Returns:
        Maskelenmiş görüntü (evrak dışı beyaz)
    """
    # Boş maske oluştur
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # Kontur içini beyaz yap
    cv2.drawContours(mask, [contour], -1, 255, -1)
    
    # Maskeyi uygula
    masked = cv2.bitwise_and(image, image, mask=mask)
    
    # Dış alanı beyaz yap (evrak tarama görüntüsü için)
    white_bg = np.ones_like(image, dtype=np.uint8) * 255
    inv_mask = cv2.bitwise_not(mask)
    white_area = cv2.bitwise_and(white_bg, white_bg, mask=inv_mask)
    result = cv2.add(masked, white_area)
    
    return result


def enhance_contrast(image, whiten_background=True, debug=False):
    """
    Evrak görüntüsünün kontrastını iyileştirir (v3).
    
    Teknik: Morfolojik Arka Plan Çıkarma + Sigmoid Kontrast
    1. Morphological closing ile arka plan tahmini (yazı kenarlarını bozmaz)
    2. Division ile normalize
    3. Sigmoid kontrast eğrisi: beyazları tam beyaz, yazıları keskin siyah yapar
    4. Bilateral filter: kenarları koruyarak gürültü temizler
    5. Unsharp mask: ince ve keskin yazılar
    
    Args:
        image: Giriş görüntüsü (BGR)
        whiten_background: True ise zemin beyazlaştırılır
        debug: True ise ara adımları gösterir
    
    Returns:
        Kontrast iyileştirilmiş görüntü
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if whiten_background:
        h, w = gray.shape
        
        # === ADIM 1: Gaussian Arka Plan Tahmini ===
        # Çok büyük kernel ile arka planı tahmin et
        # Büyük kernel = pürüzsüz arka plan, yazı çevresinde halo yok
        kernel_size = max(h, w) // 3
        kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        kernel_size = max(kernel_size, 101)
        
        background = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        
        if debug:
            show_image("3a. Arka Plan Tahmini", background)
        
        # === ADIM 2: Division Normalization ===
        # Orijinali arka plana böl → eşit aydınlık
        normalized = cv2.divide(gray, background, scale=255)
        
        if debug:
            show_image("3b. Normalize", normalized)
        
        # === ADIM 3: Kontrast ve Gamma ===
        # CLAHE ile yazıları belirginleştir
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(normalized)
        
        # Gamma: yazıları koyulaştır, zemin beyaz kalsın
        gamma = 0.65
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in range(256)
        ]).astype("uint8")
        enhanced = cv2.LUT(enhanced, table)
        
        if debug:
            show_image("3c. Kontrast + Gamma", enhanced)
        
        # === ADIM 4: Keskinleştirme ===
        blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.2)
        sharpened = cv2.addWeighted(enhanced, 1.3, blurred, -0.3, 0)
        
        # === ADIM 5: Zemin Temizleme ===
        # Açık gri arka planı tertemiz beyaza it
        _, white_mask = cv2.threshold(sharpened, 245, 255, cv2.THRESH_BINARY)
        sharpened = np.where(white_mask == 255, 255, sharpened).astype(np.uint8)
        
        if debug:
            show_image("3d. Son Sonuç", sharpened)
        
        result = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
        return result
    
    else:
        # Sadece CLAHE kontrast iyileştirme (beyazlaştırma olmadan)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)
        
        enhanced_lab = cv2.merge([l_enhanced, a_channel, b_channel])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        if debug:
            show_image("3a. CLAHE Kontrast İyileştirme", enhanced)
        
        return enhanced


def correct_perspective(image, contour, debug=False):
    """
    Evrakın perspektifini düzeltir. Yamuk (trapezoid) şekli
    düz dikdörtgene dönüştürür.
    
    Args:
        image: Giriş görüntüsü
        contour: 4 köşe noktası
        debug: True ise ara adımları gösterir
    
    Returns:
        Perspektif düzeltilmiş görüntü
    """
    # Kontur noktalarını düzleştir
    pts = contour.reshape(4, 2).astype("float32")
    
    # Perspektif dönüşümü uygula
    warped = four_point_transform(image, pts)
    
    if debug:
        show_image("4. Perspektif Düzeltme", warped)
    
    return warped


def scan_document(image_path, output_path=None, debug=False, whiten=True):
    """
    Ana doküman tarama pipeline'ı.
    
    İşlem sırası:
    1. Görüntüyü yükle
    2. Evrak konturunu bul
    3. Arka planı maskele
    4. Perspektif düzelt (trapezoid → dikdörtgen)
    5. Kontrast iyileştir ve zemin beyazlat
    6. Sonucu kaydet
    
    Args:
        image_path: Giriş görüntüsünün dosya yolu
        output_path: Çıkış dosya yolu (None ise otomatik isim verilir)
        debug: True ise tüm ara adımlar gösterilir
        whiten: True ise zemin beyazlaştırılır
    
    Returns:
        İşlenmiş görüntü (numpy array)
    """
    # 1. Görüntüyü yükle
    print(f"\n{'='*60}")
    print(f"  BÖLÜM 1: DOKÜMAN TARAMA VE DÜZELTME")
    print(f"{'='*60}")
    print(f"\n[1/5] Görüntü yükleniyor: {image_path}")
    
    original = cv2.imread(image_path)
    if original is None:
        print(f"[✗] Hata: Görüntü yüklenemedi: {image_path}")
        return None
    
    print(f"      Boyut: {original.shape[1]}x{original.shape[0]} piksel")
    
    if debug:
        show_image("0. Orijinal Görüntü", original)
    
    # İşlem için yeniden boyutlandır (performans için)
    resized, ratio = resize_image(original, width=500)
    
    # 2. Evrak konturunu bul
    print("[2/5] Evrak konturu tespit ediliyor...")
    contour = find_document_contour(resized, debug=debug)
    
    if contour is None:
        print("[✗] Evrak konturu bulunamadı!")
        print("    İpucu: Evrakın arka plandan net ayrıldığından emin olun.")
        print("    Alternatif: Kontrastlı bir zemin üzerinde tekrar çekin.")
        return None
    
    print("[✓] Evrak konturu bulundu!")
    
    # Konturu orijinal boyuta geri ölçekle
    contour_original = (contour.astype("float") / ratio).astype("int")
    
    # 3. Arka planı maskele
    print("[3/5] Arka plan maskeleniyor...")
    masked = apply_mask(original, contour_original)
    
    if debug:
        show_image("2. Maskelenmiş Görüntü", masked)
    
    # 4. Perspektif düzeltme
    print("[4/5] Perspektif düzeltiliyor (trapezoid → dikdörtgen)...")
    warped = correct_perspective(original, contour_original, debug=debug)
    
    # Köşe noktalarını raporla
    pts = order_points(contour_original.reshape(4, 2).astype("float32"))
    print(f"      Köşe noktaları:")
    print(f"        Sol-Üst:  ({pts[0][0]:.0f}, {pts[0][1]:.0f})")
    print(f"        Sağ-Üst:  ({pts[1][0]:.0f}, {pts[1][1]:.0f})")
    print(f"        Sağ-Alt:  ({pts[2][0]:.0f}, {pts[2][1]:.0f})")
    print(f"        Sol-Alt:  ({pts[3][0]:.0f}, {pts[3][1]:.0f})")
    
    # 5. Kontrast iyileştirme
    print("[5/5] Kontrast iyileştiriliyor ve zemin beyazlatılıyor...")
    result = enhance_contrast(warped, whiten_background=whiten, debug=debug)
    
    # Sonucu kaydet
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_taranmis{ext}"
    
    save_image(result, output_path)
    
    print(f"\n[✓] İşlem tamamlandı!")
    print(f"    Sonuç boyutu: {result.shape[1]}x{result.shape[0]} piksel")
    
    if debug:
        show_images_side_by_side(
            ["Orijinal", "Taranmış"],
            [original, result]
        )
    
    return result


def create_test_image():
    """
    Test amacıyla sentetik bir evrak görüntüsü oluşturur.
    Eğik/yamuk bir kağıt simüle eder.
    
    Returns:
        Test görüntüsünün dosya yolu
    """
    print("\n[i] Test görüntüsü oluşturuluyor...")
    
    # Arka plan oluştur (gri tonlu masa/zemin)
    bg = np.ones((800, 1000, 3), dtype=np.uint8) * 180
    
    # Rastgele gürültü ekle (gerçekçi zemin)
    noise = np.random.randint(0, 30, bg.shape, dtype=np.uint8)
    bg = cv2.add(bg, noise)
    
    # Evrak köşe noktaları (yamuk/trapezoid şekilde)
    doc_pts = np.array([
        [150, 100],   # Sol-üst
        [800, 80],    # Sağ-üst
        [850, 650],   # Sağ-alt
        [120, 700]    # Sol-alt
    ], dtype=np.int32)
    
    # Beyaz kağıt çiz
    cv2.fillConvexPoly(bg, doc_pts, (245, 245, 245))
    
    # Kağıdın üzerine metin simüle et (yatay çizgiler)
    # Perspektif dönüşüm ile çizgileri yamuk kağıda yerleştir
    src_pts = np.array([[0, 0], [600, 0], [600, 500], [0, 500]], dtype=np.float32)
    dst_pts = doc_pts.astype(np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    for y in range(50, 480, 30):
        # Her satır bir metin satırı
        pt1 = np.array([[[50, y]]], dtype=np.float32)
        pt2 = np.array([[[550, y]]], dtype=np.float32)
        
        pt1_transformed = cv2.perspectiveTransform(pt1, M)[0][0]
        pt2_transformed = cv2.perspectiveTransform(pt2, M)[0][0]
        
        cv2.line(bg,
                 tuple(pt1_transformed.astype(int)),
                 tuple(pt2_transformed.astype(int)),
                 (60, 60, 60), 1)
    
    # Başlık simüle et
    pt_title1 = np.array([[[150, 30]]], dtype=np.float32)
    pt_title2 = np.array([[[450, 30]]], dtype=np.float32)
    pt1_t = cv2.perspectiveTransform(pt_title1, M)[0][0]
    pt2_t = cv2.perspectiveTransform(pt_title2, M)[0][0]
    cv2.line(bg, tuple(pt1_t.astype(int)), tuple(pt2_t.astype(int)), (30, 30, 30), 3)
    
    # Kaydet
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_images")
    os.makedirs(test_dir, exist_ok=True)
    test_path = os.path.join(test_dir, "test_dokuman.jpg")
    cv2.imwrite(test_path, bg)
    print(f"[✓] Test görüntüsü oluşturuldu: {test_path}")
    
    return test_path


def main():
    parser = argparse.ArgumentParser(
        description="Bölüm 1: Doküman Tarama ve Düzeltme",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python dokuman_tarama.py --input evrak.jpg
  python dokuman_tarama.py --input evrak.jpg --output taranmis.jpg --debug
  python dokuman_tarama.py --test
        """
    )
    parser.add_argument("--input", "-i", type=str, help="Giriş görüntüsü yolu")
    parser.add_argument("--output", "-o", type=str, help="Çıkış görüntüsü yolu")
    parser.add_argument("--debug", "-d", action="store_true", help="Ara adımları göster")
    parser.add_argument("--test", "-t", action="store_true", help="Sentetik test görüntüsü ile çalıştır")
    parser.add_argument("--no-whiten", action="store_true", help="Zemin beyazlaştırmayı kapat")
    
    args = parser.parse_args()
    
    if args.test:
        # Sentetik test görüntüsü oluştur ve işle
        test_path = create_test_image()
        result = scan_document(test_path, debug=args.debug, whiten=not args.no_whiten)
    elif args.input:
        if not os.path.exists(args.input):
            print(f"[✗] Hata: Dosya bulunamadı: {args.input}")
            sys.exit(1)
        result = scan_document(
            args.input, 
            output_path=args.output,
            debug=args.debug,
            whiten=not args.no_whiten
        )
    else:
        parser.print_help()
        print("\n[i] Hızlı test için: python dokuman_tarama.py --test")
        sys.exit(0)
    
    if result is not None:
        print("\n[✓] Bölüm 1 başarıyla tamamlandı!")
    else:
        print("\n[✗] İşlem başarısız oldu.")
        sys.exit(1)


if __name__ == "__main__":
    main()
