"""
Cevap Anahtarı ve Baloncuk Okuyucu
====================================
OMR formundaki işaretlenmiş baloncukları okur.

Teknik:
- Pozisyon tabanlı okuma (şablon pozisyonlarını kullanır)
- Dairesel maske ile piksel sayma
- Doluluk oranı > %25 ise işaretli kabul edilir
"""

import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from omr_template import (
    TEMPLATE_WIDTH, TEMPLATE_HEIGHT, BUBBLE_RADIUS,
    STUDENT_ID_START_X, STUDENT_ID_START_Y,
    STUDENT_ID_DIGITS, STUDENT_ID_DIGIT_SPACING, STUDENT_ID_ROW_SPACING,
    ANSWERS_START_Y, ANSWERS_COL1_X, ANSWERS_COL2_X,
    QUESTIONS_PER_COL, BUBBLE_SPACING_X, BUBBLE_SPACING_Y,
    OPTIONS, MARKER_OFFSET, MARKER_SIZE
)
from utils import (
    order_points, four_point_transform,
    find_document_contour, save_image
)

# Doluluk eşiği — bu oranın üzeri "işaretli" sayılır
FILL_THRESHOLD = 0.25


def preprocess_omr_image(image):
    """
    OMR form görüntüsünü ön işlemden geçirir:
    1. Perspektif düzeltme (sadece gerekiyorsa)
    2. Şablon boyutuna yeniden boyutlandırma
    3. Grayscale + Otsu threshold

    Args:
        image: Giriş görüntüsü (BGR)

    Returns:
        (warped_color, warped_thresh) — renkli ve binary görüntüler
    """
    h, w = image.shape[:2]
    img_area = h * w

    # Görüntü zaten şablon boyutunda mı kontrol et
    size_ratio = abs(w / TEMPLATE_WIDTH - 1) + abs(h / TEMPLATE_HEIGHT - 1)

    if size_ratio > 0.1:
        # Farklı boyut — perspektif düzeltme dene
        contour = find_document_contour(image)
        if contour is not None:
            contour_area = cv2.contourArea(contour)
            # Kontur alanı görüntünün en az %30'u olmalı (yanlış kontur filtresi)
            if contour_area > img_area * 0.3:
                pts = contour.reshape(4, 2).astype("float32")
                warped = four_point_transform(image, pts)
            else:
                warped = image.copy()
        else:
            warped = image.copy()
    else:
        # Zaten doğru boyutta — perspektif düzeltme gereksiz
        warped = image.copy()

    # Şablon boyutuna yeniden boyutlandır
    warped = cv2.resize(warped, (TEMPLATE_WIDTH, TEMPLATE_HEIGHT))

    # Pembe rengi filtrelemek için Kırmızı (Red) kanalını kullan
    # BGR formatında Kırmızı kanal index 2'dir.
    # Pembe rengin kırmızı bileşeni yüksek (255) olduğu için beyaz görünür,
    # kurşun kalem (koyu gri/siyah) ise düşük (40) olduğu için siyah kalır.
    red_channel = warped[:, :, 2]

    # Otsu threshold — pembe çizgiler beyaz olduğu için sadece koyu işaretler kalır
    _, thresh = cv2.threshold(
        red_channel, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    return warped, thresh


def read_bubble(thresh, cx, cy, radius=None):
    """
    Tek bir baloncuğun doluluk oranını hesaplar.

    Args:
        thresh: Binary (ters) görüntü
        cx, cy: Baloncuk merkez koordinatları
        radius: Baloncuk yarıçapı

    Returns:
        Doluluk oranı (0.0 - 1.0)
    """
    if radius is None:
        radius = BUBBLE_RADIUS

    h, w = thresh.shape
    r = radius - 1
    
    # Koordinatları int'e çevir
    cx = int(cx)
    cy = int(cy)

    # ROI sınırları
    x1 = max(0, cx - r)
    y1 = max(0, cy - r)
    x2 = min(w, cx + r)
    y2 = min(h, cy + r)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    roi = thresh[y1:y2, x1:x2]

    # Dairesel maske
    mask = np.zeros_like(roi)
    center = (cx - x1, cy - y1)
    cv2.circle(mask, center, r, 255, -1)

    # Maskelenmiş alandaki beyaz piksel sayısı
    masked = cv2.bitwise_and(roi, mask)
    filled = cv2.countNonZero(masked)
    total = cv2.countNonZero(mask)

    if total == 0:
        return 0.0

    return filled / total


def read_student_id(thresh):
    """
    Öğrenci numarasını okur.

    Args:
        thresh: Binary görüntü

    Returns:
        Öğrenci numarası (str, 5 karakter)
    """
    result = ""

    for digit_idx in range(STUDENT_ID_DIGITS):
        x = STUDENT_ID_START_X + int(digit_idx * STUDENT_ID_DIGIT_SPACING)
        best_number = -1
        best_ratio = 0.0

        for number in range(10):
            y = STUDENT_ID_START_Y + int(number * STUDENT_ID_ROW_SPACING)
            ratio = read_bubble(thresh, x, y)

            if ratio > best_ratio:
                best_ratio = ratio
                best_number = number

        if best_ratio > FILL_THRESHOLD:
            result += str(best_number)
        else:
            result += "?"

    return result


def read_answers(thresh, num_questions=20):
    """
    Cevapları okur.

    Args:
        thresh: Binary görüntü
        num_questions: Okunacak soru sayısı (max 60)

    Returns:
        {soru_no: 'A'/'B'/'C'/'D'/None}
    """
    num_questions = min(num_questions, 60)
    answers = {}

    for q in range(1, num_questions + 1):
        # Kolon ve satır hesapla
        if q <= QUESTIONS_PER_COL:
            col_x = ANSWERS_COL1_X
            row = q - 1
        else:
            col_x = ANSWERS_COL2_X
            row = q - QUESTIONS_PER_COL - 1

        y = ANSWERS_START_Y + int(row * BUBBLE_SPACING_Y)

        best_option = None
        best_ratio = 0.0

        for opt_idx, option in enumerate(OPTIONS):
            bx = col_x + int(opt_idx * BUBBLE_SPACING_X)
            ratio = read_bubble(thresh, bx, y)

            if ratio > best_ratio:
                best_ratio = ratio
                best_option = option

        if best_ratio > FILL_THRESHOLD:
            answers[q] = best_option
        else:
            answers[q] = None  # Boş bırakılmış

    return answers


def scan_answer_key(image_path, num_questions=20, debug=False):
    """
    Cevap anahtarı görüntüsünü tarar.

    Args:
        image_path: Cevap anahtarı görüntü yolu
        num_questions: Soru sayısı
        debug: Debug modu

    Returns:
        {soru_no: seçenek} sözlüğü
    """
    print(f"\n[i] Cevap anahtarı taranıyor: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        print(f"[✗] Hata: Görüntü yüklenemedi: {image_path}")
        return None

    warped, thresh = preprocess_omr_image(image)
    answers = read_answers(thresh, num_questions)

    # Sonuçları göster
    print("[✓] Cevap anahtarı okundu:")
    blank_count = 0
    for q in range(1, num_questions + 1):
        ans = answers.get(q, None)
        if ans:
            print(f"    Soru {q:2d}: {ans}")
        else:
            print(f"    Soru {q:2d}: (boş)")
            blank_count += 1

    if blank_count > num_questions * 0.5:
        print(f"\n[!] UYARI: Soruların %{blank_count/num_questions*100:.0f}'ü boş!")
        print("    Tarama kalitesini kontrol edin.")

    return answers


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Cevap Anahtarı Okuyucu")
    parser.add_argument("--input", "-i", type=str, required=True)
    parser.add_argument("--questions", "-q", type=int, default=20)
    parser.add_argument("--debug", "-d", action="store_true")
    args = parser.parse_args()

    result = scan_answer_key(args.input, args.questions, args.debug)
    if result:
        print(f"\nToplam: {sum(1 for v in result.values() if v)} cevap okundu")
