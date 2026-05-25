"""
OMR Şablon Oluşturucu — Türk Optik Form Formatı
=================================================
Gerçek Türk optik okuyucu formlarını taklit eden şablon oluşturur.

Özellikler:
- 60 soruluk form (2 kolon: 1-30 sol, 31-60 sağ)
- 4 seçenek: A, B, C, D
- Öğrenci numarası alanı (8 hane)
- Pembe/magenta renk şeması
- Ayarlanabilir soru sayısı (kullanıcı 20, 30, 40, 60 vb. seçebilir)

Kullanım:
    from omr_template import create_omr_template, fill_template
"""

import cv2
import numpy as np
import os
import random

# ============================================================
#  ŞABLON SABİTLERİ
# ============================================================

# Form boyutları (Gerçek form boyutu)
TEMPLATE_WIDTH = 600
TEMPLATE_HEIGHT = 940

# Renkler (BGR formatı)
PINK = (147, 20, 255)          # Pembe (form çizgileri ve baloncuklar)
PINK_LIGHT = (200, 150, 255)   # Açık pembe (dolgu)
PINK_DARK = (100, 10, 200)     # Koyu pembe (başlıklar)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
DARK_FILL = (40, 40, 40)       # Kurşun kalem rengi (işaretleme)

# Baloncuk parametreleri
BUBBLE_RADIUS = 8
BUBBLE_SPACING_X = 18.5          # Seçenekler arası yatay boşluk
BUBBLE_SPACING_Y = 20.72         # Sorular arası dikey boşluk
OPTIONS = ['A', 'B', 'C', 'D']

# Öğrenci No alanı (Sağ üst)
STUDENT_ID_DIGITS = 5
STUDENT_ID_START_X = 504
STUDENT_ID_START_Y = 53
STUDENT_ID_DIGIT_SPACING = 18.5
STUDENT_ID_ROW_SPACING = 20.72

# Cevaplar alanı (Sağ alt)
ANSWERS_START_Y = 301
ANSWERS_COL1_X = 431           # Sol kolon (1-30) başlangıç
ANSWERS_COL2_X = 523           # Sağ kolon (31-60) başlangıç
QUESTIONS_PER_COL = 30

# Alignment markers
MARKER_SIZE = 15
MARKER_OFFSET = 10


def create_omr_template(num_questions=20, output_path=None):
    """
    Kullanıcının sağladığı gerçek optik okuyucu formunu yükler ve 
    üzerindeki baloncuk pozisyonlarını haritalar.

    Args:
        num_questions: Kullanılacak soru sayısı (max 60)
        output_path: Çıkış dosya yolu

    Returns:
        (template_image, positions_dict)
    """
    num_questions = min(num_questions, 60)

    # Gerçek formu yükle
    template_file = "/Users/yavuz/Desktop/7ff0a4b1-099d-4e8b-942c-3374cc8d3b7d.png"
    template = cv2.imread(template_file)
    if template is None:
        raise ValueError(f"Şablon resmi bulunamadı: {template_file}")

    positions = {
        "student_id": {},
        "questions": {},
        "bubble_radius": BUBBLE_RADIUS,
        "template_size": (TEMPLATE_WIDTH, TEMPLATE_HEIGHT),
        "num_questions": num_questions
    }

    # === Öğrenci No Pozisyonları ===
    x_start = STUDENT_ID_START_X
    y_start = STUDENT_ID_START_Y
    for digit_idx in range(STUDENT_ID_DIGITS):
        positions["student_id"][digit_idx] = {}
        x = x_start + int(digit_idx * STUDENT_ID_DIGIT_SPACING)
        for number in range(10):
            y = y_start + int(number * STUDENT_ID_ROW_SPACING)
            positions["student_id"][digit_idx][number] = (x, y)

    # === Cevap Baloncukları Pozisyonları ===
    for q in range(1, 61):
        if q <= QUESTIONS_PER_COL:
            col_x = ANSWERS_COL1_X
            row = q - 1
        else:
            col_x = ANSWERS_COL2_X
            row = q - QUESTIONS_PER_COL - 1

        y = ANSWERS_START_Y + int(row * BUBBLE_SPACING_Y)
        positions["questions"][q] = {}

        for opt_idx, option in enumerate(OPTIONS):
            bx = col_x + int(opt_idx * BUBBLE_SPACING_X)
            positions["questions"][q][option] = (bx, y)

    # Kaydet
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cv2.imwrite(output_path, template)

    return template, positions


def _draw_alignment_markers(template):
    """4 köşeye hizalama işaretçileri çizer."""
    h, w = template.shape[:2]
    m = MARKER_OFFSET
    s = MARKER_SIZE
    # Sol-üst
    cv2.rectangle(template, (m, m), (m + s, m + s), BLACK, -1)
    # Sağ-üst
    cv2.rectangle(template, (w - m - s, m), (w - m, m + s), BLACK, -1)
    # Sol-alt
    cv2.rectangle(template, (m, h - m - s), (m + s, h - m), BLACK, -1)
    # Sağ-alt
    cv2.rectangle(template, (w - m - s, h - m - s), (w - m, h - m), BLACK, -1)


def _draw_student_id_area(template, positions):
    """Öğrenci numarası alanını çizer."""
    x_start = STUDENT_ID_START_X
    y_start = STUDENT_ID_START_Y

    # Başlık
    cv2.rectangle(template, (x_start - 5, y_start - 25),
                  (x_start + STUDENT_ID_DIGITS * STUDENT_ID_DIGIT_SPACING + 5,
                   y_start - 5), PINK, -1)
    cv2.putText(template, "OGRENCI NO.",
                (x_start + 15, y_start - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)

    for digit_idx in range(STUDENT_ID_DIGITS):
        positions["student_id"][digit_idx] = {}
        x = x_start + digit_idx * STUDENT_ID_DIGIT_SPACING + 13

        for number in range(10):
            y = y_start + number * STUDENT_ID_ROW_SPACING + 10

            # Baloncuk çiz
            cv2.circle(template, (x, y), BUBBLE_RADIUS - 2, PINK, 1)

            # Sayı etiketi (baloncuğun içine küçük)
            label = str(number)
            cv2.putText(template, label,
                        (x - 4, y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, PINK_LIGHT, 1)

            positions["student_id"][digit_idx][number] = (x, y)


def _draw_answer_bubbles(template, positions, num_questions):
    """Cevap baloncuklarını çizer."""
    for q in range(1, 61):
        if q <= QUESTIONS_PER_COL:
            col_x = ANSWERS_COL1_X
            row = q - 1
        else:
            col_x = ANSWERS_COL2_X
            row = q - QUESTIONS_PER_COL - 1

        y = ANSWERS_START_Y + row * BUBBLE_SPACING_Y + 10

        # Aktif sorular normal, inaktif sorular soluk
        is_active = (q <= num_questions)

        # Soru numarası
        num_color = PINK_DARK if is_active else (220, 200, 240)
        cv2.putText(template, str(q),
                    (col_x - 30, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, num_color, 1)

        positions["questions"][q] = {}

        for opt_idx, option in enumerate(OPTIONS):
            bx = col_x + opt_idx * BUBBLE_SPACING_X

            # Baloncuk
            bubble_color = PINK if is_active else (230, 210, 245)
            thickness = 1 if is_active else 1
            cv2.circle(template, (bx, y), BUBBLE_RADIUS, bubble_color, thickness)

            # Seçenek etiketi
            label_color = PINK if is_active else (230, 210, 245)
            cv2.putText(template, option,
                        (bx - 5, y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, label_color, 1)

            positions["questions"][q][option] = (bx, y)

        # Aktif soru satırları arasında hafif çizgi
        if is_active and q < 60:
            line_y = y + BUBBLE_SPACING_Y // 2 + 2
            if q <= QUESTIONS_PER_COL:
                cv2.line(template, (col_x - 35, line_y),
                         (col_x + 4 * BUBBLE_SPACING_X, line_y),
                         (240, 230, 250), 1)


def fill_template(template, student_id, answers, positions,
                   fill_color=None):
    """
    Boş şablon üzerine cevapları ve öğrenci numarasını doldurur.

    Args:
        template: Boş şablon görüntüsü
        student_id: Öğrenci numarası (str, 8 hane)
        answers: {soru_no: 'A'/'B'/'C'/'D' veya None}
        positions: Şablondaki pozisyon bilgileri
        fill_color: İşaretleme rengi (None ise kurşun kalem)

    Returns:
        Doldurulmuş görüntü
    """
    filled = template.copy()
    if fill_color is None:
        fill_color = DARK_FILL

    bubble_r = positions.get("bubble_radius", BUBBLE_RADIUS)

    # Öğrenci numarasını doldur
    student_id_str = str(student_id).zfill(STUDENT_ID_DIGITS)
    for digit_idx, char in enumerate(student_id_str):
        if char.isdigit():
            number = int(char)
            if digit_idx in positions["student_id"]:
                if number in positions["student_id"][digit_idx]:
                    cx, cy = positions["student_id"][digit_idx][number]
                    cv2.circle(filled, (cx, cy), bubble_r - 3, fill_color, -1)

    # Cevapları doldur
    for q_num, option in answers.items():
        if option is None:
            continue
        q_num = int(q_num)
        if q_num in positions["questions"]:
            if option in positions["questions"][q_num]:
                cx, cy = positions["questions"][q_num][option]
                cv2.circle(filled, (cx, cy), bubble_r - 3, fill_color, -1)

    return filled


def generate_test_data(num_questions=20, num_students=10, output_dir=None):
    """
    Sentetik test verileri oluşturur: 1 cevap anahtarı + N öğrenci kağıdı.

    Args:
        num_questions: Soru sayısı (max 60)
        num_students: Öğrenci sayısı
        output_dir: Çıkış dizini

    Returns:
        (answer_key_dict, students_data, output_dir)
    """
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "test_images"
        )
    os.makedirs(output_dir, exist_ok=True)
    students_dir = os.path.join(output_dir, "ogrenci_kagitlari")
    os.makedirs(students_dir, exist_ok=True)

    num_questions = min(num_questions, 60)

    # 1. Boş şablon oluştur
    template_path = os.path.join(output_dir, "omr_template.png")
    template, positions = create_omr_template(num_questions, template_path)
    print(f"[✓] OMR şablonu oluşturuldu: {template_path}")
    print(f"    Boyut: {TEMPLATE_WIDTH}x{TEMPLATE_HEIGHT}")
    print(f"    Soru sayısı: {num_questions}")

    # 2. Cevap anahtarı oluştur
    random.seed(42)
    answer_key = {}
    for q in range(1, num_questions + 1):
        answer_key[q] = random.choice(OPTIONS)

    answer_key_path = os.path.join(output_dir, "cevap_anahtari.png")
    ak_image = fill_template(template, "00000000", answer_key, positions)

    # Cevap anahtarına "CEVAP ANAHTARI" etiketi ekle
    cv2.putText(ak_image, "CEVAP ANAHTARI",
                (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)

    # Hafif gürültü ekle (gerçekçilik)
    noise = np.random.randint(0, 8, ak_image.shape, dtype=np.uint8)
    ak_image = cv2.add(ak_image, noise)
    cv2.imwrite(answer_key_path, ak_image)
    print(f"    Cevap anahtarı oluşturuldu: {answer_key_path}")
    print(f"    Gerçek cevaplar: {answer_key}")

    # 3. Öğrenci kağıtları oluştur
    print(f"\n[3/5] {num_students} öğrenci kağıdı oluşturuluyor...")
    students_data = []

    for i in range(num_students):
        if i == 0:
            # Kullanıcının istediği özel öğrenci
            student_id = "11111"
            student_answers = {}
            for q in range(1, num_questions + 1):
                student_answers[q] = OPTIONS[(q - 1) % len(OPTIONS)]  # A, B, C, D, A, B...
            
            # Puan ve sayıları hesapla
            expected_correct = sum(1 for q in range(1, num_questions + 1) if student_answers[q] == answer_key.get(q))
            expected_blank = sum(1 for q in range(1, num_questions + 1) if student_answers[q] is None)
            expected_wrong = num_questions - expected_correct - expected_blank
            score = int((expected_correct / num_questions) * 100)
            print(f"    {student_id} (Özel Desen): D={expected_correct} Y={expected_wrong} B={expected_blank} P={score}")
        else:
            # Rastgele öğrenci
            student_id = f"{20000 + i:05d}"
            correct_rate = random.uniform(0.55, 0.90)
            blank_rate = random.uniform(0.02, 0.12)

            student_answers = {}
            expected_correct = 0
            expected_wrong = 0
            expected_blank = 0

            for q in range(1, num_questions + 1):
                r = random.random()
                if r < blank_rate:
                    student_answers[q] = None
                    expected_blank += 1
                elif r < blank_rate + (1 - blank_rate) * correct_rate:
                    student_answers[q] = answer_key[q]
                    expected_correct += 1
                else:
                    wrong_options = [o for o in OPTIONS if o != answer_key[q]]
                    student_answers[q] = random.choice(wrong_options)
                    expected_wrong += 1

            score = int((expected_correct / num_questions) * 100)
            print(f"    {student_id}: D={expected_correct} Y={expected_wrong} B={expected_blank} P={score}")

        # Formu doldur
        student_image = fill_template(
            template, student_id, student_answers, positions
        )

        # Gürültü ekle
        noise = np.random.randint(0, 6, student_image.shape, dtype=np.uint8)
        student_image = cv2.add(student_image, noise)

        # Kaydet
        student_path = os.path.join(
            students_dir, f"ogrenci_{student_id}.png"
        )
        cv2.imwrite(student_path, student_image)

        students_data.append({
            "id": student_id,
            "answers": student_answers,
            "correct": expected_correct,
            "wrong": expected_wrong,
            "blank": expected_blank,
            "score": score,
            "path": student_path
        })

    return answer_key, students_data, output_dir


if __name__ == "__main__":
    print("OMR Şablon Oluşturucu")
    print("=" * 40)
    template, positions = create_omr_template(20, "test_template.png")
    print(f"Şablon oluşturuldu: test_template.png")
    print(f"Soru pozisyonları: {len(positions['questions'])} soru")
    print(f"Öğrenci no: {len(positions['student_id'])} hane")
