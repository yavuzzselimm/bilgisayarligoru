"""
BÖLÜM 2: Optik Okuyucu (OMR) — Ana Pipeline
=============================================
Cevap anahtarı ve öğrenci kağıtlarını değerlendirir.

Özellikler:
- Ayarlanabilir soru sayısı (20, 30, 40, 45, 60 vb.)
- 4 şık (A, B, C, D)
- Öğrenci numarası otomatik okuma
- CSV ve detaylı rapor çıktısı
- Sentetik test modu

Kullanım:
    python optik_okuyucu.py --test --questions 20
    python optik_okuyucu.py --test --questions 40 --students 15
    python optik_okuyucu.py -a cevap.png -s ogrenciler/ -q 30
"""

import cv2
import numpy as np
import argparse
import os
import sys
import csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from omr_template import (
    create_omr_template, fill_template, generate_test_data, OPTIONS
)
from cevap_anahtari import (
    preprocess_omr_image, read_student_id, read_answers, scan_answer_key
)


def grade_student(image_path, answer_key, num_questions=20, debug=False):
    """
    Tek bir öğrenci kağıdını değerlendirir.

    Args:
        image_path: Öğrenci kağıdı görüntü yolu
        answer_key: {soru_no: doğru_cevap} sözlüğü
        num_questions: Soru sayısı
        debug: Debug modu

    Returns:
        Sonuç sözlüğü veya None
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"    [✗] Görüntü yüklenemedi: {image_path}")
        return None

    # Ön işleme
    warped, thresh = preprocess_omr_image(image)

    # Öğrenci numarası oku
    student_id = read_student_id(thresh)

    # Cevapları oku
    student_answers = read_answers(thresh, num_questions)

    # Değerlendir
    correct = 0
    wrong = 0
    blank = 0
    details = []

    for q in range(1, num_questions + 1):
        student_ans = student_answers.get(q, None)
        correct_ans = answer_key.get(q, None)

        if student_ans is None:
            blank += 1
            result_str = "-"
        elif student_ans == correct_ans:
            correct += 1
            result_str = f"{student_ans}(✓)"
        else:
            wrong += 1
            result_str = f"{student_ans}(✗ {correct_ans})"

        details.append((q, student_ans, correct_ans, result_str))

    score = (correct / num_questions) * 100 if num_questions > 0 else 0

    return {
        "student_id": student_id,
        "answers": student_answers,
        "correct": correct,
        "wrong": wrong,
        "blank": blank,
        "score": score,
        "details": details,
        "file": os.path.basename(image_path)
    }


def grade_all_students(students_path, answer_key, num_questions=20,
                       debug=False):
    """
    Bir dizindeki tüm öğrenci kağıtlarını değerlendirir.

    Args:
        students_path: Öğrenci kağıtları dizini
        answer_key: Cevap anahtarı
        num_questions: Soru sayısı
        debug: Debug modu

    Returns:
        Sonuç listesi
    """
    # Görüntü dosyalarını bul
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    files = sorted([
        f for f in os.listdir(students_path)
        if os.path.splitext(f)[1].lower() in extensions
    ])

    if not files:
        print("[✗] Öğrenci kağıdı bulunamadı!")
        return []

    print(f"\n[i] {len(files)} öğrenci kağıdı bulundu.\n")

    results = []
    for idx, filename in enumerate(files, 1):
        filepath = os.path.join(students_path, filename)
        print(f"  [{idx}/{len(files)}] Değerlendiriliyor: {filename}")

        result = grade_student(filepath, answer_key, num_questions, debug)
        if result:
            print(f"    Öğrenci No: {result['student_id']}")
            print(f"    Doğru: {result['correct']} | "
                  f"Yanlış: {result['wrong']} | Boş: {result['blank']}")
            print(f"    Puan: {result['score']:.1f}/100\n")
            results.append(result)

    return results


def print_results_table(results, num_questions=20):
    """Sonuç tablosunu yazdırır."""
    print("=" * 80)
    print("  DEĞERLENDİRME SONUÇLARI")
    print("=" * 80)

    header = (f"  {'#':>3} | {'Öğrenci No':<16} | {'Doğru':>6} | "
              f"{'Yanlış':>7} | {'Boş':>4} | {'Puan':>8} | {'Dosya':<25}")
    print(header)
    print("-" * 80)

    total_correct = 0
    total_wrong = 0
    total_blank = 0
    scores = []

    for idx, r in enumerate(results, 1):
        print(f"  {idx:>3} | {r['student_id']:<16} | {r['correct']:>6} | "
              f"{r['wrong']:>7} | {r['blank']:>4} | {r['score']:>7.1f} | "
              f"{r['file']:<25}")
        total_correct += r['correct']
        total_wrong += r['wrong']
        total_blank += r['blank']
        scores.append(r['score'])

    n = len(results)
    if n > 0:
        print("-" * 80)
        print(f"  {'ORT':>3} | {'':16} | {total_correct/n:>6.1f} | "
              f"{total_wrong/n:>7.1f} | {total_blank/n:>4.1f} | "
              f"{np.mean(scores):>7.1f} |")
        print("=" * 80)

        print(f"\n  📊 İstatistikler:")
        print(f"     Toplam öğrenci  : {n}")
        print(f"     Soru sayısı     : {num_questions}")
        print(f"     En yüksek puan  : {max(scores):.1f}")
        print(f"     En düşük puan   : {min(scores):.1f}")
        print(f"     Ortalama puan   : {np.mean(scores):.1f}")
        print(f"     Std. sapma      : {np.std(scores):.1f}")
        passing = sum(1 for s in scores if s >= 50)
        print(f"     Geçen (≥50)     : {passing} ({passing/n*100:.0f}%)")
        print(f"     Kalan (<50)     : {n-passing} ({(n-passing)/n*100:.0f}%)")


def export_results_csv(results, output_path, num_questions=20):
    """Sonuçları CSV dosyasına aktarır."""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Başlık satırı
        headers = ['Sıra', 'Öğrenci No', 'Doğru', 'Yanlış', 'Boş',
                    'Puan', 'Dosya']
        for q in range(1, num_questions + 1):
            headers.append(f'S{q}')
        writer.writerow(headers)

        # Veri satırları
        for idx, r in enumerate(results, 1):
            row = [idx, r['student_id'], r['correct'], r['wrong'],
                   r['blank'], r['score'], r['file']]
            for q, _, _, result_str in r['details']:
                row.append(result_str)
            writer.writerow(row)

    print(f"[✓] Sonuçlar CSV'ye kaydedildi: {output_path}")


def export_results_detailed(results, output_path, num_questions=20):
    """Detaylı metin raporu oluşturur."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("  DETAYLI SINAV RAPORU\n")
        f.write(f"  Soru Sayısı: {num_questions}\n")
        f.write("=" * 60 + "\n\n")

        for idx, r in enumerate(results, 1):
            f.write(f"--- Öğrenci {idx} ---\n")
            f.write(f"Öğrenci No : {r['student_id']}\n")
            f.write(f"Dosya      : {r['file']}\n")
            f.write(f"Doğru      : {r['correct']}\n")
            f.write(f"Yanlış     : {r['wrong']}\n")
            f.write(f"Boş        : {r['blank']}\n")
            f.write(f"Puan       : {r['score']:.1f}/100\n\n")

            f.write("Soru | Cevap | Doğru | Sonuç\n")
            f.write("-" * 35 + "\n")
            for q, s_ans, c_ans, result_str in r['details']:
                s_str = s_ans if s_ans else "-"
                c_str = c_ans if c_ans else "-"
                f.write(f"  {q:>2} |   {s_str:>1}   |   {c_str:>1}   | "
                        f"{result_str}\n")
            f.write("\n")

        # Genel istatistikler
        if results:
            scores = [r['score'] for r in results]
            f.write("=" * 60 + "\n")
            f.write("  GENEL İSTATİSTİKLER\n")
            f.write("=" * 60 + "\n")
            f.write(f"Toplam öğrenci : {len(results)}\n")
            f.write(f"Soru sayısı    : {num_questions}\n")
            f.write(f"En yüksek      : {max(scores):.1f}\n")
            f.write(f"En düşük       : {min(scores):.1f}\n")
            f.write(f"Ortalama       : {np.mean(scores):.1f}\n")
            f.write(f"Std. sapma     : {np.std(scores):.1f}\n")

    print(f"[✓] Detaylı rapor kaydedildi: {output_path}")


def run_full_test(num_questions=20, num_students=10):
    """
    Tam test pipeline'ı:
    1. Şablon oluştur
    2. Cevap anahtarı + öğrenci kağıtları oluştur
    3. Cevap anahtarını tara
    4. Tüm öğrencileri değerlendir
    5. Sonuçları karşılaştır
    """
    print(f"\n{'='*60}")
    print(f"  BÖLÜM 2: OPTİK OKUYUCU — TAM TEST")
    print(f"  Soru sayısı: {num_questions} | Öğrenci: {num_students}")
    print(f"{'='*60}")

    # 1-3. Sentetik veri oluştur
    print(f"\n[1/5] OMR şablonu ve test verileri oluşturuluyor...")
    answer_key_expected, students_data, output_dir = generate_test_data(
        num_questions=num_questions,
        num_students=num_students
    )

    # 4. Cevap anahtarını tara
    print(f"\n[4/5] Cevap anahtarı taranıyor...")
    ak_path = os.path.join(output_dir, "cevap_anahtari.png")
    answer_key_scanned = scan_answer_key(ak_path, num_questions)

    if answer_key_scanned is None:
        print("[✗] Cevap anahtarı okunamadı!")
        return

    # Cevap anahtarı doğruluk kontrolü
    ak_correct = sum(
        1 for q in range(1, num_questions + 1)
        if answer_key_scanned.get(q) == answer_key_expected.get(q)
    )
    print(f"    Cevap anahtarı doğruluk: {ak_correct}/{num_questions}")

    # 5. Öğrenci kağıtlarını değerlendir
    print(f"\n[5/5] Öğrenci kağıtları değerlendiriliyor...")
    students_dir = os.path.join(output_dir, "ogrenci_kagitlari")
    results = grade_all_students(
        students_dir, answer_key_scanned, num_questions
    )

    if not results:
        print("[✗] Sonuç yok!")
        return

    # Tabloyu yazdır
    print_results_table(results, num_questions)

    # CSV ve rapor kaydet
    csv_path = os.path.join(output_dir, "sonuclar.csv")
    export_results_csv(results, csv_path, num_questions)

    report_path = os.path.join(output_dir, "detayli_rapor.txt")
    export_results_detailed(results, report_path, num_questions)

    # Doğrulama: beklenen vs okunan
    print(f"\n{'='*60}")
    print(f"  DOĞRULAMA: Beklenen vs Okunan Sonuçlar")
    print(f"{'='*60}")

    all_match = True
    for student in students_data:
        sid = student["id"]
        expected_correct = student["correct"]
        expected_wrong = student["wrong"]
        expected_blank = student["blank"]

        # Okunan sonucu bul
        found = None
        for r in results:
            if r["student_id"] == sid:
                found = r
                break

        if found is None:
            print(f"  {sid}: ✗ BULUNAMADI")
            all_match = False
            continue

        if (found["correct"] == expected_correct and
                found["wrong"] == expected_wrong and
                found["blank"] == expected_blank):
            print(f"  {sid}: ✓ EŞLEŞME")
        else:
            print(f"  {sid}: ✗ UYUŞMAZLIK")
            print(f"    Beklenen: D={expected_correct} Y={expected_wrong} "
                  f"B={expected_blank}")
            print(f"    Okunan:   D={found['correct']} Y={found['wrong']} "
                  f"B={found['blank']}")
            all_match = False

    if all_match:
        print(f"\n  ✅ TÜM SONUÇLAR DOĞRU!")
    else:
        print(f"\n  ⚠️ BAZI SONUÇLAR UYUŞMUYOR!")


def main():
    parser = argparse.ArgumentParser(
        description="Bölüm 2: Optik Okuyucu (OMR)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  # Sentetik test (20 soru, 10 öğrenci):
  python optik_okuyucu.py --test

  # 40 soruluk test, 15 öğrenci:
  python optik_okuyucu.py --test --questions 40 --students 15

  # Gerçek kağıtlarla değerlendirme:
  python optik_okuyucu.py -a cevap.png -s ogrenciler/ -q 30
        """
    )
    parser.add_argument("--test", "-t", action="store_true",
                        help="Sentetik test ile çalıştır")
    parser.add_argument("--answer-key", "-a", type=str,
                        help="Cevap anahtarı görüntüsü")
    parser.add_argument("--students", "-s", type=str,
                        help="Öğrenci kağıtları dizini")
    parser.add_argument("--questions", "-q", type=int, default=20,
                        help="Soru sayısı (varsayılan: 20, max: 60)")
    parser.add_argument("--num-students", type=int, default=10,
                        help="Test için öğrenci sayısı (varsayılan: 10)")
    parser.add_argument("--output", "-o", type=str,
                        help="CSV çıkış dosyası")
    parser.add_argument("--debug", "-d", action="store_true",
                        help="Ara adımları göster")

    args = parser.parse_args()

    # Soru sayısı kontrolü
    if args.questions < 1 or args.questions > 60:
        print("[✗] Soru sayısı 1-60 arasında olmalıdır!")
        sys.exit(1)

    if args.test:
        run_full_test(
            num_questions=args.questions,
            num_students=args.num_students
        )
    elif args.answer_key and args.students:
        if not os.path.exists(args.answer_key):
            print(f"[✗] Cevap anahtarı bulunamadı: {args.answer_key}")
            sys.exit(1)
        if not os.path.isdir(args.students):
            print(f"[✗] Öğrenci dizini bulunamadı: {args.students}")
            sys.exit(1)

        # Cevap anahtarını oku
        answer_key = scan_answer_key(
            args.answer_key, args.questions, args.debug
        )
        if answer_key is None:
            sys.exit(1)

        # Öğrencileri değerlendir
        results = grade_all_students(
            args.students, answer_key, args.questions, args.debug
        )

        if results:
            print_results_table(results, args.questions)

            csv_path = args.output or os.path.join(
                args.students, "sonuclar.csv"
            )
            export_results_csv(results, csv_path, args.questions)

            report_path = csv_path.replace(".csv", "_detayli_rapor.txt")
            export_results_detailed(results, report_path, args.questions)
    else:
        parser.print_help()
        print("\n[i] Hızlı test: python optik_okuyucu.py --test")
        print("[i] 40 soru:    python optik_okuyucu.py --test -q 40")
        sys.exit(0)


if __name__ == "__main__":
    main()
