"""
BOLUM 3: Cok Taneli Madde Sayma
================================
Beyaz zemin uzerinde cekilmis taneli maddelerin (misir, pirinc,
kuruyemis vb.) sayisini bulur.

Teknik: Watershed segmentasyon + kontur analizi

Kullanim:
    python tane_sayma.py --input taneler.jpg
    python tane_sayma.py --test
"""

import cv2
import numpy as np
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import show_image, save_image, create_colormap


def preprocess(image, debug=False):
    """
    On isleme: grayscale, blur, threshold, morfolojik temizleme.

    Args:
        image: Giris goruntusu (BGR)
        debug: Ara adimlari goster

    Returns:
        (gray, binary) - Grayscale ve binary goruntuler
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    # Otsu threshold
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Morfolojik temizleme
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    if debug:
        show_image("1. Binary (Otsu)", binary)

    return gray, binary


def watershed_segment(image, binary, debug=False):
    """
    Watershed segmentasyon ile yapistik taneleri ayirir.

    Adimlar:
    1. Distance transform
    2. Sure foreground belirleme
    3. Sure background belirleme
    4. Unknown bolge hesaplama
    5. Marker-based watershed

    Args:
        image: Orijinal BGR goruntu
        binary: Binary goruntu
        debug: Ara adimlari goster

    Returns:
        markers - Etiketlenmis goruntu (her tane farkli etiket)
    """
    # Distance transform
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    # Normalize et
    dist_norm = cv2.normalize(
        dist_transform, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    if debug:
        show_image("2. Distance Transform", dist_norm)

    # Sure foreground: mesafe haritasinin %30'undan buyuk bolge
    # Dusuk deger = daha fazla ayrim (yapisik taneler icin onemli)
    _, sure_fg = cv2.threshold(
        dist_transform, 0.3 * dist_transform.max(), 255, 0
    )
    sure_fg = sure_fg.astype(np.uint8)

    # Sure background: binary goruntuyu dilate et
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sure_bg = cv2.dilate(binary, kernel, iterations=3)

    # Unknown bolge
    unknown = cv2.subtract(sure_bg, sure_fg)

    if debug:
        show_image("3. Sure Foreground", sure_fg)
        show_image("4. Unknown Region", unknown)

    # Marker'lari olustur
    num_labels, markers = cv2.connectedComponents(sure_fg)

    # Arka plan 0 degil, 1 olsun (watershed icin)
    markers = markers + 1

    # Unknown bolge 0 olsun
    markers[unknown == 255] = 0

    # Watershed uygula
    markers = cv2.watershed(image, markers)

    if debug:
        # Marker gorselestirme
        marker_vis = np.zeros_like(image)
        for label in range(2, num_labels + 1):
            mask = (markers == label).astype(np.uint8) * 255
            color = np.random.randint(50, 255, 3).tolist()
            marker_vis[markers == label] = color
        show_image("5. Watershed Sonucu", marker_vis)

    return markers, num_labels


def count_and_label(image, markers, num_labels,
                    min_area=50, max_area=None, debug=False):
    """
    Watershed sonucundaki taneleri sayar ve numaralandirir.

    Args:
        image: Orijinal goruntu
        markers: Watershed etiketleri
        num_labels: Toplam etiket sayisi
        min_area: Minimum alan (gurultu filtresi)
        max_area: Maksimum alan (None ise limit yok)
        debug: Ara adimlari goster

    Returns:
        (result_image, count, areas) - Numaralandirilmis goruntu,
        tane sayisi, alan listesi
    """
    result = image.copy()
    count = 0
    areas = []

    if max_area is None:
        img_area = image.shape[0] * image.shape[1]
        max_area = img_area * 0.3

    colors = create_colormap(max(num_labels, 1))

    for label in range(2, num_labels + 1):
        mask = np.zeros(markers.shape, dtype=np.uint8)
        mask[markers == label] = 255

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < min_area or area > max_area:
                continue

            count += 1
            areas.append(area)

            color = colors[count % len(colors)]
            cv2.drawContours(result, [contour], -1, color, 2)

            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                cv2.putText(
                    result, str(count), (cx - 10, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 0, 255), 1
                )

    if debug:
        show_image("6. Numaralandirilmis Taneler", result)

    return result, count, areas


def count_objects(image_path, output_path=None, min_area=50,
                  max_area=None, debug=False):
    """
    Ana pipeline: taneli maddeleri say.

    Args:
        image_path: Giris goruntusunun yolu
        output_path: Cikis dosya yolu
        min_area: Minimum tane alani
        max_area: Maksimum tane alani
        debug: Ara adimlari goster

    Returns:
        (count, result_image)
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"[HATA] Goruntu yuklenemedi: {image_path}")
        return 0, None

    if debug:
        show_image("0. Orijinal", image)

    # On isleme
    gray, binary = preprocess(image, debug=debug)

    # Watershed segmentasyon
    markers, num_labels = watershed_segment(image, binary, debug=debug)

    # Sayma ve etiketleme
    result, count, areas = count_and_label(
        image, markers, num_labels,
        min_area=min_area, max_area=max_area, debug=debug
    )

    # Sonuc bilgisi
    cv2.putText(
        result, f"Toplam: {count} tane",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (0, 0, 255), 2
    )

    # Kaydet
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_sayilmis{ext}"

    save_image(result, output_path)

    # Sade Konsol Ciktisi
    print(f"[{os.path.basename(image_path)}] --> Sayilan Obje: {count} (min_area: {min_area})")

    return count, result


def create_test_image(num_objects=50, obj_type="circle"):
    """
    Test amacıyla sentetik tane goruntusu olusturur.

    Args:
        num_objects: Tane sayisi
        obj_type: "circle" veya "ellipse"

    Returns:
        Test goruntusunun dosya yolu
    """
    import random

    print(f"\n[i] Sentetik test goruntusu olusturuluyor ({num_objects} tane)...")

    width, height = 800, 600
    # Beyaz zemin
    image = np.ones((height, width, 3), dtype=np.uint8) * 245

    # Hafif zemin gurultusu
    noise = np.random.randint(0, 10, image.shape, dtype=np.uint8)
    image = cv2.add(image, noise)

    random.seed(42)
    placed = 0
    attempts = 0
    positions = []

    while placed < num_objects and attempts < num_objects * 10:
        attempts += 1

        if obj_type == "ellipse":
            rx = random.randint(8, 18)
            ry = random.randint(5, 12)
            angle = random.randint(0, 180)
        else:
            rx = random.randint(6, 14)
            ry = rx
            angle = 0

        cx = random.randint(rx + 5, width - rx - 5)
        cy = random.randint(ry + 5, height - ry - 5)

        # Cakisma kontrolu
        overlap = False
        for (px, py, pr) in positions:
            dist = np.sqrt((cx - px)**2 + (cy - py)**2)
            if dist < (max(rx, ry) + pr + 2):
                overlap = True
                break

        if overlap:
            # Yapisik tane olustur (bazen)
            if random.random() < 0.3:
                overlap = False

        if not overlap or random.random() < 0.15:
            # Tane rengi (kahverengi/bej tonlari)
            base_color = random.choice([
                (40, 80, 140),   # kahverengi
                (50, 100, 160),  # acik kahve
                (60, 120, 180),  # bej
                (30, 60, 100),   # koyu kahve
                (70, 130, 190),  # acik bej
            ])
            variation = tuple(
                max(0, min(255, c + random.randint(-15, 15)))
                for c in base_color
            )

            cv2.ellipse(
                image, (cx, cy), (rx, ry), angle,
                0, 360, variation, -1
            )

            # Hafif parlaklik efekti
            highlight_x = cx + random.randint(-3, 0)
            highlight_y = cy + random.randint(-3, 0)
            cv2.ellipse(
                image,
                (highlight_x, highlight_y),
                (max(1, rx // 3), max(1, ry // 3)),
                angle, 0, 360,
                tuple(min(255, c + 40) for c in variation),
                -1
            )

            positions.append((cx, cy, max(rx, ry)))
            placed += 1

    # Kaydet
    test_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_images"
    )
    os.makedirs(test_dir, exist_ok=True)
    test_path = os.path.join(test_dir, f"istockphoto-613555444-1024x1024.jpg")
    cv2.imwrite(test_path, image)
    print(f"[OK] Test goruntusu olusturuldu: {test_path}")
    print(f"     Gercek tane sayisi: {placed}")

    return test_path, placed


def main():
    parser = argparse.ArgumentParser(
        description="Bolum 3: Cok Taneli Madde Sayma",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ornekler:
  python tane_sayma.py --input taneler.jpg
  python tane_sayma.py --input taneler.jpg --min-area 100 --debug
  python tane_sayma.py --test
  python tane_sayma.py --test --count 80
        """
    )
    parser.add_argument("--input", "-i", type=str,
                        help="Giris goruntusu yolu")
    parser.add_argument("--output", "-o", type=str,
                        help="Cikis goruntusu yolu")
    parser.add_argument("--min-area", type=int, default=50,
                        help="Minimum tane alani (px)")
    parser.add_argument("--max-area", type=int, default=None,
                        help="Maksimum tane alani (px)")
    parser.add_argument("--debug", "-d", action="store_true",
                        help="Ara adimlari goster")
    parser.add_argument("--test", "-t", action="store_true",
                        help="Sentetik test ile calistir")
    parser.add_argument("--count", "-c", type=int, default=50,
                        help="Test icin tane sayisi")

    args = parser.parse_args()

    if args.test:
        test_path, actual = create_test_image(args.count, "ellipse")
        counted, result = count_objects(
            test_path,
            min_area=args.min_area,
            max_area=args.max_area,
            debug=args.debug
        )

        print(f"\n  Dogrulama:")
        print(f"    Gercek sayi : {actual}")
        print(f"    Bulunan sayi: {counted}")
        accuracy = (1 - abs(counted - actual) / actual) * 100
        accuracy = max(0, accuracy)
        print(f"    Dogruluk    : %{accuracy:.1f}")

        if accuracy >= 90:
            print(f"    [OK] Basarili!")
        else:
            print(f"    [!] min-area parametresini ayarlayin.")

    elif args.input:
        if not os.path.exists(args.input):
            print(f"[x] Dosya bulunamadi: {args.input}")
            sys.exit(1)
        count_objects(
            args.input,
            output_path=args.output,
            min_area=args.min_area,
            max_area=args.max_area,
            debug=args.debug
        )
    else:
        parser.print_help()
        print("\n[i] Hizli test icin: python tane_sayma.py --test")
        sys.exit(0)

    print(f"\n[OK] Bolum 3 tamamlandi!")


if __name__ == "__main__":
    main()
