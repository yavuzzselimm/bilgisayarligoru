"""
Bölüm 1 - Yardımcı Fonksiyonlar
Doküman tarama işlemleri için ortak yardımcı fonksiyonlar.
"""

import cv2
import numpy as np
import os


def resize_image(image, width=500):
    """
    Görüntüyü belirtilen genişliğe oranını koruyarak yeniden boyutlandırır.
    
    Args:
        image: Giriş görüntüsü (numpy array)
        width: Hedef genişlik (piksel)
    
    Returns:
        Yeniden boyutlandırılmış görüntü ve oran (ratio)
    """
    h, w = image.shape[:2]
    ratio = width / float(w)
    new_height = int(h * ratio)
    resized = cv2.resize(image, (width, new_height), interpolation=cv2.INTER_AREA)
    return resized, ratio


def show_image(title, image, wait=True):
    """
    Görüntüyü ekranda gösterir.
    
    Args:
        title: Pencere başlığı
        image: Gösterilecek görüntü
        wait: True ise tuş basılana kadar bekler
    """
    # Görüntü çok büyükse yeniden boyutlandır
    h, w = image.shape[:2]
    max_dim = 800
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        display = cv2.resize(image, (int(w * scale), int(h * scale)))
    else:
        display = image.copy()
    
    cv2.imshow(title, display)
    if wait:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def show_images_side_by_side(titles, images):
    """
    Birden fazla görüntüyü yan yana gösterir.
    
    Args:
        titles: Pencere başlıkları listesi
        images: Görüntüler listesi
    """
    for title, img in zip(titles, images):
        show_image(title, img, wait=False)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_image(image, path):
    """
    Görüntüyü dosyaya kaydeder.
    
    Args:
        image: Kaydedilecek görüntü
        path: Dosya yolu
    """
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    cv2.imwrite(path, image)
    print(f"[✓] Görüntü kaydedildi: {path}")


def order_points(pts):
    """
    4 köşe noktasını sıralar: sol-üst, sağ-üst, sağ-alt, sol-alt.
    
    Bu fonksiyon perspektif dönüşümü için kritiktir.
    Noktalar tutarlı sırada olmazsa warp sonucu bozuk olur.
    
    Args:
        pts: 4 köşe noktası (4x2 numpy array)
    
    Returns:
        Sıralanmış köşe noktaları (4x2 float32 array)
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # Sol-üst: x+y toplamı en küçük olan nokta
    # Sağ-alt: x+y toplamı en büyük olan nokta
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Sol-üst
    rect[2] = pts[np.argmax(s)]  # Sağ-alt
    
    # Sağ-üst: x-y farkı en küçük olan nokta
    # Sol-alt: x-y farkı en büyük olan nokta
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Sağ-üst
    rect[3] = pts[np.argmax(diff)]  # Sol-alt
    
    return rect


def four_point_transform(image, pts):
    """
    4 köşe noktası kullanarak perspektif dönüşümü uygular.
    Yamuk (trapezoid) şekli dikdörtgene dönüştürür.
    
    Args:
        image: Giriş görüntüsü
        pts: 4 köşe noktası
    
    Returns:
        Perspektif düzeltilmiş görüntü
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Yeni görüntünün genişliğini hesapla
    # Alt kenar genişliği
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    # Üst kenar genişliği
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # Yeni görüntünün yüksekliğini hesapla
    # Sağ kenar yüksekliği
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    # Sol kenar yüksekliği
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Hedef noktalar (düz dikdörtgen)
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    
    # Perspektif dönüşüm matrisi
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped
