"""
Bölüm 2 - Yardımcı Fonksiyonlar
Optik okuyucu (OMR) işlemleri için ortak yardımcı fonksiyonlar.
"""

import cv2
import numpy as np
import os


def resize_image(image, width=500):
    """Görüntüyü oranını koruyarak yeniden boyutlandırır."""
    h, w = image.shape[:2]
    ratio = width / float(w)
    new_height = int(h * ratio)
    resized = cv2.resize(image, (width, new_height), interpolation=cv2.INTER_AREA)
    return resized, ratio


def show_image(title, image, wait=True):
    """Görüntüyü ekranda gösterir."""
    h, w = image.shape[:2]
    max_dim = 900
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        display = cv2.resize(image, (int(w * scale), int(h * scale)))
    else:
        display = image.copy()
    cv2.imshow(title, display)
    if wait:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def save_image(image, path):
    """Görüntüyü dosyaya kaydeder."""
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    cv2.imwrite(path, image)
    print(f"[✓] Görüntü kaydedildi: {path}")


def order_points(pts):
    """4 köşe noktasını sıralar: sol-üst, sağ-üst, sağ-alt, sol-alt."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    """4 noktalı perspektif dönüşümü uygular."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def find_document_contour(image):
    """
    OMR kağıdının konturunu bulur.
    
    Returns:
        4 köşe noktası veya None
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    v = np.median(blurred)
    lower = int(max(0, (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))
    edged = cv2.Canny(blurred, lower, upper)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edged = cv2.dilate(edged, kernel, iterations=1)
    
    contours, _ = cv2.findContours(
        edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            return approx
    
    return None


def sort_contours(contours, method="left-to-right"):
    """
    Konturları belirtilen yöne göre sıralar.
    
    Args:
        contours: Kontur listesi
        method: "left-to-right", "right-to-left", "top-to-bottom", "bottom-to-top"
    
    Returns:
        Sıralanmış konturlar ve bounding box'lar
    """
    reverse = False
    i = 0  # x eksenine göre sırala
    
    if method == "right-to-left":
        reverse = True
    elif method == "top-to-bottom":
        i = 1  # y eksenine göre
    elif method == "bottom-to-top":
        reverse = True
        i = 1
    
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    (sorted_contours, sorted_boxes) = zip(*sorted(
        zip(contours, bounding_boxes),
        key=lambda b: b[1][i],
        reverse=reverse
    ))
    
    return list(sorted_contours), list(sorted_boxes)


def is_bubble(contour, min_area=150, max_area=3000, 
              min_circularity=0.5, min_aspect=0.6, max_aspect=1.4):
    """
    Bir konturun balon (bubble) olup olmadığını kontrol eder.
    
    Balonlar genellikle:
    - Belirli bir alan aralığında
    - Dairesel (circularity ~1.0)
    - Kare benzeri en-boy oranı
    
    Args:
        contour: Kontur
        min_area, max_area: Alan aralığı
        min_circularity: Minimum dairesellik
        min_aspect, max_aspect: En-boy oranı aralığı
    
    Returns:
        True ise balon, False değilse
    """
    area = cv2.contourArea(contour)
    if area < min_area or area > max_area:
        return False
    
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return False
    
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    if circularity < min_circularity:
        return False
    
    (x, y, w, h) = cv2.boundingRect(contour)
    aspect_ratio = w / float(h)
    if aspect_ratio < min_aspect or aspect_ratio > max_aspect:
        return False
    
    return True
