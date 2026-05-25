"""
Bölüm 3 - Yardımcı Fonksiyonlar
Tane sayma işlemleri için ortak yardımcı fonksiyonlar.
"""

import cv2
import numpy as np
import os


def show_image(title, image, wait=True):
    """Görüntüyü ekranda gösterir."""
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


def save_image(image, path):
    """Görüntüyü dosyaya kaydeder."""
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    cv2.imwrite(path, image)
    print(f"[✓] Görüntü kaydedildi: {path}")


def create_colormap(n):
    """N adet farklı renk üretir."""
    colors = []
    for i in range(n):
        hue = int(180 * i / n)
        color_hsv = np.array([[[hue, 255, 200]]], dtype=np.uint8)
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)
        colors.append(tuple(int(c) for c in color_bgr[0][0]))
    return colors
