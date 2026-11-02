"""
pipeline.py — Pipeline end-to-end de desidentificación de radiografías
  1. Detección con YOLOv8 (umbral de confianza adaptativo por clase)
  2. OCR con EasyOCR sobre cada región detectada
  3. Anonimización: blur gaussiano o caja negra sobre las bboxes
  4. Métrica de utilidad clínica: % de píxeles modificados
Hackathon TreeLogic / IABioMed
"""

import os
import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm

# ── Configuración ────────────────────────────────────────────────────────────

from pathlib import Path
BASE_DIR = str(Path(__file__).resolve().parent)
MODEL_PATH  = os.path.join(BASE_DIR, "runs", "deid_v1-3", "weights", "best.pt")

INPUT_DIR   = os.path.join(BASE_DIR, "images", "val")
OUTPUT_DIR  = os.path.join(BASE_DIR, "output_anon")

IOU_THRESH  = 0.45
BLUR_KERNEL = 51

CLASS_NAMES = {0: "name", 1: "id", 2: "age", 3: "date", 4: "time"}

CLASS_CONF = {
    "name": 0.15,
    "id":   0.15,
    "date": 0.20,
    "age":  0.25,
    "time": 0.15,
}

# ── Inicialización ───────────────────────────────────────────────────────────

print("🔧 Cargando modelo YOLO...")
model = YOLO(MODEL_PATH)

print("🔧 Cargando EasyOCR...")
reader = easyocr.Reader(["en"], gpu=True)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Funciones auxiliares ─────────────────────────────────────────────────────

def blur_region(image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    roi = image[y1:y2, x1:x2]
    blurred = cv2.GaussianBlur(roi, (BLUR_KERNEL, BLUR_KERNEL), 0)
    image[y1:y2, x1:x2] = blurred
    return image


def redact_region(image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    image[y1:y2, x1:x2] = 0
    return image


def compute_privacy_utility(original: np.ndarray, anonymized: np.ndarray) -> dict:
    """
    Métrica de utilidad clínica:
    - pixels_modified: número de píxeles que han cambiado
    - pct_modified: % del área total modificada
    - pct_preserved: % de la imagen que se conserva intacta
    Un buen modelo borra poco (quirúrgico) pero borra todo lo sensible.
    """
    diff           = np.any(original != anonymized, axis=2)  # máscara de cambios
    total_pixels   = diff.size
    modified       = int(diff.sum())
    pct_modified   = round(modified / total_pixels * 100, 3)
    pct_preserved  = round(100 - pct_modified, 3)
    return {
        "total_pixels":  total_pixels,
        "pixels_modified": modified,
        "pct_modified":  pct_modified,
        "pct_preserved": pct_preserved,
    }


def draw_debug_box(image: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                   label: str, conf: float, ocr_text: str) -> np.ndarray:
    CLASS_COLORS = {
        "name": (255, 99,  99),
        "id":   (99,  99,  255),
        "age":  (99,  255, 99),
        "date": (255, 200, 99),
        "time": (200, 99,  255),
    }
    color  = CLASS_COLORS.get(label, (255, 255, 255))
    corner = 10
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
    cv2.line(image, (x1, y1), (x1 + corner, y1), color, 4)
    cv2.line(image, (x1, y1), (x1, y1 + corner), color, 4)
    cv2.line(image, (x2, y1), (x2 - corner, y1), color, 4)
    cv2.line(image, (x2, y1), (x2, y1 + corner), color, 4)
    cv2.line(image, (x1, y2), (x1 + corner, y2), color, 4)
    cv2.line(image, (x1, y2), (x1, y2 - corner), color, 4)
    cv2.line(image, (x2, y2), (x2 - corner, y2), color, 4)
    cv2.line(image, (x2, y2), (x2, y2 - corner), color, 4)
    return image


def process_image(img_path: str, anon_mode: str = "blur") -> dict:
    image_orig = cv2.imread(img_path)
    if image_orig is None:
        print(f"  ⚠️  No se pudo leer: {img_path}")
        return {}

    image_anon  = image_orig.copy()
    image_debug = image_orig.copy()

    results = model.predict(
        source  = img_path,
        conf    = min(CLASS_CONF.values()),
        iou     = IOU_THRESH,
        device  = 0,
        verbose = False,
    )

    detections = []

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            label  = CLASS_NAMES.get(cls_id, "unknown")

            if conf < CLASS_CONF.get(label, 0.25):
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            roi      = image_orig[y1:y2, x1:x2]
            ocr_out  = reader.readtext(roi, detail=0)
            ocr_text = " ".join(ocr_out).strip()

            detections.append({
                "class_name": label,
                "confidence": round(conf, 3),
                "threshold":  CLASS_CONF.get(label, 0.25),
                "bbox":       [x1, y1, x2, y2],
                "ocr_text":   ocr_text,
            })

            if anon_mode == "blur":
                image_anon = blur_region(image_anon, x1, y1, x2, y2)
            else:
                image_anon = redact_region(image_anon, x1, y1, x2, y2)

            image_debug = draw_debug_box(image_debug, x1, y1, x2, y2,
                                         label, conf, ocr_text)

    # Métrica de utilidad clínica
    privacy_utility = compute_privacy_utility(image_orig, image_anon)

    # Guardar imágenes
    fname    = Path(img_path).stem
    out_path = os.path.join(OUTPUT_DIR, f"{fname}_anon.png")
    cv2.imwrite(out_path, image_anon)

    debug_dir  = os.path.join(OUTPUT_DIR, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    cv2.imwrite(os.path.join(debug_dir, f"{fname}_debug.png"), image_debug)

    return {
        "image":          fname,
        "detections":     detections,
        "privacy_utility": privacy_utility,
        "anon_path":      out_path,
    }

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    img_paths = sorted(Path(INPUT_DIR).glob("*.png"))
    if not img_paths:
        print(f"❌ No se encontraron imágenes .png en {INPUT_DIR}")
        return

    print(f"\n📂 Procesando {len(img_paths)} imágenes → {OUTPUT_DIR}\n")

    all_results  = []
    total_pct    = []

    for img_path in tqdm(img_paths, desc="Desidentificando"):
        res = process_image(str(img_path), anon_mode="blur")
        if res:
            all_results.append(res)
            pu = res["privacy_utility"]
            total_pct.append(pu["pct_modified"])
            for det in res["detections"]:
                print(f"  [{det['class_name']:5s}] conf={det['confidence']:.2f} "
                      f"(umbral={det['threshold']}) | OCR: '{det['ocr_text']}'")

    # Resumen global de utilidad clínica
    if total_pct:
        avg_modified  = round(sum(total_pct) / len(total_pct), 3)
        avg_preserved = round(100 - avg_modified, 3)
        print("\n" + "═" * 50)
        print("  MÉTRICA DE UTILIDAD CLÍNICA")
        print("═" * 50)
        print(f"  Imágenes procesadas : {len(all_results)}")
        print(f"  % medio modificado  : {avg_modified}%")
        print(f"  % medio preservado  : {avg_preserved}%")
        print(f"  → El modelo modifica solo el {avg_modified}% de cada imagen,")
        print(f"    preservando el {avg_preserved}% de la información clínica.")
        print("═" * 50)

    print(f"\n✅ Listo.")
    print(f"   Anonimizadas en : {OUTPUT_DIR}")
    print(f"   Debug en        : {OUTPUT_DIR}/debug/")


if __name__ == "__main__":
    main()
