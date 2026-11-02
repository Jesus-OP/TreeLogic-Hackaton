"""
app.py — Interfaz Gradio para desidentificación de radiografías
Hackathon TreeLogic / IABioMed
"""

import os
import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
from PIL import Image
import gradio as gr

# ── Configuración ────────────────────────────────────────────────────────────

from pathlib import Path
BASE_DIR = str(Path(__file__).resolve().parent)
MODEL_PATH = os.path.join(BASE_DIR, "runs", "deid_v1-3", "weights", "best.pt")

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

CLASS_COLORS = {
    "name": (255, 99,  99),
    "id":   (99,  99,  255),
    "age":  (99,  255, 99),
    "date": (255, 200, 99),
    "time": (200, 99,  255),
}

# ── Carga de modelos ──────────────────────────────────────────────────────────

print("🔧 Cargando modelo YOLO...")
model = YOLO(MODEL_PATH)

print("🔧 Cargando EasyOCR...")
reader = easyocr.Reader(["en"], gpu=True)

print("✅ Modelos cargados. Arrancando interfaz...\n")

# ── Funciones ─────────────────────────────────────────────────────────────────

def blur_region(image, x1, y1, x2, y2):
    roi = image[y1:y2, x1:x2]
    image[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (BLUR_KERNEL, BLUR_KERNEL), 0)
    return image


def redact_region(image, x1, y1, x2, y2):
    image[y1:y2, x1:x2] = 0
    return image


def compute_privacy_utility(original, anonymized):
    diff          = np.any(original != anonymized, axis=2)
    total_pixels  = diff.size
    modified      = int(diff.sum())
    pct_modified  = round(modified / total_pixels * 100, 3)
    pct_preserved = round(100 - pct_modified, 3)
    return pct_modified, pct_preserved


def draw_box(image, x1, y1, x2, y2, label, conf, ocr_text):
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


def process(pil_image, anon_mode):
    if pil_image is None:
        return None, None, "⚠️ Sube una imagen primero."

    img_np  = np.array(pil_image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    img_anon  = img_bgr.copy()
    img_debug = img_bgr.copy()

    results = model.predict(
        source  = img_bgr,
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

            roi      = img_bgr[y1:y2, x1:x2]
            ocr_out  = reader.readtext(roi, detail=0)
            ocr_text = " ".join(ocr_out).strip()

            detections.append({
                "clase":     label,
                "confianza": conf,
                "umbral":    CLASS_CONF.get(label, 0.25),
                "ocr":       ocr_text,
            })

            if anon_mode == "Blur gaussiano":
                img_anon = blur_region(img_anon, x1, y1, x2, y2)
            else:
                img_anon = redact_region(img_anon, x1, y1, x2, y2)

            img_debug = draw_box(img_debug, x1, y1, x2, y2, label, conf, ocr_text)

    # Métrica de utilidad clínica
    pct_modified, pct_preserved = compute_privacy_utility(img_bgr, img_anon)

    # BGR → RGB
    img_anon_rgb  = cv2.cvtColor(img_anon,  cv2.COLOR_BGR2RGB)
    img_debug_rgb = cv2.cvtColor(img_debug, cv2.COLOR_BGR2RGB)

    # Resumen
    if not detections:
        summary = "✅ No se detectaron datos sensibles en esta imagen."
    else:
        lines = [f"🔍 **{len(detections)} zona(s) detectada(s) y anonimizada(s):**\n"]
        for d in detections:
            ocr_str = f'→ OCR: *"{d["ocr"]}"*' if d["ocr"] else ""
            lines.append(
                f"- **{d['clase'].upper()}** (conf: {d['confianza']:.0%} | umbral: {d['umbral']}) {ocr_str}"
            )
        lines.append(f"\n---")
        lines.append(f"📊 **Métrica de utilidad clínica:**")
        lines.append(f"- Píxeles modificados : **{pct_modified}%**")
        lines.append(f"- Información preservada : **{pct_preserved}%**")
        lines.append(f"\n> El modelo actúa de forma quirúrgica, modificando solo el {pct_modified}% de la imagen.")
        summary = "\n".join(lines)

    return Image.fromarray(img_debug_rgb), Image.fromarray(img_anon_rgb), summary


# ── Interfaz Gradio ───────────────────────────────────────────────────────────

with gr.Blocks(title="Desidentificación de Radiografías", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🏥 Desidentificación Visual de Radiografías
    **Hackathon TreeLogic · IABioMed**

    Sube una radiografía y el sistema detectará y anonimizará automáticamente
    la información sensible del paciente (nombre, ID, edad, fecha, hora).
    """)

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                type="pil",
                label="📂 Radiografía original",
                height=400,
            )
            anon_mode = gr.Radio(
                choices=["Blur gaussiano", "Caja negra"],
                value="Blur gaussiano",
                label="Método de anonimización",
            )
            btn = gr.Button("🔍 Desidentificar", variant="primary", size="lg")

            gr.Markdown("""
            **Umbrales por clase:**
            | Clase | Umbral | Motivo |
            |-------|--------|--------|
            | name  | 0.15   | Dato más identificativo |
            | id    | 0.15   | Dato crítico |
            | date  | 0.20   | Puede reidentificar combinado |
            | age   | 0.25   | Recall perfecto en validación |
            | time  | 0.15   | Recall más bajo (0.95) |
            """)

        with gr.Column(scale=2):
            with gr.Tab("📦 Detecciones"):
                debug_image = gr.Image(label="Zonas detectadas", height=400)
            with gr.Tab("✅ Anonimizada"):
                anon_image = gr.Image(label="Imagen anonimizada", height=400)
            summary_box = gr.Markdown(label="Resumen")

    btn.click(
        fn      = process,
        inputs  = [input_image, anon_mode],
        outputs = [debug_image, anon_image, summary_box],
    )

if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True)
