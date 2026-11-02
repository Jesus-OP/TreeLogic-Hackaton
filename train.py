"""
train.py — Entrenamiento YOLOv8 para desidentificación de radiografías
Hackathon TreeLogic / IABioMed
"""

from ultralytics import YOLO
import os

# ── Configuración ────────────────────────────────────────────────────────────

from pathlib import Path
BASE_DIR = str(Path(__file__).resolve().parent)
DATA_YAML = os.path.join(BASE_DIR, "data.yaml")

# Modelo base: yolov8n (nano) es rápido; cambia a yolov8s o yolov8m si quieres más precisión
MODEL_NAME = "yolov8n.pt"

EPOCHS     = 100
IMG_SIZE   = 640
BATCH_SIZE = 16   
WORKERS    = 4

PROJECT    = os.path.join(BASE_DIR, "runs")
RUN_NAME   = "deid_v1"

# ── Entrenamiento ────────────────────────────────────────────────────────────

def main():
    model = YOLO(MODEL_NAME)

    results = model.train(
        data      = DATA_YAML,
        epochs    = EPOCHS,
        imgsz     = IMG_SIZE,
        batch     = BATCH_SIZE,
        workers   = WORKERS,
        device    = 0,          # GPU 0 (tu 3060)
        project   = PROJECT,
        name      = RUN_NAME,
        patience  = 20,         # Early stopping si no mejora en 20 épocas
        save      = True,
        plots     = True,
        verbose   = True,
    )

    print("\n✅ Entrenamiento finalizado.")
    print(f"   Mejor modelo guardado en: {PROJECT}/{RUN_NAME}/weights/best.pt")
    print(f"   mAP50 final: {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")

if __name__ == "__main__":
    main()
