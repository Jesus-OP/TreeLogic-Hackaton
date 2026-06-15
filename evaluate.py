"""
evaluate.py — Evaluación del modelo YOLOv8 sobre el conjunto de validación
Métricas: mAP50, mAP50-95, Precisión, Recall — por clase y global
Hackathon TreeLogic / IABioMed
"""

import os
from ultralytics import YOLO

# ── Configuración ────────────────────────────────────────────────────────────

BASE_DIR = r"D:\Proyectos\TreeLogicHackaton"
MODEL_PATH = os.path.join(BASE_DIR, "runs", "deid_v1-3", "weights", "best.pt")
DATA_YAML  = os.path.join(BASE_DIR, "data.yaml")

CLASS_NAMES = ["name", "id", "age", "date", "time"]

# ── Evaluación ───────────────────────────────────────────────────────────────

def main():
    print("🔧 Cargando modelo...")
    model = YOLO(MODEL_PATH)

    print("📊 Evaluando sobre el conjunto de validación...\n")
    metrics = model.val(
        data    = DATA_YAML,
        device  = 0,
        verbose = True,
        plots   = True,
        save_json = True,
    )

    # ── Resultados globales ──
    map50    = metrics.box.map50
    map5095  = metrics.box.map
    prec     = metrics.box.mp      # mean precision
    rec      = metrics.box.mr      # mean recall

    print("\n" + "═" * 50)
    print("  RESULTADOS GLOBALES")
    print("═" * 50)
    print(f"  mAP@0.50       : {map50:.4f}")
    print(f"  mAP@0.50:0.95  : {map5095:.4f}")
    print(f"  Precisión media: {prec:.4f}")
    print(f"  Recall medio   : {rec:.4f}")

    # ── Resultados por clase ──
    print("\n" + "═" * 50)
    print("  MÉTRICAS POR CLASE")
    print("═" * 50)
    print(f"  {'Clase':<8}  {'mAP50':>7}  {'Prec':>7}  {'Recall':>7}")
    print(f"  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*7}")

    per_class_map50 = metrics.box.ap50          # array por clase
    per_class_prec  = metrics.box.p              # precisión por clase
    per_class_rec   = metrics.box.r              # recall por clase

    for i, cls_name in enumerate(CLASS_NAMES):
        m  = per_class_map50[i] if i < len(per_class_map50) else float("nan")
        p  = per_class_prec[i]  if i < len(per_class_prec)  else float("nan")
        r  = per_class_rec[i]   if i < len(per_class_rec)   else float("nan")
        print(f"  {cls_name:<8}  {m:>7.4f}  {p:>7.4f}  {r:>7.4f}")

    print("\n✅ Evaluación completada.")
    print(f"   Gráficas guardadas en: {BASE_DIR}/runs/deid_v1/")


if __name__ == "__main__":
    main()
