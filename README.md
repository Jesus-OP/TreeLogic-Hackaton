# Medical Image De-identification (Treelogic Challenge)

**Finalist Solution - I Hackathon IA en Biomedicina (IABiomed) 2026**

## Overview

This repository contains the solution developed for the "Reto Treelogic: Desidentificación visual de imágenes médicas" at the IABiomed Hackathon 2026.

The goal is to automatically detect and anonymize personally identifiable information (PII) burned into medical images (X-rays). This is a critical step to protect patient privacy and to enable the safe use of clinical data for research and AI development.

## Achievement

- **Status:** Finalist
- **Event:** I Hackathon IA en Biomedicina 2026
- **Organizer:** IABiomed (Sociedad Española de Inteligencia Artificial en Biomedicina)

## How It Works

The solution is an end-to-end pipeline with four stages:

1. **Detection:** a YOLOv8 model detects text regions containing sensitive data, using a per-class adaptive confidence threshold.
2. **OCR:** EasyOCR reads the text inside each detected region (useful for auditing and debugging).
3. **Anonymization:** each detected region is hidden with a Gaussian blur or a black box.
4. **Clinical utility metric:** the percentage of modified pixels is computed, so we can verify that the image is anonymized surgically while the rest of the clinical information is preserved.

### Detected classes

| ID | Class  | Description        | Confidence threshold |
|----|--------|--------------------|----------------------|
| 0  | `name` | Patient name       | 0.15                 |
| 1  | `id`   | Patient / study ID | 0.15                 |
| 2  | `age`  | Patient age        | 0.25                 |
| 3  | `date` | Dates              | 0.20                 |
| 4  | `time` | Time stamps        | 0.15                 |

Thresholds are intentionally low for the most identifying classes (`name`, `id`, `time`) to favor **recall**: it is preferable to blur an extra area than to leave personal data exposed.

## Repository Structure

The project is built entirely in **Python**:

```
.
├── models/
│   └── best.pt        # Trained YOLOv8 weights (ready to use)
├── data.yaml          # Dataset paths and class definitions (YOLO format)
├── train.py           # YOLOv8 training script
├── evaluate.py        # Model evaluation on the validation set
├── pipeline.py        # Batch de-identification pipeline
├── app.py             # Gradio web interface
├── requirements.txt   # Python dependencies
└── README.md
```

- `models/best.pt`: trained weights used by `pipeline.py` and `app.py`. Training is optional.
- `data.yaml`: dataset location, number of classes (5) and class names.
- `train.py`: trains a YOLOv8 model (`yolov8n.pt` as base, 100 epochs, early stopping with patience 20).
- `evaluate.py`: evaluates the model on the validation set and reports mAP@0.50, mAP@0.50:0.95, precision and recall, both global and per class.
- `pipeline.py`: processes every `.png` in `images/val` and saves anonymized images plus debug images with the detected boxes.
- `app.py`: Gradio UI. Upload an X-ray, choose *Blur gaussiano* or *Caja negra*, and get the detections, the anonymized image and the clinical utility metric.

## Dataset Structure

The dataset follows the standard YOLO format:

```
images/
├── train/
└── val/
labels/
├── train/
└── val/
```

> **Note:** the actual medical image datasets used during the hackathon are **not** included in this repository to comply with data privacy regulations.

## Requirements

- Python 3.9+
- An **NVIDIA GPU with CUDA** (the code uses `device=0` and `easyocr.Reader(gpu=True)`). To run on CPU, change `device=0` to `device="cpu"` and `gpu=True` to `gpu=False` in `train.py`, `evaluate.py`, `pipeline.py` and `app.py`.

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/Jesus-OP/TreeLogic-Hackaton.git
   cd TreeLogic-Hackaton
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Launch the web interface (uses the included `models/best.pt`):

   ```bash
   python app.py
   ```

4. Or run the batch pipeline. Place your images in `images/val/` first:

   ```bash
   python pipeline.py
   ```

   Anonymized images are saved to `output_anon/`, and debug images with the detected boxes to `output_anon/debug/`.

### Optional: train and evaluate your own model

Requires your own dataset in the structure above.

```bash
python train.py
python evaluate.py
```

Training saves the weights to `runs/deid_v1/weights/best.pt`. To use them, copy `best.pt` to `models/best.pt`.

## Clinical Utility Metric

For every image, the pipeline reports the percentage of pixels modified by the anonymization:

- **% modified:** area changed by blur / redaction.
- **% preserved:** area of the image left untouched.

A good de-identification model hides all sensitive data while modifying as little of the image as possible.

## Privacy Notes

- `pipeline.py` prints the OCR text of each detection to the console for debugging. This text is the personal data itself, so **disable or remove that output when working with real patient data**.
- Never commit real patient images or labels to the repository.
