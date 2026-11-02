# Medical Image De-identification (Treelogic Challenge)

**Finalist Solution - I Hackathon IA en Biomedicina (IABiomed) 2026**

## Overview

This repository contains the solution developed for the "Reto Treelogic: Desidentificación visual de imágenes médicas" at the IABiomed Hackathon 2026.

The main objective of this project is to design an automated computer vision solution to remove and anonymize personally identifiable information (PII) embedded within medical images. This is a critical challenge for ensuring patient privacy and enabling the secure use of clinical data for research and AI development.

## Achievement

- **Status:** Finalist
- **Event:** I Hackathon IA en Biomedicina 2026
- **Organizer:** IABiomed (Sociedad Española de Inteligencia Artificial en Biomedicina)

## Repository Structure

The project is built entirely in **Python** and follows a modular machine learning pipeline:

- `data.yaml`: Configuration file for dataset paths and hyperparameter settings.
- `train.py`: Script for training the computer vision model on the medical image dataset.
- `evaluate.py`: Evaluation metrics and validation scripts to assess the model's anonymization accuracy.
- `pipeline.py`: End-to-end execution pipeline connecting data ingestion, processing, and inference.
- `app.py`: Application entry point (e.g., API or lightweight UI) for deploying the solution.
- `requirements.txt`: Python dependencies required to run the environment.

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/Jesus-OP/TreeLogic-Hackaton.git
   cd TreeLogic-Hackaton
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Execute the main pipeline:

   ```bash
   python pipeline.py
   ```

> **Note:** The actual medical image datasets used during the hackathon are not included in this repository to comply with data privacy regulations.
