# Deepfake Video Detection

A deep learning system that classifies videos as **real** or **fake** using a Spatial-Temporal Attention model (ResNeXt-50 + CBAM + BiLSTM).

---

## Overview

Deepfakes leave traces in two places: subtle visual artifacts inside individual frames, and unnatural changes between frames. This project targets both.

```
Video → Frame Sampling → Face Detection (MTCNN / RetinaFace) → Face Alignment
      → ResNeXt-50 (per-frame features) → CBAM Attention
      → BiLSTM (temporal modeling) → Real / Fake
```

| Component | Role |
|-----------|------|
| **ResNeXt-50** | Extracts spatial features from every sampled face frame |
| **CBAM** | Channel + spatial attention that focuses on manipulated regions |
| **BiLSTM** | Learns temporal inconsistencies across the frame sequence |

## Features

- End-to-end pipeline: face extraction, training, evaluation, inference
- Two-phase training strategy with focal loss
- Support for FaceForensics++, Celeb-DFv2 and DFDC
- Attention heatmap visualization for explainability
- FastAPI backend and React web interface
- Metrics: Accuracy, F1, AUC, EER, Log Loss, ROC curve, confusion matrix

## Project Structure

```
├── model.py              # Full DeepfakeDetector model
├── dataset.py            # Video dataset loader
├── train.py              # Training entry point
├── test.py               # Evaluation entry point
├── infer.py              # Single-video inference
├── prepare_dataset.py    # Dataset preparation
├── preextract_faces.py   # Pre-extract faces for faster training
├── preprocessing/        # Face detection, alignment, augmentation
├── models/               # CBAM, CNN backbone, BiLSTM modules
├── training/             # Trainer, focal loss, phased strategy
├── evaluation/           # Metrics and plots
├── utils/                # Config, helpers, checkpointing
├── app/                  # FastAPI backend
└── frontend/             # React web UI
```

## Installation

**Requirements**

- Python 3.9+
- CUDA-capable GPU (recommended; CPU works but is much slower)
- Node.js 16+ (only for the frontend)

```bash
# 1. Clone
https://github.com/itsgarvmalik-glitch/Deepfake-Video-detector.git
cd Deepfake-Detection

# 2. Virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 3. Dependencies
pip install -r requirements.txt
```

## Dataset Setup

Arrange your videos like this:

```
data/
├── train/
│   ├── real/
│   └── fake/
├── val/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

Supported public datasets: **FaceForensics++**, **Celeb-DFv2**, **DFDC**.

Optionally pre-extract faces to speed up training:

```bash
python preextract_faces.py --data-dir data/ --output-dir data_cache/
```

## Training

```bash
python train.py \
    --train-dir data/train \
    --val-dir data/val \
    --epochs 30 \
    --batch-size 8 \
    --lr 1e-4 \
    --num-frames 16
```

Training runs in two phases:

1. **Phase 1:** backbone frozen, only CBAM and BiLSTM are trained.
2. **Phase 2:** full fine-tuning, with a 10x lower learning rate on the backbone.

A checkpoint is saved every epoch in `checkpoints/`, and the best model is stored as `checkpoints/best_model.pt`.

Resume a stopped run:

```bash
python train.py --train-dir data/train --val-dir data/val --resume checkpoints/best_model.pt
```

## Evaluation

```bash
python test.py \
    --test-dir data/test \
    --checkpoint checkpoints/best_model.pt \
    --dataset-name "FaceForensics++"
```

**Expected performance**

| Scenario | Accuracy | AUC | F1 |
|----------|----------|-----|----|
| FaceForensics++ (intra-dataset) | >90% | >0.95 | >0.90 |
| FF++ → Celeb-DFv2 (cross-dataset) | 80–88% | 0.85–0.93 | 0.78–0.87 |
| FF++ → DFDC (stress test) | 60–72% | 0.65–0.76 | 0.58–0.70 |

## Inference

```bash
python infer.py \
    --video path/to/video.mp4 \
    --checkpoint checkpoints/best_model.pt \
    --visualize
```

Sample output:

```json
{
  "prediction": "FAKE",
  "confidence": 0.9234,
  "fake_probability": 0.9234,
  "processing_details": {
    "frames_analyzed": 16,
    "video_duration_seconds": 8.4,
    "processing_time_seconds": 1.23,
    "device": "cuda"
  }
}
```

The `--visualize` flag saves attention heatmap overlays to `outputs/inference/`.

## Web App

**Backend**

```bash
export MODEL_CHECKPOINT=checkpoints/best_model.pt
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

Test the API:

```bash
curl -X POST http://localhost:8000/predict -F "video=@test_video.mp4"
```

**Frontend**

```bash
cd frontend
npm install
npm start
```

Open `http://localhost:3000`, upload a video, and view the prediction, confidence score and frame-by-frame attention visualization.

## Tech Stack

| Component | Libraries |
|-----------|-----------|
| Model training | PyTorch, timm |
| Face detection | facenet-pytorch (MTCNN), insightface (RetinaFace) |
| Video processing | OpenCV, albumentations |
| API | FastAPI, uvicorn |
| Frontend | React |
| Metrics | scikit-learn, matplotlib, seaborn |

## Troubleshooting

- **"No faces detected" warning:** videos with no detectable face are skipped automatically. If most videos are skipped, check for corrupted files.
- **CUDA out of memory:** lower `--batch-size` (e.g. 4) or `--num-frames` (e.g. 8).
- **Slow CPU training:** use `--workers 0` to avoid multiprocessing issues on Windows/WSL. A GPU is strongly recommended.
- **InsightFace install fails:** try `pip install insightface --no-build-isolation`.

## Acknowledgements

Built at PIET under the guidance of Dr. Deepti Dhingra
