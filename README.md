# 🌑 Crater Detection on Planetary Surfaces

Automatic detection of impact craters on Mars and the Moon using YOLOv8.
![Crater Detection Results](test_results_grid.png)


## Results

| Metric | Score |
|--------|-------|
| Precision | 0.674 |
| Recall | 0.603 |
| mAP50 | 0.617 |
| F1 Score | 0.636 |

## Why This Matters

Crater counting is used by planetary scientists to estimate the age of 
planetary surfaces. More craters = older surface, like scratches on an 
old table. This project automates that process using computer vision.

## Dataset

- **Source:** [Martian/Lunar Crater Detection Dataset](https://www.kaggle.com/datasets/lincolnzh/martianlunar-crater-detection-dataset)
- **Training images:** 98
- **Validation images:** 26  
- **Test images:** 19
- **Annotation format:** YOLO format bounding boxes

## Model

- **Architecture:** YOLOv8s (fine-tuned from ImageNet pretrained weights)
- **Training:** 50 epochs with early stopping (best at epoch 4)
- **Inference speed:** ~25ms per image on T4 GPU
- **Input size:** 640x640

## Project Journey

This project went through several iterations:

1. Built custom patch dataset from NASA Trek tiles + LROC crater catalogue
2. Trained CNN binary classifier — hit class imbalance and resolution issues
3. Switched to properly labelled YOLO-format dataset
4. Fine-tuned YOLOv8s — achieved mAP50 of 0.617

## How to Run
```bash
pip install ultralytics

from ultralytics import YOLO
model = YOLO('weights/best.pt')
results = model('your_image.jpg', conf=0.25)
results[0].show()
```

## Stack

- Python, PyTorch
- YOLOv8 (Ultralytics)
- Google Colab / Kaggle (T4 GPU)
- NASA LRO WAC Mosaic imagery
- LROC Crater Catalogue (Silburt et al.)

## Author

Built as a computer vision portfolio project using real NASA/ESA planetary imagery.
