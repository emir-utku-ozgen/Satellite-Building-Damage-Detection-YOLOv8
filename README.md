# Satellite & Drone Image Based Building Damage Detection Using YOLOv8

<a href="https://colab.research.google.com/github/emir-utku-ozgen/Satellite-Building-Damage-Detection-YOLOv8/blob/main/deprem.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

This project fine-tunes a **YOLOv8-Medium** object detection model to locate buildings in post-disaster satellite and drone imagery and classify their damage level into four states: **No Damage, Minor Damage, Major Damage, and Destroyed**.

## 🔧 Training Setup

| Setting | Value |
| --- | --- |
| **Architecture** | YOLOv8-Medium (`yolov8m.pt`, COCO-pretrained → transfer learning) |
| **Dataset** | Roboflow building-damage dataset, exported in YOLOv8 format |
| **Image size** | 640 × 640 |
| **Epochs** | 25 |
| **Optimizer** | AdamW |
| **Learning rate schedule** | Cosine annealing (`cos_lr=True`) |
| **Data augmentation** | Enabled (`augment=True`) |
| **Loss weighting** | Classification loss weight raised to `cls=1.5` (default `0.5`), box loss kept at default `box=7.5` |
| **Hardware** | Google Colab GPU |

**Why raise the classification loss weight?** Locating a building is relatively easy; telling *Minor* damage apart from *Major* damage is the hard part. Increasing `cls` makes the model pay more attention to getting the damage level right.

## 📊 Validation Results

Evaluated on 419 validation images containing 2,379 labeled buildings.

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- | --- | --- |
| **All** | 419 | 2379 | 0.712 | 0.656 | **0.747** | 0.529 |
| Class 0 | 278 | 1070 | 0.746 | 0.750 | 0.812 | 0.579 |
| Class 1 | 180 | 564 | 0.709 | 0.587 | 0.701 | 0.497 |
| Class 2 | 146 | 327 | 0.724 | 0.676 | 0.777 | 0.568 |
| Class 3 | 111 | 291 | 0.691 | 0.530 | 0.650 | 0.414 |

**Inference speed:** ~22 ms per 640×640 image on validation, ~34 ms per image in standalone prediction (Colab GPU).

## 🔍 Observations

- **Performance follows data volume.** The class with the most examples (1,070 instances) reaches the highest mAP50 (81.2%), while the class with the fewest examples (291 instances) is the hardest (65.0%).
- **Recall is lower than precision.** When the model labels a building it is usually right, but it misses some buildings. In a disaster-response setting recall matters more, because a missed damaged building is more costly than a false alarm.
- **Dataset note:** The export also contains a small fifth label (`undefined`, 127 instances). It was kept as-is in this run.

## 🔐 Security Practice

The Roboflow API key is not hardcoded; it is read from Google Colab Secrets (`userdata.get('ROBOFLOW_API_KEY')`).

## 🚀 Future Work

- [ ] Collect more examples for under-represented damage classes, or rebalance them with targeted augmentation.
- [ ] Clean or relabel the `undefined` class.
- [ ] Run a smaller-model baseline (YOLOv8-Nano) under the same settings for a speed vs. accuracy comparison.
- [ ] Build a **Streamlit** web app for interactive inference on user-uploaded images.
- [ ] Add a **Destruction Index** heatmap to visualize regional damage density.
