# Satellite & Drone Image Based Building Damage Detection Using YOLOv8

This project implements an advanced computer vision pipeline to detect and classify structural building damage from satellite and drone imagery following natural disasters. Utilizing the **YOLOv8-Medium** architecture, the system categorizes damage into four distinct states: No Damage, Minor Damage, Major Damage, and Destroyed.

## 📊 Technical Performance & Optimization

Rather than executing a standard out-of-the-box script, this project focuses on rigorous hyperparameter tuning and model scaling to solve complex pixel-level feature extraction challenges.

| Feature / Metric | Baseline Model (v1) | Optimized Model (v3) |
| :--- | :--- | :--- |
| **Architecture** | YOLOv8-Nano (`yolov8n.pt`) | **YOLOv8-Medium (`yolov8m.pt`)** |
| **Optimizer** | Auto (SGD) | **AdamW (Weight Decay Tuned)** |
| **Data Augmentation** | None | **Enabled (`augment=True`)** |
| **Learning Rate** | Constant | **Cosine Annealing (`cos_lr=True`)** |
| **mAP50 Accuracy** | %76.3 | **%82.4+ (Significantly Improved)** |
| **Inference Speed** | ~2.5ms | **33.5ms (Real-Time Capable)** |

### Key Engineering Interventions:
* **Transfer Learning & Fine-Tuning:** Leveraged pre-trained weights to bend general vision capabilities into specialized remote sensing features.
* **Security & Best Practices:** Hardcoded credentials were systematically removed by integrating Google Colab Secrets (`userdata`) to protect Roboflow API parameters during deployment.
* **Loss Function Warping:** Tweaked bounding box scaling factors to harshly penalize misclassified damage states on tight coordinate grids.

## 🚀 Future Roadmap
- [ ] Develop a **Streamlit** web application for interactive user-uploaded image inference.
- [ ] Implement a **Destruction Index** density heatmap to visualize regional devastation levels.
