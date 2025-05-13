# 🌊 IGARSS 2024 Flood Mapping Competition (Track 1) — Flooded vs Non-Flooded Pixel Classification

Welcome to our repository for **IGARSS 2024 Track 1: Flood Detection Challenge**, where we focus on identifying **flooded vs non-flooded pixels** in satellite imagery. This project represents weeks of dedicated research, experimentation, and model optimization in our pursuit of accurate flood detection using deep learning and remote sensing techniques.

---

## 📌 Problem Statement

Floods are among the most devastating natural disasters, and rapid flood mapping using satellite imagery is essential for timely response and relief. The goal of this competition was to accurately classify each pixel in multi-modal satellite imagery (Sentinel-1 and Sentinel-2) as **flooded** or **non-flooded** under varying geographical and seasonal conditions.

---

## 🧠 What We Did ?

Our approach to the challenge involved integrating **multiple advanced deep learning models**, **custom loss functions**, and **fusion architectures** to improve classification accuracy. Here's a summary of the models and techniques we implemented:

### 🔹 1. Preprocessing
- We tried applying various filtering techniques like Lee, Frost, William, Multi Look Ahead filters for noise reduction and smoothening effect.
- As the data was obtained from various different sources, having different resolutions for each feature, we applied **bicubic interpolation** to make the resolutions for each feature same.

### 🔹 2. UNet with Attention Mechanism
- Built upon the standard UNet encoder-decoder framework.
- Incorporated **attention gates** in skip connections to enhance relevant feature propagation.
- Helped suppress irrelevant background noise often present in SAR imagery.

### 🔹 3. Dice Loss and Combined Losses
- Applied **Dice loss** to tackle class imbalance, ensuring improved segmentation of flooded areas.
- Explored **hybrid loss functions** combining Binary Cross Entropy (BCE) and Focal Loss with Dice to balance pixel-wise and region-wise accuracy.

### 🔹 4. Uncertainty Ranking Algorithm
- Developed a ranking-based approach using output **entropy and softmax uncertainty** to refine post-processing.
- Enabled selective thresholding to improve confidence in ambiguous areas.

### 🔹 5. Feature Fusion Network (FFN)
- Designed a custom architecture that combines multi-resolution features from different stages of the backbone.
- Encourages the network to **learn spatial patterns at multiple scales** — crucial for flood detection across diverse geographies.

### 🔹 6. Multi-Scale Feature Scaling Network
- Built a network inspired by **pyramid pooling** and **Atrous Spatial Pyramid Pooling (ASPP)**.
- Fuses features from various receptive fields to better capture local and global contextual information.

---



## 🖼️ Sample Predictions

We include visualizations of model predictions overlayed with ground truth for:
- The LULC patterns of the areas under study
- DEM of the sites
- Water susceptibility map
- SAR VV and VH polarisation images

---

## 📊 Evaluation Metrics

We used a comprehensive set of metrics for validation:
- **Intersection over Union (IoU)**
- **Precision / Recall**
- **F1-Score**
- **Standard Training and Testing accuracies**

These were calculated both pixel-wise and on a per-image basis to ensure robust model assessment.

---

## 🏆 Best Model Performance Summary

Below are the top metrics achieved by our best-performing model after extensive experimentation and fine-tuning for **UNet  with attention and Uncertainity Ranking Algorithm**:

| Metric          | Value     |
|-----------------|-----------|
| **Train Accuracy**   | 97.44%    |
| **Test Accuracy**    | 97.52%    |
| **Test F1 Score**    | 0.8413    |
| **Train Loss**       | 1.5147    |
| **Train Dice Score** | 0.1141    |
| **Validation Dice**  | 0.1417    |

These results highlight the strong generalization capability of our approach across diverse flood scenarios.


## 🧪 Experimental Highlights

- 📈 Ablation studies to test individual components (e.g., effect of attention, impact of fusion depth)
- 🧊 Data augmentation for SAR (rotation, flipping, speckle simulation)
- 🧮 K-Fold cross-validation for performance stability
- 💡 Insights on false positives near riverbanks

---