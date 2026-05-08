# 🧠 CNN Optimizer Comparison Study
### Breast Cancer Ultrasound Image Classification

> A systematic, controlled evaluation of **5 optimizers** on a fixed CNN architecture — isolating the optimizer's true contribution to training dynamics and accuracy.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-API-red?style=flat-square&logo=keras)](https://keras.io)
[![Platform](https://img.shields.io/badge/Platform-Google%20Colab-yellow?style=flat-square&logo=googlecolab)](https://colab.research.google.com)
[![License](https://img.shields.io/badge/License-Educational%20Use-green?style=flat-square)](LICENSE)

---

## 📌 Project Summary

| Property | Detail |
|---|---|
| **Task** | 3-class Breast Ultrasound Image Classification |
| **Classes** | `benign` · `malignant` · `normal` |
| **Architecture** | 4-Block CNN (32 → 64 → 128 → 256 filters) + 2 Dense layers |
| **Dataset Split** | 80% Train / 20% Test (stratified) |
| **Best Test Accuracy** | **83.23%** — RMSprop, Adagrad, Adadelta (3-way tie) |
| **Framework** | TensorFlow 2 / Keras |

---

## 🏆 Results at a Glance

| Optimizer | Train Acc | Test Acc | Epochs | Learning Rate |
|---|---|---|---|---|
| **Adam** | 96.99% | 82.91% | 30 | 0.001 |
| **SGD** | 98.18% | 81.96% | 10 | 0.0001 |
| 🥇 **RMSprop** | 98.26% | **83.23%** | 10 | 0.001 |
| 🥇 **Adagrad** | 98.26% | **83.23%** | 10 | 0.01 |
| 🥇 **Adadelta** | 98.26% | **83.23%** | 10 | default |

> **Key Finding:** RMSprop, Adagrad, and Adadelta all **outperformed Adam** in just 10 fine-tuning epochs — starting from Adam's pre-trained weights.

---

## 📂 Dataset

**Breast Ultrasound Images Dataset** — sourced from Kaggle  
`aryashah2k/breast-ultrasound-images-dataset`

| Class | Clinical Meaning |
|---|---|
| `benign` | Non-cancerous mass — potential to become malignant |
| `malignant` | Active cancer — cells can grow and spread |
| `normal` | Healthy breast tissue — no abnormality detected |

**Preprocessing pipeline:**
```python
def load_preprocess_image(image_path):
    image = Image.open(image_path)           # Load with PIL
    image = image.resize((150, 150))          # Standardise dimensions
    image = image.convert('L')                # Convert to grayscale
    image = np.array(image)
    image = image.reshape((150, 150, 1))      # Add channel dimension
    image = image.astype('float32') / 255.0   # Normalise to [0, 1]
    return image
```

> **Why Grayscale?** Ultrasound images encode all diagnostic information in pixel intensity, not colour. Single-channel conversion reduces model parameters without losing clinically relevant information.

---

## 🏗️ CNN Architecture

```
INPUT          →  150 × 150 × 1  (Grayscale ultrasound)
CONV BLOCK 1   →  Conv2D(32,  3×3, ReLU)  + MaxPool(2×2)  →  74×74×32
CONV BLOCK 2   →  Conv2D(64,  3×3, ReLU)  + MaxPool(2×2)  →  36×36×64
CONV BLOCK 3   →  Conv2D(128, 3×3, ReLU)  + MaxPool(2×2)  →  17×17×128
CONV BLOCK 4   →  Conv2D(256, 3×3, ReLU)  + MaxPool(2×2)  →  7×7×256
FLATTEN        →  12,544 neurons → 1D vector
DENSE 1        →  Dense(512, ReLU) + Dropout(0.5)
DENSE 2        →  Dense(256, ReLU) + Dropout(0.5)
OUTPUT         →  Dense(3, Softmax) → benign / malignant / normal
```

**Common config across all experiments:**
- Loss: `categorical_crossentropy`
- Batch size: `32`
- Validation split: `10%` during training
- Final evaluation: held-out 20% test set

---

## ⚙️ Optimizer Deep Dives

### 1. Adam — *Adaptive Moment Estimation*
Combines momentum + per-parameter adaptive learning rates with bias correction. The **base optimizer** for this study — all subsequent optimizers fine-tune from Adam's converged weights.

```
θ_t = θ_{t-1} − α × m̂_t / (√v̂_t + ε)
```
- lr = `0.001` · Epochs = `30` · Train = 96.99% · Test = **82.91%**

---

### 2. SGD — *Stochastic Gradient Descent*
Pure gradient step — no adaptation. Most sensitive to learning rate choice.

```
θ_t = θ_{t-1} − α × g_t
```

> ⚠️ **Critical finding:** `lr=0.01` caused divergence (validation loss increased). Final choice: `lr=0.0001`.

- lr = `0.0001` · Epochs = `10` · Train = 98.18% · Test = **81.96%**

---

### 3. RMSprop — *Root Mean Square Propagation*
Uses exponential moving average of squared gradients. Naturally robust to noisy gradients — ideal for ultrasound images with speckle noise.

```
v_t = ρ × v_{t-1} + (1−ρ) × g_t²
θ_t = θ_{t-1} − (α / √(v_t + ε)) × g_t
```
- lr = `0.001` · Epochs = `10` · Train = 98.26% · Test = **83.23%** 🥇

---

### 4. Adagrad — *Adaptive Gradient*
Accumulates ALL past squared gradients — great for sparse data, but the learning rate shrinks toward zero over long runs.

```
G_t = G_{t-1} + g_t²
θ_t = θ_{t-1} − (α / √(G_t + ε)) × g_t
```
- lr = `0.01` (higher to compensate for self-decay) · Epochs = `10` · Train = 98.26% · Test = **83.23%** 🥇

---

### 5. Adadelta — *Extension of Adagrad*
Fixes Adagrad's vanishing learning rate using a windowed accumulation. Notably **requires no manual learning rate**.

```
Δθ_t = − (RMS[Δθ]_{t-1} / RMS[g]_t) × g_t
```
- lr = `default` · Epochs = `10` · Train = 98.26% · Test = **83.23%** 🥇

---

## 📊 Visualisations Generated

| Plot | Purpose |
|---|---|
| Training Accuracy Curve | Track convergence health per optimizer |
| Training Loss Curve | Detect divergence or overfitting early |
| Bar Chart — Training Accuracy | Compare all 5 optimizers side-by-side |
| Bar Chart — Validation Accuracy | Identify the best-generalising optimizer |

```python
# Bar chart — validation accuracy
sns.barplot(x=['Adam', 'SGD', 'RMSprop', 'Adagrad', 'Adadelta'],
            y=[82.91, 81.96, 83.23, 83.23, 83.23],
            palette='coolwarm')
```

---

## 🔬 Experimental Design Note

> The model is trained with **Adam first (30 epochs)** to convergence. Each subsequent optimizer then continues training from those pre-trained weights for **10 additional epochs**. This makes the study a **fine-tuning comparison** — later optimizers benefit from Adam's learned representations.

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/Mindbender66/<repo-name>.git
cd <repo-name>

# Install dependencies
pip install tensorflow numpy pillow scikit-learn matplotlib seaborn

# Or open directly in Google Colab
# Runtime → Run all
```

**Dataset setup:**
1. Download from [Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)
2. Place in `Breast_cancer/` folder
3. Update `dataset_path` in the notebook

---

## 🧰 Tech Stack

| Library | Role |
|---|---|
| TensorFlow 2 / Keras | CNN model, optimizers, training loop |
| NumPy | Array ops, argmax, data conversion |
| Pillow (PIL) | Image loading, resizing, grayscale |
| scikit-learn | train_test_split, LabelEncoder, metrics |
| matplotlib | Per-optimizer training curves |
| seaborn | Comparative bar charts |
| Google Colab | Cloud GPU training environment |

---

## 🔮 Future Improvements

- [ ] Add `EarlyStopping(patience=5)` — prevent overfitting automatically
- [ ] Data augmentation (flip, rotate, zoom) — improve generalisation
- [ ] Address class imbalance via `class_weight`
- [ ] Transfer learning benchmark (VGG16, ResNet50)
- [ ] `BatchNormalization` after Conv layers
- [ ] `AdamW` with weight decay for better regularisation
- [ ] GradCAM visualisation — interpretability for clinicians
- [ ] 5-fold cross-validation for robust accuracy estimates

---

## ⚠️ Disclaimer

> This project is for **educational and research purposes only**.  
> It is **not a clinical diagnostic tool** and should not be used for medical decision-making.  
> Always consult a qualified medical professional for diagnosis.

---

## 👤 Author

**Valmiki Sarath Kumar**  
[@Mindbender66](https://github.com/Mindbender66) · he/him

---

*If this project helped you, consider giving it a ⭐ on GitHub!*
