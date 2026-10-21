# OCT-C8 Retinal Disease Classification Using Deep Learning

An Explainable Deep Learning Framework for 8-Class Retinal OCT Image Classification

## 📌 Project Overview

This research project focuses on developing an accurate, robust, and explainable deep learning framework for classifying retinal Optical Coherence Tomography (OCT) images into eight different categories.

The project investigates deep learning architectures, including VGG19, ResNet50, and DenseNet121, and explores a hybrid CNN-based approach for improved feature representation and classification. Grad-CAM is used to visualize important image regions influencing model predictions.

The primary goal is to support automated retinal OCT image analysis through comparative model evaluation and explainable artificial intelligence (XAI).

**Note:** This project is intended for research and educational purposes and is not a substitute for professional medical diagnosis.

## 🎯 Research Objectives

- Develop a deep learning pipeline for 8-class retinal OCT image classification.
- Establish baseline performance using VGG19, ResNet50, and DenseNet121.
- Investigate a hybrid CNN architecture for feature extraction and classification.
- Evaluate model performance using standard classification metrics.
- Apply Grad-CAM to visualize image regions relevant to model predictions.
- Investigate model robustness and compare experimental results.

## 🩺 Dataset Information

**Dataset:** Retinal OCT Image Classification – OCT-C8

**Source:** [OCT-C8 Dataset on Kaggle](https://www.kaggle.com/datasets/obulisainaren/retinal-oct-c8)

The dataset is described as containing eight retinal OCT image categories.

### Classification Classes

| Class | Description |
|---|---|
| AMD | Age-Related Macular Degeneration |
| CNV | Choroidal Neovascularization |
| CSR | Central Serous Retinopathy |
| DME | Diabetic Macular Edema |
| DR | Diabetic Retinopathy |
| DRUSEN | Retinal Drusen |
| MH | Macular Hole |
| NORMAL | Normal Retinal OCT |

The dataset is reported to contain approximately 24,000 images, with 3,000 images per class. Please verify these figures against the downloaded dataset before reporting them as the final experimental dataset statistics.

### Data Preparation

The preprocessing pipeline may include:

- Image loading and validation
- Resizing to the input dimensions required by each model
- Pixel normalization
- Training-only data augmentation
- Training, validation, and testing splits

All dataset splits should be documented clearly to avoid data leakage.

## 🧠 Deep Learning Models

### 1. VGG19
A deep convolutional neural network used as a baseline for extracting hierarchical image features.

### 2. ResNet50
A residual network that uses skip connections to support effective training of deeper neural networks.

### 3. DenseNet121
A densely connected convolutional neural network that encourages feature reuse and information flow between layers.

### 4. Hybrid CNN Architecture
An experimental hybrid approach intended to investigate complementary feature representations from different convolutional architectures.

The exact fusion strategy and attention modules should be documented according to the final implemented model.

## 🔍 Explainable Artificial Intelligence (XAI)

### Grad-CAM

Gradient-weighted Class Activation Mapping (Grad-CAM) is used to generate visual explanations of model predictions.

It helps researchers inspect which image regions contribute to a predicted class and investigate potentially incorrect predictions.

Grad-CAM visualizations are interpretability aids; they do not independently establish the medical correctness of a prediction.

## 📊 Model Evaluation

The models can be evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Per-class classification report
- Robustness evaluation

A fair comparison should use the same held-out test set and consistent evaluation procedures across models.

**Experimental Results:** The following table summarizes the performance of the evaluated models on the test dataset.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| VGG19 | 97.93% | 97.93% | 97.93% | 97.93% |
| ResNet50 | 97.89% | 97.91% | 97.89% | 97.89% |
| DenseNet121 | 98.00% | 98.02% | 98.00% | 98.00% |
| Proposed Hybrid CNN | 97.43% | 97.43% | 97.43% | 97.43% |

**Note:** These values must be verified against the corresponding classification reports. Precision, recall, and F1-score should use the same averaging method for all models.

## 🗂️ Project Structure

```text
OCT-C8-Retinal-Disease-Classification/
│
├── src/
│   ├── data.py
│   ├── evaluate.py
│   └── models.py
│
├── .gitignore
├── ablation.py
├── app.py
├── config.py
├── dataset_check.py
├── evaluate.py
├── final_comparison.py
├── gradcam.py
├── predict.py
├── requirements.txt
├── robustness.py
├── system_check.py
└── train.py
```

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.10 or a compatible version supported by the dependencies
- pip
- Git
- A CPU or compatible GPU for model training

### Step 1: Clone the Repository

```bash
git clone https://github.com/Bhavik94290/OCT-C8-Retinal-Disease-Classification.git

cd OCT-C8-Retinal-Disease-Classification
```

### Step 2: Create a Virtual Environment

Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Download the Dataset

Download the OCT-C8 dataset from Kaggle and extract it locally.

Configure the dataset path according to the directory structure expected by the project's data loader and configuration file.

Dataset images are not included in this repository.

## 🚀 Running the Project

Run the following commands from the repository root, after configuring the dataset path and confirming the corresponding scripts' arguments.

### Check the Environment

```bash
python system_check.py
```

### Check the Dataset

```bash
python dataset_check.py
```

### Train the Models

```bash
python train.py
```

### Evaluate the Models

```bash
python evaluate.py
```

### Compare Experimental Results

```bash
python final_comparison.py
```

### Generate Grad-CAM Visualizations

```bash
python gradcam.py
```

### Run Robustness Experiments

```bash
python robustness.py
```

**Important:** The commands above reflect the current project script names. Check each script's configuration and command-line arguments before running it. If the project requires a particular model name, checkpoint path, or dataset path, configure those values accordingly.

## 🧪 Research Methodology

The experimental workflow consists of:

1. Dataset preparation and verification
2. Image preprocessing and training augmentation
3. Baseline model training
4. Hybrid model experimentation
5. Validation-based model selection
6. Evaluation on a held-out test set
7. Confusion matrix and classification report generation
8. Grad-CAM visualization
9. Robustness and ablation experiments
10. Comparative analysis and documentation

## 🔬 Reproducibility

For reproducible experiments, document:

- Dataset source and version
- Train/validation/test split strategy
- Random seeds
- Input image dimensions
- Model configurations
- Optimizer, learning rate, batch size, and epochs
- Hardware and software versions
- Best checkpoint selection criteria
- Final test-set metrics

## 🔐 Repository and Data Policy

- Dataset images are excluded from version control.
- Model checkpoints and generated results are excluded by default.
- Credentials and API tokens must never be committed.
- Only share datasets and derived materials in accordance with their licenses and usage terms.

## 🛠️ Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- scikit-learn
- Pillow
- Streamlit
- Matplotlib
- Grad-CAM / XAI techniques

Please verify the installed dependencies against `requirements.txt` and the actual implementation.

## 👨‍💻 Author

**Bhavik Nakarani**

M.Tech Research Project — Computer Engineering

GitHub: [Bhavik94290](https://github.com/Bhavik94290)

## 📄 Disclaimer

This repository is intended for academic research and educational experimentation. The models have not been established as clinically validated diagnostic systems. Predictions may be incorrect and must not be used as the sole basis for medical decisions.

## ⭐ Acknowledgements

The author acknowledges the OCT-C8 dataset contributors and the open-source deep learning community for providing resources that support retinal image classification research.
