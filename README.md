# MRI Alzheimer's Classifier

A multi-class CNN to classify brain MRI scans into 4 stages of Alzheimer's disease.

## Overview

This project uses a Convolutional Neural Network to identify Alzheimer's disease progression from brain MRI images. The model classifies scans into:

- Non-Demented (healthy)
- Very Mild Demented
- Mild Demented
- Moderate Demented

**Test Accuracy: 96.67%**

## Dataset

- **Source:** Kaggle Alzheimer's MRI Dataset
- **Total Images:** 6,400
- **Classes:**
  - Non_Demented: 3,200 (50%)
  - Very_Mild_Demented: 2,240 (35%)
  - Mild_Demented: 896 (14%)
  - Moderate_Demented: 64 (1%)

## Project Structure

```
mri-classifier/
├── data/                  # MRI images organized by class
├── results/               # Saved model and visualizations
├── src/
│   ├── preprocessing.py   # Image loading and normalization
│   ├── data_loader.py     # Dataset loading and splitting
│   ├── model.py           # CNN architecture
│   ├── train.py           # Training script
│   ├── evaluate.py        # Evaluation and metrics
│   └── visualize.py       # Generate charts
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/RadiantBunny633/mri-classifier.git
cd mri-classifier
pip install -r requirements.txt
```

## Usage

**Test preprocessing:**

```bash
python3 src/preprocessing.py
```

**Load and split dataset:**

```bash
python3 src/data_loader.py
```

**Train the model:**

```bash
python3 src/train.py
```

**Evaluate on test set:**

```bash
python3 src/evaluate.py
```

**Generate visualizations:**

```bash
python3 src/visualize.py
```
