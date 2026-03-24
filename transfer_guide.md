# Project Transfer & Setup Guide

This guide explains how to set up the **Ovarian Cancer Histopathology AI** on a new device.

## 1. Prerequisites
- **Python 3.10+** (Recommended)
- **Internet Access** (To download the Kaggle dataset)

## 2. Setting Up the Environment
Open a terminal in the project folder and run:
```bash
pip install -r requirements.txt
```

## 3. Getting the Dataset
The project uses the *unzipped* Kaggle dataset (~3GB). Run this script once to download it to your local cache:
```python
import kagglehub
kagglehub.dataset_download("sunilthite/ovarian-cancer-classification-dataset")
```

## 4. Running the Application
Launch the Streamlit app to start analyzing biopsy scans:
```bash
python -m streamlit run app.py
```

## 5. Folder Structure Summary
- `app.py`: Main application interface.
- `model_weights/`: Optimized AI model (Consolidated).
- `train.py`: Modified script for optimized CPU training.
- `requirements.txt`: List of mandatory Python packages.
- `sample_test_images/`: Provided histological slides for verification.

---
**Note:** If you have a GPU on the new device, the app will automatically detect it and run even faster!
