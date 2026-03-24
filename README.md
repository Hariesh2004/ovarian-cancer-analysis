# Ovarian Cancer ViT Analysis

This repository contains a full pipeline to classify histopathological images of Ovarian Cancer into 4 main classes using a Fine-tuned Vision Transformer (ViT).

## Main Classes
As per the original dataset description, images are categorized into:
1. Epithelial Tumors
2. Germ Cell Tumors
3. Stromal Tumors
4. Small Cell Carcinoma

## Workflow & Structure
- **`dataset_cleanup.py`**: A preprocessing script that organizes the inconsistently named dataset folders into the four strict main classes. 
- **`dataset.py`**: A custom PyTorch `Dataset` that loads the images and passes them through a Hugging Face `ViTImageProcessor`.
- **`train.py`**: Fine-tunes the `google/vit-base-patch16-224-in21k` architecture using Hugging Face's `Trainer` API.
- **`predict.py`**: A convenient inference script to evaluate a single image.

## Setup Instructions

1. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Dataset Cleanup** (Required if working with the raw dataset):
   Update `dataset_path` in the scripts, then run:
   ```bash
   python dataset_cleanup.py
   ```

3. **Train the Model**:
   ```bash
   python train.py
   ```
   *This saves the best weights into `./vit-ovarian-cancer-final`.*

4. **Run Inference**:
   ```bash
   python predict.py <path_to_test_image>
   ```

## Requirements
- Python 3.8+
- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- scikit-learn
- accelerate

*Original dataset was provided as an AI laboratory project by Linette Dannah Cartagena under CC BY 4.0.*
