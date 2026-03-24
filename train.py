import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from transformers import ViTForImageClassification, TrainingArguments, Trainer
from transformers import ViTImageProcessor
from dataset import OvarianCancerDataset
from torch import nn

# Check if CUDA is reachable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Path to the downloaded Kaggle dataset
dataset_path = r"C:\Users\pkshi\.cache\kagglehub\datasets\sunilthite\ovarian-cancer-classification-dataset\versions\1"
model_name_or_path = 'google/vit-base-patch16-224-in21k'

# Load feature extractor from Hugging Face
feature_extractor = ViTImageProcessor.from_pretrained(model_name_or_path)

# Initialize datasets
# Note: Kaggle dataset has Train_Images and Test_Images. 
# We'll use Test_Images as our validation set for now.
train_dataset = OvarianCancerDataset(dataset_path, 'train', feature_extractor=feature_extractor)
valid_dataset = OvarianCancerDataset(dataset_path, 'test', feature_extractor=feature_extractor)

num_classes = len(train_dataset.classes)
print(f"Detected Classes focus: {train_dataset.classes}")

# --- CPU Optimization: Subsetting Data ---
def get_subset(dataset, max_per_class=500):
    import random
    from collections import defaultdict
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(dataset.samples[idx])
    
    new_samples = []
    for label in class_indices:
        samples = class_indices[label]
        random.shuffle(samples)
        new_samples.extend(samples[:max_per_class])
    
    random.shuffle(new_samples)
    dataset.samples = new_samples
    return dataset

print("Applying CPU Optimization: Subsetting to 500 samples per class...")
train_dataset = get_subset(train_dataset, max_per_class=500)
valid_dataset = get_subset(valid_dataset, max_per_class=100)

print(f"Optimized Train samples: {len(train_dataset)}, Valid samples: {len(valid_dataset)}")

# --- Handle Class Imbalance ---
# Calculate class frequency
labels = [sample[1] for sample in train_dataset.samples]
class_counts = np.bincount(labels, minlength=num_classes)
print(f"Class counts: {class_counts}")

# Compute weights (Inverse Frequency)
# weight = total_samples / (num_classes * count)
weights = len(labels) / (num_classes * class_counts)
class_weights = torch.tensor(weights, dtype=torch.float).to(device)
print(f"Computed Class Weights: {weights}")

# Custom Trainer to inject Weighted CrossEntropyLoss
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # Compute custom loss
        loss_fct = nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# Load pre-trained ViT
model = ViTForImageClassification.from_pretrained(
    model_name_or_path,
    num_labels=num_classes,
    id2label={str(i): c for i, c in enumerate(train_dataset.classes)},
    label2id={c: str(i) for i, c in enumerate(train_dataset.classes)}
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }

training_args = TrainingArguments(
    output_dir="./vit-ovarian-cancer",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=1, # Reduced for fast CPU feedback
    fp16=torch.cuda.is_available(),
    logging_steps=10,
    learning_rate=3e-5, # Slightly adjusted
    weight_decay=0.01,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    load_best_model_at_end=True,
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
    processing_class=feature_extractor,
)

if __name__ == "__main__":
    print("Starting training with Weighted Loss to handle class imbalance...")
    trainer.train()
    
    print("Evaluating on validation set...")
    metrics = trainer.evaluate()
    print("Metrics:", metrics)
    
    trainer.save_model("./vit-ovarian-cancer-final")
    print("Model saved to ./vit-ovarian-cancer-final")
