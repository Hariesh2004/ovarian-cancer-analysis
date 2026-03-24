import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class OvarianCancerDataset(Dataset):
    def __init__(self, root_dir, split, feature_extractor=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            split (string): 'train', 'valid', 'test' or 'Train_Images', 'Test_Images'.
            feature_extractor (callable, optional): transforms/processor.
        """
        kaggle_map = {
            'train': 'Train_Images',
            'test': 'Test_Images',
            'valid': 'Valid_Images'
        }
        
        target_dir = os.path.join(root_dir, split)
        if not os.path.exists(target_dir) and split in kaggle_map:
            target_dir = os.path.join(root_dir, kaggle_map[split])

        self.split_dir = target_dir
        self.feature_extractor = feature_extractor
        self.split = split.lower()
        
        if not os.path.exists(self.split_dir):
            self.classes = []
            self.samples = []
            return

        self.classes = sorted([d for d in os.listdir(self.split_dir) if os.path.isdir(os.path.join(self.split_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(self.split_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((img_path, self.class_to_idx[cls_name]))

        # Enhanced Augmentation for Medical Images
        if 'train' in self.split:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.feature_extractor:
            # For ViT, we primarily use the feature extractor but apply our custom transforms first
            # to ensure the model sees "augmented" pixels
            if 'train' in self.split:
                # Apply rotation/flips before converting to tensor via processor
                image = transforms.RandomHorizontalFlip()(image)
                image = transforms.RandomVerticalFlip()(image)
            
            encoded_inputs = self.feature_extractor(images=image, return_tensors="pt")
            item = {key: val.squeeze(0) for key, val in encoded_inputs.items()}
            item['labels'] = label
            return item
        else:
            image = self.transform(image)
            return {"pixel_values": image, "labels": label}

# Quick test
if __name__ == "__main__":
    dataset_path = r"c:\Users\pkshi\Downloads\OVARIAN CANCER BIOPSY ANALYSIS.v4-ovarian-cancer-histopathology.folder"
    train_dataset = OvarianCancerDataset(dataset_path, 'train')
    print(f"Number of training samples: {len(train_dataset)}")
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        print(f"Sample shape: {sample['pixel_values'].shape}, Label: {sample['labels']}")
