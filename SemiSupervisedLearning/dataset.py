import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea
import os
import re
from tqdm import tqdm
import torch
from PIL import Image
import cv2
import kagglehub
import torchvision.transforms as T
from torch.utils.data import Dataset

sea.set_style('whitegrid')

# Download latest version
path = kagglehub.dataset_download("gunavenkatdoddi/eye-diseases-classification")
print("Path to dataset files:", path)

DATASET_PATH = os.path.join(path, 'dataset')
PATH = DATASET_PATH
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

labels = sorted([d for d in os.listdir(PATH) if os.path.isdir(os.path.join(PATH, d))])
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

import re

filenames, outcome, patient_ids = [], [], []

# Load images from Kaggle dataset
for label in tqdm(labels, desc="Loading Kaggle dataset"):
    label_path = os.path.join(PATH, label)
    for img_file in os.listdir(label_path):
        if os.path.splitext(img_file.lower())[1] in IMAGE_EXTENSIONS:
            filenames.append(os.path.join(label_path, img_file))
            outcome.append(label2id[label])
            
            # Extract patient ID from filename (e.g., "left1003.jpg" or "right1003.jpg" -> "1003")
            # Handle various naming patterns: left1003, right1003, patient_1003, etc.
            # Try to extract numbers that likely represent patient ID
            numbers = re.findall(r'\d+', os.path.splitext(img_file)[0])
            if numbers:
                # Use the longest number sequence as patient ID (most likely to be the ID)
                patient_id = max(numbers, key=len)
            else:
                # Fallback: use filename without extension as patient ID
                patient_id = os.path.splitext(img_file)[0]
            # Include label to ensure unique grouping (same patient ID in different classes are separate)
            patient_ids.append(f"{label}_{patient_id}")

# Add additional classes from local directories (can have multiple paths per class)
additional_classes = [
    ("glaucoma", r"C:\Users\u\Documents\Python\481Project\glaucoma"),
    ("diabetic_retinopathy", r"C:\Users\u\Documents\Python\481Project\diabetic_retinopathy"),
    ("normal", r"C:\Users\u\Documents\Python\481Project\normal"),
    ("normal", r"C:\Users\u\Documents\Python\481Project\1_normal"),  # Additional normal class folder
    ("cataract", r"C:\Users\u\Documents\Python\481Project\cataract")
]
# Update label mappings to include new classes or merge with existing ones
for class_name, class_path in additional_classes:
    if os.path.exists(class_path):
        # Check if class already exists (folder names match exactly)
        if class_name in label2id:
            # Merge with existing class
            label_id = label2id[class_name]
            print(f"Merging {class_name} data into existing class (ID: {label_id})")
        else:
            # Create new class (shouldn't happen if folder names match)
            current_max_id = max(label2id.values()) if label2id else -1
            current_max_id += 1
            label2id[class_name] = current_max_id
            id2label[current_max_id] = class_name
            label_id = current_max_id
            print(f"Added new class: {class_name} with ID {current_max_id}")
        
        # Load images from this class directory
        if os.path.isdir(class_path):
            for img_file in tqdm(os.listdir(class_path), desc=f"Loading {class_name}"):
                if os.path.splitext(img_file.lower())[1] in IMAGE_EXTENSIONS:
                    filenames.append(os.path.join(class_path, img_file))
                    outcome.append(label_id)
                    
                    # Extract patient ID from filename
                    numbers = re.findall(r'\d+', os.path.splitext(img_file)[0])
                    if numbers:
                        patient_id = max(numbers, key=len)
                    else:
                        patient_id = os.path.splitext(img_file)[0]
                    patient_ids.append(f"{class_name}_{patient_id}")
        else:
            print(f"Warning: Path {class_path} does not exist or is not a directory")
    else:
        print(f"Warning: Path {class_path} does not exist")

df = pd.DataFrame({
    "filename" : filenames,
    "outcome" : outcome,
    "patient_id" : patient_ids
})

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

plt.figure(figsize=(10, 6))
# Create a custom color palette with distinct colors for each class
n_classes = len(id2label)
# Define distinct colors for each class type
color_palette = []
for i in range(n_classes):
    class_name = id2label[i].lower()
    if 'cataract' in class_name:
        color_palette.append('#FF6B6B')  # Red/Orange for Cataract
    elif 'diabetic' in class_name:
        color_palette.append('#4ECDC4')  # Teal for Diabetic Retinopathy
    elif 'glaucoma' in class_name:
        color_palette.append('#45B7D1')  # Blue for Glaucoma
    elif 'normal' in class_name or 'healthy' in class_name:
        color_palette.append('#96CEB4')  # Green for Normal/Healthy
    else:
        # Default color for any other classes
        color_palette.append('#FFEAA7')  # Yellow for other classes

sea.countplot(x='outcome', data=df, hue='outcome', palette=color_palette, legend=False)
plt.title('Distribution of Eye Disease Classes')
plt.xlabel('Disease Type')
plt.ylabel('Count')
plt.xticks(
    ticks=range(len(id2label)),
    labels=[id2label[i] for i in range(len(id2label))],
    rotation=20
)
plt.tight_layout()
plt.show()

def load_image(path):
    img = plt.imread(path)
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    return img

counter = 0

n_classes = len(id2label)
samples_per_class = 3
total_subplots = n_classes * samples_per_class
n_rows = (total_subplots + 2) // 3  # Calculate rows needed (3 columns)
plt.figure(figsize = (10, 4 * n_rows))

for i in range(n_classes):
    class_df = df[df['outcome'] == i]
    if len(class_df) > 0:
        sample_size = min(samples_per_class, len(class_df))
        for path in class_df.sample(n=sample_size)['filename']:
            plt.subplot(n_rows, 3, counter + 1)
            img = load_image(path)
            plt.imshow(img)
            plt.axis('off')
            plt.title('Class:' + " " + id2label[i])
            counter += 1

plt.tight_layout()
plt.show()

class ApplyCLAHE(object):
    def __init__(self, clip=2.0, grid=(8,8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)

    def __call__(self, img):
        img = np.array(img)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        if img.ndim == 2:  # grayscale
            img = self.clahe.apply(img)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:  # RGB
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l = self.clahe.apply(l)
            lab = cv2.merge((l,a,b))
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(img)

train_transform = T.Compose([
    T.Resize((224,224)),
    T.RandomHorizontalFlip(0.5),
    T.RandomAffine(degrees=12, translate=(0.05,0.05), scale=(0.92,1.08)),
    T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.08, hue=0.02),
    T.RandomApply([ApplyCLAHE()], p=0.2),
    T.ToTensor(),
    T.RandomErasing(p=0.05, scale=(0.02, 0.08), value='random'),  
    T.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
])

val_transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
])

teacher_transform = T.Compose([
    T.Resize((224,224)),
    T.RandomHorizontalFlip(0.5),
    T.ToTensor(),
    T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
])

class EyeDataset(Dataset):
    def __init__(self, df, transform=None, is_labeled=True, return_pil=False):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.is_labeled = is_labeled
        self.return_pil = return_pil  # If True, return PIL image instead of tensor for unlabeled data

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        path = self.df.iloc[index, 0]
        
        img = Image.open(path).convert("RGB")

        if self.return_pil and not self.is_labeled:
            # Return PIL image for unlabeled data (will apply augmentations in training loop)
            label = torch.tensor(-1, dtype=torch.long)
        else:
            # Normal behavior: apply transform and return tensor
            if self.transform:
                img = self.transform(img)

            if self.is_labeled:
                label = torch.tensor(self.df.iloc[index, 1], dtype=torch.long)
            else:
                # Return -1 as tensor for unlabeled data (DataLoader can't collate None)
                label = torch.tensor(-1, dtype=torch.long)
        
        return img, label

