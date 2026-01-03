import torch

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("PyTorch CUDA version:", torch.version.cuda)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea
import os
from tqdm import tqdm
import torch
from torchsummary import summary
import torchmetrics

sea.set_style('whitegrid')
np.__version__

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device



import kagglehub
import os

# Download latest version
path = kagglehub.dataset_download("gunavenkatdoddi/eye-diseases-classification")
print("Path to dataset files:", path)
DATASET_PATH = os.path.join(path, 'dataset')

print("Contents of base path:", os.listdir(path))
print("Contents of dataset path:", os.listdir(DATASET_PATH))

PATH = DATASET_PATH # Use the dynamically determined path
label2id = {}
for i, label in enumerate(sorted(os.listdir(PATH))):
    label2id[label] = i

id2label = {key : value for (value, key) in label2id.items()}

filenames, outcome = [], []

for label in tqdm(os.listdir(PATH)):
    for img in os.listdir(os.path.join(PATH, label)):
        filenames.append(os.path.join(PATH, label, img))
        outcome.append(label2id[label])


df = pd.DataFrame({
    "filename" : filenames,
    "outcome" : outcome
})

df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.head()

plt.figure(figsize=(10, 6))
sea.countplot(x='outcome', data=df, palette='Blues')
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
    img = (img - img.min())/img.max()
    return img

counter = 0

plt.figure(figsize = (10, 12))

for i in range(4):
    for path in df[df['outcome'] == i].sample(n = 3)['filename']:
        plt.subplot(4, 3, counter + 1)
        img = load_image(path)
        plt.imshow(img)
        plt.axis('off')
        plt.title('Class:' + " " + id2label[i])
        counter += 1

plt.show()

from PIL import Image
import numpy as np
import cv2

class ApplyCLAHE(object):
    def __init__(self, clip=2.0, grid=(8,8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)

    def __call__(self, img):
        img = np.array(img)
        if len(img.shape) == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l = self.clahe.apply(l)
            lab = cv2.merge((l,a,b))
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(img)

import torchvision.transforms as transforms
import torchvision.transforms as T

IMG_SIZE = 300
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

train_transform = T.Compose([
    ApplyCLAHE(clip=2.0, grid=(8,8)),
    T.RandomResizedCrop(IMG_SIZE, scale=(0.80, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(15),
    T.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.9, 1.1)),
    T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05, hue=0.02),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

val_transform = T.Compose([
    ApplyCLAHE(clip=2.0, grid=(8,8)),
    T.Resize((300,300)),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

from torch.utils.data import Dataset

class EyeDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        path = self.df.iloc[index, 0]
        label = self.df.iloc[index, 1]

        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(label, dtype=torch.long)
        return img, label


from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

from transformers import AutoImageProcessor, SwinForImageClassification, get_scheduler
loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

import numpy as np

def train_epoch(model, device, dataloader, optimizer, loss_fn):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    lrs = []

    for images, labels in tqdm(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)

        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        # LR log (scheduler her batch'te ilerliyor)
        lrs.append(optimizer.param_groups[0]["lr"])

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    avg_lr = float(np.mean(lrs)) if len(lrs) else optimizer.param_groups[0]["lr"]

    return avg_loss, accuracy, avg_lr

import torch
import numpy as np

@torch.no_grad()
def eval_epoch(model, device, dataloader, loss_fn, return_outputs=True):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    all_labels = []
    all_preds = []
    all_probs = []

    for images, labels in tqdm(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = loss_fn(logits, labels)

        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        total_loss += loss.item() * images.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        if return_outputs:
            all_labels.append(labels.detach().cpu())
            all_preds.append(preds.detach().cpu())
            all_probs.append(probs.detach().cpu())

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    if return_outputs:
        y_true = torch.cat(all_labels).numpy()
        y_pred = torch.cat(all_preds).numpy()
        y_prob = torch.cat(all_probs).numpy()
        return avg_loss, accuracy, y_true, y_pred, y_prob

    return avg_loss, accuracy

from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader

batch_size = 48
freeze_epochs = 5
finetune_epochs = 10
n_splits = 5
patience = 3

dev_df, test_df = train_test_split(
    df,
    test_size=0.20,
    stratify=df["outcome"],
    random_state=42
)

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
indices = np.arange(len(dev_df))

fold_histories = []
fold_summaries = []
cv_val_accs = []
cv_val_losses = []

for fold, (train_idx, val_idx) in enumerate(skf.split(indices, dev_df["outcome"])):
    print(f"\nFold {fold + 1}/{n_splits}")

    train_df = dev_df.iloc[train_idx].reset_index(drop=True)
    val_df = dev_df.iloc[val_idx].reset_index(drop=True)

    train_dataset = EyeDataset(train_df, transform=train_transform)
    val_dataset   = EyeDataset(val_df, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    weights = EfficientNet_B3_Weights.IMAGENET1K_V1
    model = efficientnet_b3(weights=weights).to(device)

    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, len(label2id)).to(device)
    model = model.to(device)

    # freezing
    for p in model.features.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=1e-03, 
        weight_decay=1e-4
    )

    best_val_loss = float("inf")
    best_val_acc = 0.0
    epochs_without_improvement = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_outputs = None

    # freezing epochs
    for epoch in range(freeze_epochs):
        train_loss, train_acc, avg_lr = train_epoch(model, device, train_loader, optimizer, loss_fn)
        val_loss, val_acc = eval_epoch(model, device, val_loader, loss_fn, return_outputs=False)
        
        print(f"[Fold {fold+1} - Freeze Ep {epoch+1}] Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(avg_lr)

    # fine-tuning epochs
    print(f"\n{'='*40}")
    print(f"PHASE 2: Fine-Tuning Whole Model")
    print(f"{'='*40}")
    
    for p in model.features.parameters():
        p.requires_grad = True
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-04,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=finetune_epochs
    )

    for epoch in range(finetune_epochs):
        train_loss, train_acc, avg_lr = train_epoch(model, device, train_loader, optimizer, loss_fn)
        val_loss, val_acc, y_true, y_pred, y_prob = eval_epoch(model, device, val_loader, loss_fn, return_outputs=True)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(avg_lr)

        print(
            f"[Fold {fold + 1}] Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | LR: {avg_lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            epochs_without_improvement = 0
            best_outputs = (y_true, y_pred, y_prob)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    f"Early stopping on fold {fold + 1} at epoch {epoch + 1} "
                    f"(no val_loss improvement for {patience} epochs)."
                )
                break

    print(
        f"Best validation for fold {fold + 1}: "
        f"Val Loss={best_val_loss:.4f}, Val Acc={best_val_acc:.4f}"
    )

    fold_histories.append(history)
    fold_summaries.append({
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "best_outputs": best_outputs
    })
    cv_val_accs.append(best_val_acc)
    cv_val_losses.append(best_val_loss)

print("\n" + "="*60)
print("K-FOLD CROSS-VALIDATION RESULTS")
print("="*60)
print(f"Mean Val Accuracy: {np.mean(cv_val_accs):.4f} (+/- {np.std(cv_val_accs):.4f})")
print(f"Mean Val Loss: {np.mean(cv_val_losses):.4f} (+/- {np.std(cv_val_losses):.4f})")
print("="*60)

# Use full development set for final training
final_train_dataset = EyeDataset(dev_df, transform=train_transform)
final_train_loader = DataLoader(final_train_dataset, batch_size=batch_size, shuffle=True)

weights = EfficientNet_B3_Weights.IMAGENET1K_V1
final_model = efficientnet_b3(weights=weights)

final_model.classifier[1] = torch.nn.Linear(
    final_model.classifier[1].in_features,
    len(label2id)
)

final_model = final_model.to(device)

for p in final_model.features.parameters():
    p.requires_grad = False

final_optimizer = torch.optim.AdamW(
    final_model.classifier.parameters(),
    lr=1e-03,
    weight_decay=1e-4
)

best_final_loss = float("inf")
best_final_acc = 0.0
final_history = {"train_loss": [], "train_acc": [], "lr": []}

# freeze epochs
for epoch in range(freeze_epochs):
    print(f"Final Model - Freeze Epoch {epoch + 1}/{freeze_epochs}")

    train_loss, train_acc, avg_lr = train_epoch(
        final_model, device, final_train_loader, final_optimizer, loss_fn
    )
    
    final_history["train_loss"].append(train_loss)
    final_history["train_acc"].append(train_acc)
    final_history["lr"].append(avg_lr)

    print(
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | LR: {avg_lr:.2e}"
    )

print(f"\n{'='*40}")
print(f"PHASE 2: Final Model Fine-Tuning Whole Model")
print(f"{'='*40}")

for p in final_model.features.parameters():
    p.requires_grad = True

final_optimizer = torch.optim.AdamW(
    final_model.parameters(),
    lr=1e-04,
    weight_decay=1e-4
)

final_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    final_optimizer, T_max=finetune_epochs
)


for epoch in range(finetune_epochs):
    print(f"Final Model - Fine-Tune Epoch {epoch + 1}/{finetune_epochs}")

    train_loss, train_acc, avg_lr = train_epoch(
        final_model, device, final_train_loader, final_optimizer, loss_fn
    )

    final_scheduler.step()
    
    final_history["train_loss"].append(train_loss)
    final_history["train_acc"].append(train_acc)
    final_history["lr"].append(avg_lr)

    print(
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | LR: {avg_lr:.2e}"
    )

print("\n" + "="*60)
print("STEP 3: FINAL EVALUATION (on Held-Out Test Set)")
print("="*60 + "\n")

test_dataset = EyeDataset(test_df, transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

test_loss, test_acc, y_true, y_pred, y_prob = eval_epoch(
    final_model, device, test_loader, loss_fn, return_outputs=True
)

print(f"Final Test Loss: {test_loss:.4f} | Final Test Accuracy: {test_acc:.4f}")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# Training curves (Final Model)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

epochs = range(1, len(final_history["train_loss"]) + 1)
axes[0].plot(epochs, final_history["train_loss"], 'b-', label='Train Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Final Model Training - Loss')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(epochs, final_history["train_acc"], 'b-', label='Train Acc')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Final Model Training - Accuracy')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

# Confusion Matrix (Final Test Set)
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[id2label[i] for i in range(len(id2label))]
)
fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(cmap=plt.cm.Blues, ax=ax)
ax.set_title("Confusion Matrix (Final Test Set - Held-Out)")
plt.show()

# Classification Report (Final Test Set)
print("\n" + "="*60)
print("FINAL CLASSIFICATION REPORT (Held-Out Test Set)")
print("="*60)
print(classification_report(
    y_true,
    y_pred,
    target_names=[id2label[i] for i in range(len(id2label))],
    digits=4
))
print("="*60)
print(f"\nSummary:")
print(f"  K-Fold CV Mean Val Accuracy: {np.mean(cv_val_accs):.4f} (+/- {np.std(cv_val_accs):.4f})")
print(f"  Final Test Accuracy: {test_acc:.4f}")
print("="*60)