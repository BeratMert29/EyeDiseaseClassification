import numpy as np
import matplotlib.pyplot as plt
import torch
import copy
import transformers
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, SwinForImageClassification, get_scheduler
from dataset import df, label2id, id2label, train_transform, val_transform, teacher_transform, EyeDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

num_epochs = 30
patience = 3 
batch_size = 32
n_classes = len(label2id)
num_workers = 4
labeled_ratio = 0.40
consistency_weight = 0.3

# Class-weighted cross-entropy with stronger weighting for underrepresented classes
class_counts = df["outcome"].value_counts().reindex(range(n_classes), fill_value=1)
print(f"Class distribution: {dict(zip([id2label[i] for i in range(n_classes)], class_counts.tolist()))}")

class_weights = torch.tensor(np.sqrt(1.0 / class_counts.values), dtype=torch.float)

for i, label in enumerate([id2label[j] for j in range(n_classes)]):
    if 'cataract' in label.lower():
        class_weights[i] *= 2.0  
    elif 'glaucoma' in label.lower():
        class_weights[i] *= 1.5 

class_weights = class_weights / class_weights.sum()  # Normalize
print(f"Class weights: {dict(zip([id2label[i] for i in range(n_classes)], class_weights.tolist()))}")
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.1)
consistency_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')

# Helper: collate unlabeled PIL images (returned by EyeDataset with return_pil=True)
def collate_pil_images(batch):
    images, labels = zip(*batch)
    return list(images), torch.stack(labels)

def compute_consistency_weight(current_epoch, target_weight, rampup_start_epoch, rampup_end_epoch):
    if current_epoch < rampup_start_epoch:
        return 0.0
    if current_epoch >= rampup_end_epoch:
        return target_weight
    ramp_span = max(1, rampup_end_epoch - rampup_start_epoch)
    progress = (current_epoch - rampup_start_epoch + 1) / ramp_span
    return target_weight * progress

def swin_model():
    return SwinForImageClassification.from_pretrained(
        "microsoft/swin-small-patch4-window7-224",
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    ).to(device)

def build_swin_optimizer(model, lr=2e-5, weight_decay=1e-2):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith("bias") or "LayerNorm.weight" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
    )

def update_teacher_model(student_model, teacher_model, alpha=0.998):
    for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
        teacher_param.data.mul_(alpha).add_(student_param.data, alpha=1 - alpha)

def train_epoch_mean_teacher(
    student_model, teacher_model, device, 
    labeled_loader, unlabeled_loader, optimizer, scheduler, 
    supervised_weight=1.0, consistency_weight=0.5, 
    consistency_rampup_epochs=max(1, int(0.8 * num_epochs)), current_epoch=0
):

    student_model.train()
    teacher_model.eval()
    
    total_supervised_loss = 0.0
    total_consistency_loss = 0.0
    total_loss = 0.0
    total_correct = 0
    total_labeled_samples = 0
    total_unlabeled_samples = 0
    
    rampup_start_epoch = 4
    current_consistency_weight = compute_consistency_weight(
        current_epoch=current_epoch,
        target_weight=consistency_weight,
        rampup_start_epoch=rampup_start_epoch,
        rampup_end_epoch=consistency_rampup_epochs,
    )
    
    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)
    max_batches = max(len(labeled_loader), len(unlabeled_loader))
    
    for batch_idx in range(max_batches):
        try:
            labeled_images, labeled_targets = next(labeled_iter)
        except StopIteration:
            labeled_iter = iter(labeled_loader)
            labeled_images, labeled_targets = next(labeled_iter)
        
        try:
            unlabeled_images, unlabeled_labels = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            unlabeled_images, unlabeled_labels = next(unlabeled_iter)
        
        
        labeled_images = labeled_images.to(device)
        labeled_targets = labeled_targets.to(device)
    
        optimizer.zero_grad()
        
        # Supervised loss on labeled data (student model)
        student_labeled_outputs = student_model(pixel_values=labeled_images)
        student_labeled_logits = student_labeled_outputs.logits
        supervised_loss = loss_fn(student_labeled_logits, labeled_targets)
        
        student_unlabeled_images = torch.stack([train_transform(img) for img in unlabeled_images]).to(device)
        teacher_unlabeled_images = torch.stack([teacher_transform(img) for img in unlabeled_images]).to(device)
        
        student_unlabeled_outputs = student_model(pixel_values=student_unlabeled_images)
        student_unlabeled_logits = student_unlabeled_outputs.logits
        
        # Teacher prediction on weakly augmented view (no gradients)
        with torch.no_grad():
            teacher_unlabeled_outputs = teacher_model(pixel_values=teacher_unlabeled_images)
            teacher_unlabeled_logits = teacher_unlabeled_outputs.logits
        
        teacher_probs = torch.softmax(teacher_unlabeled_logits, dim=1)
        student_log_probs = torch.log_softmax(student_unlabeled_logits, dim=1)
        consistency_loss = consistency_loss_fn(student_log_probs, teacher_probs)
        
        # Total loss
        total_batch_loss = (
            supervised_weight * supervised_loss + 
            current_consistency_weight * consistency_loss
        )
        
        total_batch_loss.backward()
        # Gradient clipping to prevent exploding gradients and improve stability
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Update teacher model with EMA
        update_teacher_model(student_model, teacher_model)
        
        # Track metrics
        batch_size_labeled = labeled_images.size(0)
        batch_size_unlabeled = len(unlabeled_images)  # unlabeled_images is a list of PIL Images
        total_supervised_loss += supervised_loss.item() * batch_size_labeled
        total_consistency_loss += consistency_loss.item() * batch_size_unlabeled
        total_loss += total_batch_loss.item() * batch_size_labeled
        total_unlabeled_samples += batch_size_unlabeled
        
        preds = student_labeled_logits.argmax(dim=1)
        total_correct += (preds == labeled_targets).sum().item()
        total_labeled_samples += labeled_images.size(0)
    
    avg_supervised_loss = total_supervised_loss / total_labeled_samples if total_labeled_samples > 0 else 0
    avg_consistency_loss = total_consistency_loss / total_unlabeled_samples if total_unlabeled_samples > 0 else 0
    avg_loss = total_loss / total_labeled_samples if total_labeled_samples > 0 else 0
    accuracy = total_correct / total_labeled_samples if total_labeled_samples > 0 else 0
    
    return avg_loss, avg_supervised_loss, avg_consistency_loss, accuracy, current_consistency_weight


def train_epoch(model, device, dataloader, optimizer, scheduler, loss_fn):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in tqdm(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(pixel_values=images)
        logits = outputs.logits

        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


def eval_epoch(model, device, dataloader, loss_fn, return_predictions=False):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(pixel_values=images)
            logits = outputs.logits

            loss = loss_fn(logits, labels)

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            
            if return_predictions:
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    if return_predictions:
        return avg_loss, accuracy, all_labels, all_preds
    return avg_loss, accuracy


consistency_rampup_epochs = max(1, int(0.8 * num_epochs))

if 'patient_id' in df.columns:
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(gss.split(df, df["outcome"], groups=df["patient_id"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)
    
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx, test_idx = next(gss2.split(temp_df, temp_df["outcome"], groups=temp_df["patient_id"]))
    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)
    
    print(f"Patient-aware split:")
    print(f"  Train: {len(train_df)} images from {train_df['patient_id'].nunique()} patients")
    print(f"  Val: {len(val_df)} images from {val_df['patient_id'].nunique()} patients")
    print(f"  Test: {len(test_df)} images from {test_df['patient_id'].nunique()} patients")

else:
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["outcome"],
        random_state=42
    )
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["outcome"],
        random_state=42
    )

# Split training data into labeled and unlabeled for semi-supervised learning
labeled_idx, unlabeled_idx = train_test_split(
    train_df.index, 
    test_size=1-labeled_ratio, 
    stratify=train_df["outcome"], 
    random_state=42
)

labeled_df = train_df.loc[labeled_idx].reset_index(drop=True)
unlabeled_df = train_df.loc[unlabeled_idx].reset_index(drop=True)

labeled_dataset = EyeDataset(labeled_df, transform=train_transform, is_labeled=True)
unlabeled_dataset = EyeDataset(unlabeled_df, transform=None, is_labeled=False, return_pil=True)
val_dataset = EyeDataset(val_df, transform=val_transform, is_labeled=True)
test_dataset = EyeDataset(test_df, transform=val_transform, is_labeled=True)

labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_pil_images)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

student_model = swin_model()
teacher_model = swin_model()

# Initialize teacher with student weights
teacher_model.load_state_dict(student_model.state_dict())

# Optimizer with grouped weight decay (no decay on bias/LayerNorm/1D params)
swin_optimizer = build_swin_optimizer(student_model, lr=2e-5, weight_decay=1e-2)

training_steps = num_epochs * max(len(labeled_loader), len(unlabeled_loader))
warmup_steps = int(0.1 * training_steps)

swin_scheduler = get_scheduler(
    name="cosine",
    optimizer=swin_optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=training_steps,
)

# Training history
history = {
    "train_loss": [],
    "train_acc": [],
    "supervised_loss": [],
    "consistency_loss": [],
    "consistency_weight": [],
    "val_loss": [],
    "val_acc": [],
    "lr": [],
}

best_val_loss = float("inf")
best_val_acc = 0.0
best_state_dict = None
epochs_without_improvement = 0

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")

    # Mean Teacher training
    train_loss, supervised_loss, consistency_loss, train_acc, current_consistency_weight = train_epoch_mean_teacher(
        student_model, teacher_model, device,
        labeled_loader, unlabeled_loader, swin_optimizer, swin_scheduler,
        supervised_weight=1.0,
        consistency_weight=consistency_weight,
        consistency_rampup_epochs=consistency_rampup_epochs,
        current_epoch=epoch
    )
    
    student_model.eval()
    student_val_loss, student_val_acc = eval_epoch(student_model, device, val_loader, loss_fn)
    
    teacher_model.eval()
    teacher_val_loss, teacher_val_acc = eval_epoch(teacher_model, device, val_loader, loss_fn)
    
    train_eval_dataset = EyeDataset(labeled_df, transform=val_transform, is_labeled=True)
    train_eval_loader = DataLoader(train_eval_dataset, batch_size=batch_size, shuffle=False)
    train_eval_loss, train_eval_acc = eval_epoch(student_model, device, train_eval_loader, loss_fn)

    # record history
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["supervised_loss"].append(supervised_loss)
    history["consistency_loss"].append(consistency_loss)
    history["consistency_weight"].append(current_consistency_weight)
    history["val_loss"].append(teacher_val_loss)  # Use teacher val loss for best model selection
    history["val_acc"].append(teacher_val_acc)
    current_lr = swin_optimizer.param_groups[0]["lr"]
    history["lr"].append(current_lr)

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Sup Loss: {supervised_loss:.4f} | Cons Loss: {consistency_loss:.4f} | "
        f"Cons Weight: {current_consistency_weight:.2f} | "
        f"Train Acc (aug): {train_acc:.4f} | Train Acc (eval): {train_eval_acc:.4f} | "
        f"Student Val: {student_val_acc:.4f} | Teacher Val: {teacher_val_acc:.4f}"
    )

    if teacher_val_loss < best_val_loss:
        best_val_loss = teacher_val_loss
        best_val_acc = teacher_val_acc
        best_state_dict = copy.deepcopy(teacher_model.state_dict())  # Deep copy to avoid reference issues
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(
                f"\nEarly stopping at epoch {epoch + 1} "
                f"(no val_loss improvement for {patience} epochs)."
            )
            break

# Load best model for test evaluation
if best_state_dict is not None:
    teacher_model.load_state_dict(best_state_dict)
    torch.save({
        'model_state_dict': best_state_dict,
        'label2id': label2id,
        'id2label': id2label,
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc,
        'model_name': 'microsoft/swin-small-patch4-window7-224',  # Save model name for loading
        'transformers_version': transformers.__version__,  # Save transformers version
    }, 'best_eye_disease_model.pt')
    print("Model saved to 'best_eye_disease_model.pt'")

print(
    f"\nBest validation: Val Loss={best_val_loss:.4f}, Val Acc={best_val_acc:.4f}"
)

test_loss, test_acc, y_true, y_pred = eval_epoch(
    teacher_model, device, test_loader, loss_fn, return_predictions=True
)
print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

# Training curves
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
epochs = range(1, len(history["train_loss"]) + 1)

# Loss curves
axes[0, 0].plot(epochs, history["train_loss"], 'b-', label='Train Loss')
axes[0, 0].plot(epochs, history["val_loss"], 'r-', label='Val Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training Curves - Loss')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Accuracy curves
axes[0, 1].plot(epochs, history["train_acc"], 'b-', label='Train Acc')
axes[0, 1].plot(epochs, history["val_acc"], 'r-', label='Val Acc')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].set_title('Training Curves - Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Supervised and consistency loss
axes[1, 0].plot(epochs, history["supervised_loss"], 'g-', label='Supervised Loss')
axes[1, 0].plot(epochs, history["consistency_loss"], 'm-', label='Consistency Loss')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].set_title('Semi-Supervised Loss Components')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Consistency weight over time
axes[1, 1].plot(epochs, history["consistency_weight"], 'orange', label='Consistency Weight')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Weight')
axes[1, 1].set_title('Consistency Weight Rampup')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[id2label[i] for i in range(len(id2label))]
)
fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(cmap=plt.cm.Blues, ax=ax)
ax.set_title("Confusion Matrix (Test Set)")
plt.show()

# Classification Report
print("\n" + "="*60)
print("CLASSIFICATION REPORT (Test Set)")
print("="*60)
print(classification_report(
    y_true, 
    y_pred, 
    target_names=[id2label[i] for i in range(len(id2label))],
    digits=4
))