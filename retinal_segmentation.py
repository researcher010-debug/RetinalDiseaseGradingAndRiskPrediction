
import json
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from architecture import SGMODLFSegmentation, reset_random

IMAGE_DIR = Path("Data/Preprocessed/Images")
MASK_DIR = Path("Data/Preprocessed/Masks")
MODEL_DIR = Path("model")
NUM_CLASSES = 6
IMAGE_SIZE = 512
BATCH_SIZE = 4
EPOCHS = 200
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

class RetinalSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        if not self.image_dir.exists() or not self.mask_dir.exists():
            raise FileNotFoundError("Segmentation image or mask directory not found")
        mask_paths = {p.relative_to(self.mask_dir).with_suffix("").as_posix(): p for p in self.mask_dir.rglob("*") if p.suffix.lower() in VALID_EXTENSIONS}
        self.samples = []
        for image_path in sorted(p for p in self.image_dir.rglob("*") if p.suffix.lower() in VALID_EXTENSIONS):
            key = image_path.relative_to(self.image_dir).with_suffix("").as_posix()
            if key in mask_paths:
                self.samples.append((image_path, mask_paths[key]))
        if not self.samples:
            raise RuntimeError(f"No matching image-mask pairs found in {self.image_dir} and {self.mask_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if image is None or mask is None:
            raise RuntimeError(f"Unable to read image or mask: {image_path}")
        if mask.ndim == 3:
            if not np.array_equal(mask[:, :, 0], mask[:, :, 1]) or not np.array_equal(mask[:, :, 1], mask[:, :, 2]):
                raise ValueError(f"Color-encoded mask requires an explicit color-to-class mapping: {mask_path}")
            mask = mask[:, :, 0]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(np.int64)
        if mask.max() >= NUM_CLASSES or mask.min() < 0:
            raise ValueError(f"Mask labels must be between 0 and {NUM_CLASSES - 1}: {mask_path}")
        image = torch.from_numpy(image.transpose(2, 0, 1).copy()).float() / 255.0
        mask = torch.from_numpy(mask.copy()).long()
        return image, mask

class DiceCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        probabilities = torch.softmax(logits, dim=1)
        one_hot = nn.functional.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()
        intersection = (probabilities * one_hot).sum(dim=(0, 2, 3))
        denominator = probabilities.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1.0 - dice.mean()
        return ce_loss + dice_loss

def calculate_metrics(predictions, targets):
    predictions = predictions.reshape(-1)
    targets = targets.reshape(-1)
    pixel_accuracy = (predictions == targets).float().mean().item()
    ious, dices = [], []
    for class_id in range(NUM_CLASSES):
        pred_class = predictions == class_id
        target_class = targets == class_id
        intersection = (pred_class & target_class).sum().float()
        union = (pred_class | target_class).sum().float()
        denominator = pred_class.sum().float() + target_class.sum().float()
        if union.item() > 0:
            ious.append((intersection / union).item())
        if denominator.item() > 0:
            dices.append((2.0 * intersection / denominator).item())
    return pixel_accuracy, float(np.mean(ious)) if ious else 0.0, float(np.mean(dices)) if dices else 0.0

def run_epoch(model, loader, criterion, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss, total_accuracy, total_iou, total_dice, total_samples = 0.0, 0.0, 0.0, 0.0, 0
    progress = tqdm(loader, desc="Training" if training else "Testing", leave=False)
    for images, masks in progress:
        images, masks = images.to(DEVICE, non_blocking=True), masks.to(DEVICE, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, masks)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        predictions = torch.argmax(logits.detach(), dim=1)
        accuracy, iou, dice = calculate_metrics(predictions, masks)
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_accuracy += accuracy * batch_size
        total_iou += iou * batch_size
        total_dice += dice * batch_size
        total_samples += batch_size
        progress.set_postfix(loss=f"{loss.item():.4f}", IoU=f"{iou:.4f}", Dice=f"{dice:.4f}")
    return {
        "loss": total_loss / total_samples,
        "pixel_accuracy": total_accuracy / total_samples,
        "iou": total_iou / total_samples,
        "dice": total_dice / total_samples
    }

def train_segmentation():
    reset_random(42)
    train_dataset = RetinalSegmentationDataset(IMAGE_DIR / "train", MASK_DIR / "train")
    test_dataset = RetinalSegmentationDataset(IMAGE_DIR / "test", MASK_DIR / "test")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=DEVICE.type == "cuda")
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=DEVICE.type == "cuda")
    model = SGMODLFSegmentation(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = DiceCrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    history = {"train_loss": [], "test_loss": [], "train_pixel_accuracy": [], "test_pixel_accuracy": [], "train_iou": [], "test_iou": [], "train_dice": [], "test_dice": []}
    best_iou = -1.0
    for epoch in range(EPOCHS):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, scaler)
        test_metrics = run_epoch(model, test_loader, criterion)
        for key in train_metrics:
            history[f"train_{key}"].append(train_metrics[key])
            history[f"test_{key}"].append(test_metrics[key])
        if test_metrics["iou"] > best_iou:
            best_iou = test_metrics["iou"]
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_test_iou": best_iou,
                "test_metrics": test_metrics
            }, MODEL_DIR / "retinal_segmentation_best.pth")
        with open(MODEL_DIR / "retinal_segmentation_history.json", "w") as file:
            json.dump(history, file, indent=4)
        print(f"Epoch [{epoch + 1}/{EPOCHS}] Train Loss: {train_metrics['loss']:.4f} | Test Loss: {test_metrics['loss']:.4f} | Train IoU: {train_metrics['iou']:.4f} | Test IoU: {test_metrics['iou']:.4f} | Test Dice: {test_metrics['dice']:.4f}")
    print(f"Best test IoU: {best_iou:.4f}")
    return model, history

if __name__ == "__main__":
    train_segmentation()
