
import json
import random
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from architecture import SGMODLF, SGMODLFSegmentation, reset_random

DATA_DIR = Path("Data/Preprocessed/Images")
CSV_PATH = Path("Data/Source/IDRiD_Disease Grading_Training Labels.csv")
SEGMENTATION_CHECKPOINT = Path("model/retinal_segmentation_best.pth")
MODEL_DIR = Path("model")
IMAGE_COLUMN = "Image name"
DR_COLUMN = "Retinopathy grade"
DME_COLUMN = "Risk of macular edema"
IMAGE_SIZE = 512
NUM_DR_CLASSES = 5
NUM_DME_CLASSES = 3
BATCH_SIZE = 2
EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2
PRETRAINED = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

DR_LABELS = {
    "no dr": 0, "mild": 1, "moderate": 2, "severe": 3, "proliferative": 4,
    "proliferative dr": 4
}
DME_LABELS = {
    "no dme": 0, "non-csme": 1, "non csme": 1, "csme": 2
}

def encode_label(value, mapping, num_classes, column):
    if pd.isna(value):
        raise ValueError(f"Missing label in {column}")
    if isinstance(value, str):
        value = value.strip().lower()
        if value in mapping:
            return mapping[value]
        try:
            value = int(float(value))
        except ValueError as error:
            raise ValueError(f"Unknown label '{value}' in {column}") from error
    value = int(value)
    if not 0 <= value < num_classes:
        raise ValueError(f"Label {value} in {column} must be between 0 and {num_classes - 1}")
    return value

class RetinalClassificationDataset(Dataset):
    def __init__(self, dataframe, image_dir):
        self.image_dir = Path(image_dir)
        self.image_index = {p.stem.lower(): p for p in self.image_dir.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}}
        self.samples = []
        for _, row in dataframe.iterrows():
            image_name = Path(str(row[IMAGE_COLUMN])).stem.lower()
            if image_name not in self.image_index:
                raise FileNotFoundError(f"Image not found: {row[IMAGE_COLUMN]}")
            dr = encode_label(row[DR_COLUMN], DR_LABELS, NUM_DR_CLASSES, DR_COLUMN)
            dme = encode_label(row[DME_COLUMN], DME_LABELS, NUM_DME_CLASSES, DME_COLUMN)
            self.samples.append((self.image_index[image_name], dr, dme))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, dr_label, dme_label = self.samples[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Unable to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        image = torch.from_numpy(image.transpose(2, 0, 1).copy()).float() / 255.0
        return image, torch.tensor(dr_label, dtype=torch.long), torch.tensor(dme_label, dtype=torch.long)

def load_datasets():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Label CSV not found: {CSV_PATH}")
    dataframe = pd.read_csv(CSV_PATH)
    required = [IMAGE_COLUMN, DR_COLUMN, DME_COLUMN]
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}. Available columns: {list(dataframe.columns)}")
    dataframe = dataframe.dropna(subset=required).reset_index(drop=True)
    train_df, test_df = train_test_split(
        dataframe,
        test_size=0.30,
        random_state=42,
        stratify=dataframe[DR_COLUMN].astype(str)
    )
    train_dataset = RetinalClassificationDataset(train_df, DATA_DIR)
    test_dataset = RetinalClassificationDataset(test_df, DATA_DIR)
    return train_dataset, test_dataset

def load_segmentation_model():
    if not SEGMENTATION_CHECKPOINT.exists():
        raise FileNotFoundError(f"Segmentation checkpoint not found: {SEGMENTATION_CHECKPOINT}")
    model = SGMODLFSegmentation(num_classes=6).to(DEVICE)
    checkpoint = torch.load(SEGMENTATION_CHECKPOINT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model

def prepare_classifier_images(images):
    return (images - MEAN.to(images.device)) / STD.to(images.device)

def calculate_metrics(true_labels, predicted_labels, num_classes):
    return {
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "precision": precision_score(true_labels, predicted_labels, labels=list(range(num_classes)), average="macro", zero_division=0),
        "recall": recall_score(true_labels, predicted_labels, labels=list(range(num_classes)), average="macro", zero_division=0),
        "f1": f1_score(true_labels, predicted_labels, labels=list(range(num_classes)), average="macro", zero_division=0)
    }

def run_epoch(classifier, segmentation_model, loader, criterion, optimizer=None, scaler=None):
    training = optimizer is not None
    classifier.train(training)
    segmentation_model.eval()
    total_loss = 0.0
    dr_true, dr_pred, dme_true, dme_pred = [], [], [], []
    progress = tqdm(loader, desc="Training" if training else "Testing", leave=False)
    for images, dr_labels, dme_labels in progress:
        images = images.to(DEVICE, non_blocking=True)
        dr_labels = dr_labels.to(DEVICE, non_blocking=True)
        dme_labels = dme_labels.to(DEVICE, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
                segmentation_logits = segmentation_model(images)
                segmentation_maps = torch.softmax(segmentation_logits.float(), dim=1)
        classifier_images = prepare_classifier_images(images)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
                outputs = classifier(classifier_images, segmentation_maps)
                dr_loss = criterion(outputs["dr_logits"], dr_labels)
                dme_loss = criterion(outputs["dme_logits"], dme_labels)
                loss = dr_loss + dme_loss
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
        dr_predictions = outputs["dr_logits"].detach().argmax(dim=1)
        dme_predictions = outputs["dme_logits"].detach().argmax(dim=1)
        dr_true.extend(dr_labels.cpu().tolist())
        dr_pred.extend(dr_predictions.cpu().tolist())
        dme_true.extend(dme_labels.cpu().tolist())
        dme_pred.extend(dme_predictions.cpu().tolist())
        total_loss += loss.item() * images.size(0)
        progress.set_postfix(loss=f"{loss.item():.4f}")
    dr_metrics = calculate_metrics(dr_true, dr_pred, NUM_DR_CLASSES)
    dme_metrics = calculate_metrics(dme_true, dme_pred, NUM_DME_CLASSES)
    return {
        "loss": total_loss / len(loader.dataset),
        "dr": dr_metrics,
        "dme": dme_metrics,
        "accuracy": (dr_metrics["accuracy"] + dme_metrics["accuracy"]) / 2
    }

def train_model():
    reset_random(42)
    train_dataset, test_dataset = load_datasets()
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=DEVICE.type == "cuda")
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=DEVICE.type == "cuda")
    segmentation_model = load_segmentation_model()
    classifier = SGMODLF(
        num_dr_classes=NUM_DR_CLASSES,
        num_dme_classes=NUM_DME_CLASSES,
        num_seg_classes=6,
        pretrained=PRETRAINED
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(classifier.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=8)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    history = {
        "train_loss": [], "test_loss": [],
        "train_dr_accuracy": [], "test_dr_accuracy": [],
        "train_dr_precision": [], "test_dr_precision": [],
        "train_dr_recall": [], "test_dr_recall": [],
        "train_dr_f1": [], "test_dr_f1": [],
        "train_dme_accuracy": [], "test_dme_accuracy": [],
        "train_dme_precision": [], "test_dme_precision": [],
        "train_dme_recall": [], "test_dme_recall": [],
        "train_dme_f1": [], "test_dme_f1": []
    }
    best_accuracy = -1.0
    for epoch in range(EPOCHS):
        train_metrics = run_epoch(classifier, segmentation_model, train_loader, criterion, optimizer, scaler)
        test_metrics = run_epoch(classifier, segmentation_model, test_loader, criterion)
        for task in ["dr", "dme"]:
            for metric in ["accuracy", "precision", "recall", "f1"]:
                history[f"train_{task}_{metric}"].append(train_metrics[task][metric])
                history[f"test_{task}_{metric}"].append(test_metrics[task][metric])
        history["train_loss"].append(train_metrics["loss"])
        history["test_loss"].append(test_metrics["loss"])
        scheduler.step(test_metrics["accuracy"])
        if test_metrics["accuracy"] > best_accuracy:
            best_accuracy = test_metrics["accuracy"]
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": classifier.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_test_accuracy": best_accuracy,
                "test_metrics": test_metrics
            }, MODEL_DIR / "sgmodlf_best.pth")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": classifier.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }, MODEL_DIR / "sgmodlf_last.pth")
        with open(MODEL_DIR / "sgmodlf_history.json", "w") as file:
            json.dump(history, file, indent=4)
        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_metrics['loss']:.4f} | Test Loss: {test_metrics['loss']:.4f} | "
            f"DR Acc: {test_metrics['dr']['accuracy']:.4f} | DR F1: {test_metrics['dr']['f1']:.4f} | "
            f"DME Acc: {test_metrics['dme']['accuracy']:.4f} | DME F1: {test_metrics['dme']['f1']:.4f}"
        )
    print(f"Best combined test accuracy: {best_accuracy:.4f}")
    return classifier, history

if __name__ == "__main__":
    train_model()
