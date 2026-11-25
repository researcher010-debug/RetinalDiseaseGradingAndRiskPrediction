
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
from architecture import SGMODLF
from model_training import (
    load_datasets, load_segmentation_model, prepare_classifier_images,
    MODEL_DIR, NUM_DR_CLASSES, NUM_DME_CLASSES, BATCH_SIZE, NUM_WORKERS, DEVICE
)

CHECKPOINT = MODEL_DIR / "sgmodlf_best.pth"
OUTPUT_DIR = MODEL_DIR / "evaluation"
DR_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
DME_NAMES = ["No DME", "Non-CSME", "CSME"]

def load_classifier():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Classifier checkpoint not found: {CHECKPOINT}")
    model = SGMODLF(
        num_dr_classes=NUM_DR_CLASSES,
        num_dme_classes=NUM_DME_CLASSES,
        num_seg_classes=6,
        pretrained=False
    ).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def collect_predictions(classifier, segmentation_model, loader):
    results = {
        "dr_true": [], "dr_pred": [], "dr_prob": [],
        "dme_true": [], "dme_pred": [], "dme_prob": []
    }
    classifier.eval()
    segmentation_model.eval()
    with torch.no_grad():
        for images, dr_labels, dme_labels in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(DEVICE, non_blocking=True)
            with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
                segmentation_logits = segmentation_model(images)
                segmentation_maps = torch.softmax(segmentation_logits.float(), dim=1)
                classifier_images = prepare_classifier_images(images)
                outputs = classifier(classifier_images, segmentation_maps)
            dr_prob = torch.softmax(outputs["dr_logits"].float(), dim=1)
            dme_prob = torch.softmax(outputs["dme_logits"].float(), dim=1)
            results["dr_true"].extend(dr_labels.numpy().tolist())
            results["dr_pred"].extend(dr_prob.argmax(dim=1).cpu().numpy().tolist())
            results["dr_prob"].extend(dr_prob.cpu().numpy().tolist())
            results["dme_true"].extend(dme_labels.numpy().tolist())
            results["dme_pred"].extend(dme_prob.argmax(dim=1).cpu().numpy().tolist())
            results["dme_prob"].extend(dme_prob.cpu().numpy().tolist())
    for key in ["dr_true", "dr_pred", "dme_true", "dme_pred"]:
        results[key] = np.asarray(results[key])
    results["dr_prob"] = np.asarray(results["dr_prob"])
    results["dme_prob"] = np.asarray(results["dme_prob"])
    return results

def calculate_metrics(y_true, y_pred, task):
    names = DR_NAMES if task == "dr" else DME_NAMES
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "quadratic_weighted_kappa": float(cohen_kappa_score(y_true, y_pred, labels=list(range(len(names))), weights="quadratic")) if task == "dr" else None,
        "classification_report": classification_report(
            y_true, y_pred, labels=list(range(len(names))),
            target_names=names, output_dict=True, zero_division=0
        )
    }

def plot_confusion_matrix(y_true, y_pred, names, title, save_path):
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(names))))
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=names, yticklabels=names)
    plt.title(title)
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_roc_curve(y_true, probabilities, names, title, save_path):
    y_binary = label_binarize(y_true, classes=list(range(len(names))))
    plt.figure(figsize=(8, 6))
    plotted = False
    for index, name in enumerate(names):
        if len(np.unique(y_binary[:, index])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_binary[:, index], probabilities[:, index])
        score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={score:.4f})")
        plotted = True
    if plotted:
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
    plt.close()

def plot_precision_recall(y_true, probabilities, names, title, save_path):
    y_binary = label_binarize(y_true, classes=list(range(len(names))))
    plt.figure(figsize=(8, 6))
    plotted = False
    for index, name in enumerate(names):
        if len(np.unique(y_binary[:, index])) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_binary[:, index], probabilities[:, index])
        score = average_precision_score(y_binary[:, index], probabilities[:, index])
        plt.plot(recall, precision, label=f"{name} (AP={score:.4f})")
        plotted = True
    if plotted:
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(title)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
    plt.close()

def evaluate_task(results, split_name, task, names):
    y_true = results[f"{task}_true"]
    y_pred = results[f"{task}_pred"]
    probabilities = results[f"{task}_prob"]
    metrics = calculate_metrics(y_true, y_pred, task)
    prefix = f"{split_name}_{task}"
    plot_confusion_matrix(
        y_true, y_pred, names,
        f"{split_name.title()} {task.upper()} Confusion Matrix",
        OUTPUT_DIR / f"{prefix}_confusion_matrix.png"
    )
    plot_roc_curve(
        y_true, probabilities, names,
        f"{split_name.title()} {task.upper()} ROC Curve",
        OUTPUT_DIR / f"{prefix}_roc_curve.png"
    )
    plot_precision_recall(
        y_true, probabilities, names,
        f"{split_name.title()} {task.upper()} Precision-Recall Curve",
        OUTPUT_DIR / f"{prefix}_pr_curve.png"
    )
    predictions = pd.DataFrame({
        "true_label": y_true,
        "predicted_label": y_pred,
        "predicted_class": [names[index] for index in y_pred],
        "confidence": probabilities.max(axis=1)
    })
    predictions.to_csv(OUTPUT_DIR / f"{prefix}_predictions.csv", index=False)
    return metrics

def evaluate_model():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_dataset, test_dataset = load_datasets()
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=DEVICE.type == "cuda")
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=DEVICE.type == "cuda")
    classifier = load_classifier()
    segmentation_model = load_segmentation_model()
    report = {}
    for split_name, loader in [("train", train_loader), ("test", test_loader)]:
        print(f"\nEvaluating {split_name} dataset...")
        results = collect_predictions(classifier, segmentation_model, loader)
        report[split_name] = {
            "dr_grading": evaluate_task(results, split_name, "dr", DR_NAMES),
            "dme_prediction": evaluate_task(results, split_name, "dme", DME_NAMES)
        }
        print(f"DR Accuracy: {report[split_name]['dr_grading']['accuracy']:.4f}")
        print(f"DR Precision: {report[split_name]['dr_grading']['precision']:.4f}")
        print(f"DR Recall: {report[split_name]['dr_grading']['recall']:.4f}")
        print(f"DR F1-score: {report[split_name]['dr_grading']['f1_score']:.4f}")
        print(f"DR QWK: {report[split_name]['dr_grading']['quadratic_weighted_kappa']:.4f}")
        print(f"DME Accuracy: {report[split_name]['dme_prediction']['accuracy']:.4f}")
        print(f"DME Precision: {report[split_name]['dme_prediction']['precision']:.4f}")
        print(f"DME Recall: {report[split_name]['dme_prediction']['recall']:.4f}")
        print(f"DME F1-score: {report[split_name]['dme_prediction']['f1_score']:.4f}")
    with open(OUTPUT_DIR / "evaluation_metrics.json", "w") as file:
        json.dump(report, file, indent=4)
    summary = []
    for split_name in ["train", "test"]:
        for task_name in ["dr_grading", "dme_prediction"]:
            metrics = report[split_name][task_name]
            summary.append({
                "Dataset": split_name,
                "Task": task_name,
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1-score": metrics["f1_score"],
                "QWK": metrics["quadratic_weighted_kappa"]
            })
    pd.DataFrame(summary).to_csv(OUTPUT_DIR / "evaluation_summary.csv", index=False)
    print(f"\nEvaluation results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    evaluate_model()
