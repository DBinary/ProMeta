import os
import random
import json
import datetime
import numpy as np
import torch
import torchmetrics
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

def compute_task_metrics(preds, labels, threshold=0.5):
    """Computes binary classification metrics."""
    try:
        preds_t = preds.clone().detach().cpu().float().view(-1) 
        labels_t = labels.clone().detach().cpu().int().view(-1)
        
        if len(torch.unique(labels_t)) < 2: 
            return None
        
        auroc = torchmetrics.functional.auroc(preds_t, labels_t, task='binary')
        auprc = torchmetrics.functional.average_precision(preds_t, labels_t, task='binary')
        
        preds_bin = (preds_t.numpy() >= threshold).astype(int)
        labels_np = labels_t.numpy()
        
        precision = precision_score(labels_np, preds_bin, zero_division=0)
        recall = recall_score(labels_np, preds_bin, zero_division=0)
        
        return {
            "auroc": float(auroc), 
            "auprc": float(auprc),
            "f1": f1_score(labels_np, preds_bin),
            "accuracy": accuracy_score(labels_np, preds_bin),
            "precision": float(precision),
            "recall": float(recall)
        }
    except Exception as e:
        print(f"[Error in Metrics] {e}") 
        return None

def plot_and_save_curves(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
    axes[0].set_title('Training Loss')
    axes[0].set_ylabel('Loss')
    
    axes[1].plot(epochs, history['val_auroc'], 'g-o', label='Val AUROC')
    axes[1].set_title('Validation AUROC')
    
    axes[2].plot(epochs, history['val_auprc'], 'm-o', label='Val AUPRC')
    axes[2].set_title('Validation AUPRC')
    
    for ax in axes:
        ax.set_xlabel('Epochs')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close() 
    print(f"📊 Training curves saved to: {save_path}")

def save_results(summary, task_results, args, model_name, output_dir, history=None):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(output_dir, "benchmark_results", f"support_{args.support_size}")
    os.makedirs(result_dir, exist_ok=True)
    
    filename_base = f"{model_name}_seed{args.random_seed}_{timestamp}"
    json_path = os.path.join(result_dir, f"{filename_base}.json")
    plot_path = os.path.join(result_dir, f"{filename_base}.png")

    plot_saved = False
    if history is not None:
        try:
            plot_and_save_curves(history, plot_path)
            plot_saved = True
        except Exception as e:
            print(f"[Warning] Could not plot curves: {e}")

    data = {
        "model": model_name, 
        "experiment_name": 'ProMeta', 
        "config": vars(args), 
        "timestamp": timestamp,
        "paths": {
            "json": json_path,
            "plot": plot_path if plot_saved else None
        },
        "summary_metrics": summary, 
        "per_task_details": task_results
    }
    
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
        
    print(f"💾 Results JSON saved to: {json_path}")