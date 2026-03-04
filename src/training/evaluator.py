import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

def evaluate(model, data_loader, loss_fn, active_branches, device, save_prediction=False, prediction_saving_path=None):
    """
    Evaluate the model on the given dataset.
    Returns: avg_loss, avg_acc, metrics_dict
    """
    model.eval()
    
    # Storage for predictions if saving
    if save_prediction:
        file_names = []
        prob_predicted = []
        label_predicted = []
        label_actual = []
        channel = []
        abnormality_type = []
        # data = [] # Avoid saving data in CSV to save space? Original saved it.

    eval_loss = 0.0
    correct_eval = 0.0
    
    # Class-wise correct counts
    correct_class = {0: 0, 1: 0, 2: 0}
    total_class = {0: 0, 1: 0, 2: 0}
    
    final_labels = []
    final_predictions = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            # Unpack batch
            # Dataset returns: file_name, x1, x2, x3, label
            file_name, x1, x2, x3, y = batch
            
            x1 = x1.to(device)
            x2 = x2.to(device)
            x3 = x3.to(device)
            label = y.to(device).long()

            output, aux_output = model([x1, x2, x3], active_branch_indices=active_branches)
            preds = torch.argmax(output, dim=1)

            # Metrics
            loss = loss_fn(output, label)
            eval_loss += loss.item()
            correct_eval += (preds == label).sum().item()

            # Class-wise stats
            for c in [0, 1, 2]:
                correct_class[c] += ((preds == c) & (label == c)).sum().item()
                total_class[c] += (label == c).sum().item()
            
            final_labels.extend(label.cpu().tolist())
            final_predictions.extend(preds.cpu().tolist())

            if save_prediction:
                for i in range(len(file_name)):
                    pred_probs = output[i].cpu().tolist()
                    actual_label = label[i].cpu().item()
                    predicted_label = preds[i].cpu().item()
                    
                    file_names.append(file_name[i])
                    # Parse filename for metadata if possible (from original code)
                    # "SSD 4TB_EEG_Data_channel_abnormality.npz" -> channel, abnormality
                    f_parts = file_name[i].split('_')
                    # This parsing is brittle if filenames vary. 
                    # Assuming format: ..._channel_abnormality.npz
                    try:
                        ch = f_parts[-2]
                        ab_type = f_parts[-1].split(".")[0]
                    except:
                        ch = "unknown"
                        ab_type = "unknown"
                    
                    prob_predicted.append(pred_probs)
                    label_predicted.append(predicted_label)
                    label_actual.append(actual_label)
                    channel.append(ch)
                    abnormality_type.append(ab_type)
                    # data.append(x1[i].cpu().numpy())

    avg_loss = eval_loss / len(data_loader)
    avg_acc = correct_eval / len(data_loader.dataset)

    # Detailed metrics
    precision = precision_score(final_labels, final_predictions, average=None, labels=[0, 1, 2], zero_division=0)
    recall = recall_score(final_labels, final_predictions, average=None, labels=[0, 1, 2], zero_division=0)
    f1 = f1_score(final_labels, final_predictions, average=None, labels=[0, 1, 2], zero_division=0)
    cm = confusion_matrix(final_labels, final_predictions, labels=[0, 1, 2])
    
    # Macro averages
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()

    metrics = {
        "loss": avg_loss,
        "accuracy": avg_acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "cm": cm.tolist(),
        "class_0_acc": correct_class[0] / max(total_class[0], 1),
        "class_1_acc": correct_class[1] / max(total_class[1], 1),
        "class_2_acc": correct_class[2] / max(total_class[2], 1),
        "precision_0": precision[0],
        "precision_1": precision[1],
        "precision_2": precision[2],
        "recall_0": recall[0],
        "recall_1": recall[1],
        "recall_2": recall[2],
        "f1_0": f1[0],
        "f1_1": f1[1],
        "f1_2": f1[2],
    }

    if save_prediction and prediction_saving_path:
        pred_df = pd.DataFrame({
            "file_name": file_names,
            "channel": channel,
            "abnormality_type": abnormality_type,
            "label_actual": label_actual,
            "label_predicted": label_predicted,
            "predicted_prob": prob_predicted
        })
        pred_df.to_csv(prediction_saving_path, index=False)
        print(f"Predictions saved to {prediction_saving_path}")

    return avg_loss, avg_acc, metrics
