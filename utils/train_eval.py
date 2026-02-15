import pandas as pd
import torch
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import sys


def safe_tqdm(*args, **kwargs):
    try:
        from tqdm.notebook import tqdm as notebook_tqdm
        return notebook_tqdm(*args, **kwargs)
    except ImportError:
        from tqdm import tqdm as std_tqdm
        return std_tqdm(*args, **kwargs)


def eval(model, data_loader, loss_fn, eval_batch, save_prediction, prediction_saving_path, active_branches, device, tqdm_position=0):
    model.eval()
    pred_csv = pd.DataFrame(columns=["file_name", "channel", "abnormality_type", "label_actual", "label_predicted", "predicted_prob"])
    
    if save_prediction:
        file_names = []
        prob_predicted = []
        label_predicted = []
        label_actual = []
        channel = []
        abnormality_type = []
        data = []

    eval_loss = 0.0
    correct_eval = 0.0
    correct_class_0 = 0.0
    correct_class_1 = 0.0
    correct_class_2 = 0.0
    final_labels = []
    final_predictions = []

    with torch.no_grad():
        for file_name, x1, x2, x3, y in safe_tqdm(data_loader, leave=True, position=tqdm_position, desc="Eval", disable=False):
            x1 = x1.to(device).view(x1.size(0), x1.size(2), -1)
            x2 = x2.to(device).view(x2.size(0), x2.size(2), -1)
            x3 = x3.to(device).view(x3.size(0), x3.size(2), -1)

            x1 = x1.to(device)
            x2 = x2.to(device)
            x3 = x3.to(device)
            label = y.to(device).long()

            output, aux_output = model([x1, x2, x3], active_branch_indices=active_branches)
            preds = torch.argmax(output, dim=1)

            if save_prediction:
                for i in range(len(file_name)):
                    pred_probs = output[i].cpu().tolist()
                    actual_label = label[i].cpu().item()
                    predicted_label = preds[i].cpu().item()
                    
                    file_names.append(file_name[i])
                    file_parts = file_name[i].split('_')
                    prob_predicted.append(pred_probs)
                    label_predicted.append(predicted_label)
                    label_actual.append(actual_label)
                    channel.append(file_parts[-2])
                    abnormality_type.append(file_parts[-1].split(".")[0])
                    data.append(x1[i].cpu().numpy())

            loss = loss_fn(output, label)
            eval_loss += loss.item()

            correct_class_0 += ((preds == 0) & (label == 0)).sum().item()
            correct_class_1 += ((preds == 1) & (label == 1)).sum().item()
            correct_class_2 += ((preds == 2) & (label == 2)).sum().item()
            correct_eval += (preds == label).sum().item()

            final_labels.extend(label.cpu().tolist())
            final_predictions.extend(preds.cpu().tolist())
            # time.sleep(0.1)
            sys.stdout.flush()

    if save_prediction:
        pred_csv = pd.DataFrame({
            "file_name": file_names,
            "channel": channel,
            "abnormality_type": abnormality_type,
            "label_actual": label_actual,
            "label_predicted": label_predicted,
            "predicted_prob": prob_predicted,
            "data": data
        })
        pred_csv.to_csv(prediction_saving_path, index=False)

    precision = precision_score(final_labels, final_predictions, average='macro')
    recall = recall_score(final_labels, final_predictions, average='macro')
    f1 = f1_score(final_labels, final_predictions, average='macro')
    cm = confusion_matrix(final_labels, final_predictions, labels=[0, 1, 2])

    print(f"Correct classified: {correct_eval} / {len(data_loader.dataset)}")
    print(f"Class-wise correct: Class 0 = {correct_class_0}, Class 1 = {correct_class_1}, Class 2 = {correct_class_2}")
    print(f"Precision: {precision*100:.2f}  Recall: {recall*100:.2f}  F1: {f1*100:.2f}")
    print("Confusion Matrix:\n", cm)

    eval_loss /= len(data_loader)
    eval_acc = correct_eval / len(data_loader.dataset)

    return eval_loss, eval_acc


def train(model, data_loader, optimizer, loss_fn, active_branches, device, tqdm_position=0):
    model.train()
    train_loss = 0.0
    correct_train = 0.0
    correct_class_0 = 0.0
    correct_class_1 = 0.0
    correct_class_2 = 0.0

    for z, x1, x2, x3, y in safe_tqdm(data_loader, leave=True, position=tqdm_position, desc="Training", disable=False):
        optimizer.zero_grad()

        x1 = x1.to(device).view(x1.size(0), x1.size(2), -1)
        x2 = x2.to(device).view(x2.size(0), x2.size(2), -1)
        x3 = x3.to(device).view(x3.size(0), x3.size(2), -1)

        x1 = x1.to(device)
        x2 = x2.to(device)
        x3 = x3.to(device)
        label = y.to(device).long()

        output, aux_output = model([x1, x2, x3], active_branch_indices=active_branches)
        preds = torch.argmax(output, dim=1)

        correct_class_0 += ((preds == 0) & (label == 0)).sum().item()
        correct_class_1 += ((preds == 1) & (label == 1)).sum().item()
        correct_class_2 += ((preds == 2) & (label == 2)).sum().item()
        correct_train += (preds == label).sum().item()

        main_loss = loss_fn(output, label)
        aux_loss = sum(loss_fn(aux, label) for aux in aux_output)
        total_loss = main_loss + 0.3 * aux_loss

        total_loss.backward()
        optimizer.step()
        train_loss += total_loss.item()

    total_samples = len(data_loader.dataset)
    print(f"Correct Classified: {correct_train} / {total_samples}")
    print(f"Class-wise correct: Class 0 = {correct_class_0}, Class 1 = {correct_class_1}, Class 2 = {correct_class_2}")

    train_loss /= len(data_loader)
    train_acc = correct_train / total_samples

    return train_loss, train_acc



