import torch
from tqdm import tqdm
import wandb
import gc

def train_epoch(model, data_loader, optimizer, loss_fn, active_branches, device, aux_loss_weight=0.3, enable_tracker=False):
    """
    Train the model for one epoch.
    Returns: avg_loss, avg_acc
    """
    model.train()
    
    # Stability settings for Laptop GPUs
    torch.backends.cudnn.enabled = False   # Nuclear option for cuDNN internal errors
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # Clear cache at start of epoch
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    
    train_loss = 0.0
    correct_train = 0.0
    total_samples = 0
    
    # Create progress bar
    pbar = tqdm(data_loader, desc="Training", leave=False)
    
    for batch_idx, (file_name, x1, x2, x3, y) in enumerate(pbar):
        optimizer.zero_grad()

        x1 = x1.to(device)
        x2 = x2.to(device)
        x3 = x3.to(device)
        label = y.to(device).long()

        # Forward pass
        output, aux_output = model([x1, x2, x3], active_branch_indices=active_branches)
        
        # Calculate loss
        main_loss = loss_fn(output, label)
        
        # Auxiliary loss (from branches)
        aux_loss = 0.0
        # aux_output corresponds to active_branches. 
        # Wait, model returns aux_outputs list consistently?
        # Model returns aux_outputs list of length = number of active branches?
        # Let's check model code.
        # "aux = branch(x) ... aux_outputs.append(aux)" inside the loop over branches.
        # The loop iterates over all branches but appends zeros if inactive.
        # So aux_outputs has length == len(model.branches).
        # But we only want to train active branches.
        # Inactive branches output zeros, so calculating loss on them might be weird if labels are not relevant?
        # But wait, original code did: "aux_loss = sum(loss_fn(aux, label) for aux in aux_output)" 
        # If aux is zeros, loss_fn might be high/garbage?
        # In original code, active_branches logic:
        # if idx in active_branches: feat, aux = branch(x) ... 
        # else: feat=zeros, aux=zeros.
        # So inactive branches return zero predictions.
        # If we optimize this, we try to force zeros to match labels? That's bad.
        # We should only sum loss for active branches.
        
        for idx, aux in enumerate(aux_output):
             if idx in active_branches:
                 aux_loss += loss_fn(aux, label)
        
        total_loss = main_loss + aux_loss_weight * aux_loss

        # Backward pass
        total_loss.backward()
        optimizer.step()

        # Metrics
        preds = torch.argmax(output, dim=1)
        correct_train += (preds == label).sum().item()
        train_loss += total_loss.item()
        total_samples += label.size(0)
        
        # Update progress bar
        current_acc =  (preds == label).float().mean().item()
        pbar.set_postfix({'loss': total_loss.item(), 'acc': current_acc})
        
        if enable_tracker and batch_idx % 10 == 0:
             wandb.log({"batch_train_loss": total_loss.item(), "batch_train_acc": current_acc})
        
        # Cleanup
        del x1, x2, x3, output, aux_output, label, total_loss

    avg_loss = train_loss / len(data_loader)
    avg_acc = correct_train / total_samples

    return avg_loss, avg_acc
