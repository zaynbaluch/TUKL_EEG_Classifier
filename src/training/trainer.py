import torch
from tqdm import tqdm
import gc

def train_epoch(model, data_loader, optimizer, loss_fn, active_branches, device, 
                aux_loss_weight=0.3, writer=None, epoch=0, global_step=0):
    """
    Train the model for one epoch.
    
    Args:
        model: The model to train
        data_loader: Training data loader
        optimizer: Optimizer
        loss_fn: Loss function
        active_branches: List of active branch indices
        device: torch.device
        aux_loss_weight: Weight for auxiliary branch losses
        writer: TensorBoard SummaryWriter (optional)
        epoch: Current epoch number (for TensorBoard logging)
        global_step: Running global step counter (for batch-level logging)
        
    Returns: avg_loss, avg_acc, global_step
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
    
    # Track per-batch losses for detailed logging
    main_losses = []
    aux_losses = []
    
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
        
        # Auxiliary loss (from active branches only)
        aux_loss = 0.0
        for idx, aux in enumerate(aux_output):
             if idx in active_branches:
                 aux_loss += loss_fn(aux, label)
        
        total_loss = main_loss + aux_loss_weight * aux_loss

        # Backward pass
        total_loss.backward()
        
        # Gradient clipping for stability
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()

        # Metrics
        preds = torch.argmax(output, dim=1)
        correct_train += (preds == label).sum().item()
        train_loss += total_loss.item()
        total_samples += label.size(0)
        
        # Track for logging
        main_losses.append(main_loss.item())
        aux_losses.append(aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss)
        
        # Update progress bar
        current_acc = (preds == label).float().mean().item()
        pbar.set_postfix({'loss': total_loss.item(), 'acc': current_acc})
        
        # TensorBoard batch-level logging (every 10 batches)
        if writer is not None and batch_idx % 10 == 0:
            writer.add_scalar('Batch/train_total_loss', total_loss.item(), global_step)
            writer.add_scalar('Batch/train_main_loss', main_loss.item(), global_step)
            writer.add_scalar('Batch/train_aux_loss', 
                              aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss, 
                              global_step)
            writer.add_scalar('Batch/train_acc', current_acc, global_step)
            writer.add_scalar('Batch/grad_norm', grad_norm.item(), global_step)
        
        global_step += 1
        
        # Cleanup
        del x1, x2, x3, output, aux_output, label, total_loss

    avg_loss = train_loss / len(data_loader)
    avg_acc = correct_train / total_samples

    return avg_loss, avg_acc, global_step
