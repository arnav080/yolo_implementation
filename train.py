import torch
from model import YOLO
from dataset import dataset_p
from loss import YOLOloss  # Use your fixed loss
from utils import (
    intersection_over_union,
    non_max_suppression,
    pred_to_boxes,
    mean_average_precision,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint
)
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np

seed = 123
torch.manual_seed(seed)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Hyperparams
S = 7
B = 2
C = 20
batch_size = 16
epochs = 50
weight_decay = 0.0005
save_model = True
learning_rate = 1e-4  # Reduced learning rate

def plot_predictions(model, val_loader, device, S, B, C, num_images=3):
    model.eval()
    
    data_iter = iter(val_loader)
    images, targets = next(data_iter)
    
    with torch.no_grad():
        images = images.to(device)
        predictions = model(images)
        
        pred_boxes = pred_to_boxes(predictions, S=S, B=B, C=C)
        
        # Apply NMS to predictions
        for i in range(len(pred_boxes)):
            pred_boxes[i] = non_max_suppression(
                pred_boxes[i], 
                iou_threshold=0.5, 
                confidence_threshold=0.5
            )

    for i in range(min(num_images, len(images))):
        img = images[i].cpu()
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img * std + mean
        img = torch.clamp(img, 0, 1)
        
        boxes = pred_boxes[i]
        plot_image(img, boxes)
    
    model.train()

def check_for_nan_inf(tensor, name):
    """Check if tensor contains NaN or Inf values"""
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
        return True
    if torch.isinf(tensor).any():
        print(f"Inf detected in {name}")
        return True
    return False

# Initialize model
model = YOLO(S=S, B=B, C=C).to(device)
loss_func = YOLOloss(S=S, B=B, C=C)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Add gradient clipping
max_grad_norm = 1.0

# Loading dataset
train_dataset = dataset_p(S=S, B=B, C=C)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=0  # Set to 0 for debugging
)

# Validation dataset
val_dataset = dataset_p(S=S, B=B, C=C)
val_indices = list(range(100))
val_subset = torch.utils.data.Subset(val_dataset, val_indices)
val_loader = DataLoader(
    val_subset,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=False,
    drop_last=False,
    num_workers=0
)

print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_subset)}")
print(f"Batches per epoch: {len(train_loader)}")

# Training loop
for epoch in range(epochs):
    print(f"\n=== Epoch {epoch + 1}/{epochs} ===")
    model.train()
    epoch_loss = 0
    batch_losses = []

    # Create progress bar for the epoch
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

    for batch_idx, (x, y) in enumerate(train_bar):
        x, y = x.to(device), y.to(device)

        # Forward pass
        predictions = model(x)
        
        # Check for NaN/Inf in predictions
        if check_for_nan_inf(predictions, "predictions"):
            print(f"Stopping training due to NaN/Inf in predictions at epoch {epoch+1}, batch {batch_idx}")
            break
        
        # Compute loss
        loss = loss_func(predictions, y)
        
        # Check for NaN/Inf in loss
        if check_for_nan_inf(loss, "loss"):
            print(f"Stopping training due to NaN/Inf in loss at epoch {epoch+1}, batch {batch_idx}")
            break
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Check for NaN/Inf in gradients
        nan_grads = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                if check_for_nan_inf(param.grad, f"gradient of {name}"):
                    nan_grads = True
                    break
        
        if nan_grads:
            print(f"Stopping training due to NaN/Inf in gradients at epoch {epoch+1}, batch {batch_idx}")
            break
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        # Update parameters
        optimizer.step()

        # Track loss
        current_loss = loss.item()
        epoch_loss += current_loss
        batch_losses.append(current_loss)
        
        # Update progress bar
        train_bar.set_postfix({
            'Loss': f'{current_loss:.4f}',
            'Avg Loss': f'{np.mean(batch_losses[-10:]):.4f}'  # Moving average of last 10 batches
        })

        # Print detailed loss info periodically
        if batch_idx % 20 == 0:
            print(f"\nBatch {batch_idx}/{len(train_loader)} | Loss: {current_loss:.4f}")

    # Calculate average loss for the epoch
    avg_loss = epoch_loss / len(train_loader)
    print(f"\nEpoch {epoch+1} Average Loss: {avg_loss:.4f}")
    
    # Early stopping if loss becomes NaN
    if np.isnan(avg_loss) or np.isinf(avg_loss):
        print(f"Training stopped due to NaN/Inf loss at epoch {epoch+1}")
        break

    # Calculate mAP every 5 epochs
    if (epoch + 1) % 5 == 0:
        print("Calculating mAP...")
        model.eval()
        
        try:
            # Get predictions and ground truth boxes
            pred_boxes, true_boxes = get_bboxes(
                val_loader,
                model, 
                iou_threshold=0.5, 
                confidence_threshold=0.5, 
                S=S, 
                B=B, 
                C=C, 
                device=device
            )
            
            # Calculate mAP
            map_score = mean_average_precision(
                pred_boxes, 
                true_boxes, 
                iou_threshold=0.5, 
                num_classes=C
            )

            print(f"mAP@0.5: {map_score:.4f}")
            
        except Exception as e:
            print(f"Error calculating mAP: {e}")
        
        model.train()

    # Save checkpoint
    if save_model and (epoch + 1) % 10 == 0:
        try:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "loss": avg_loss
            }
            save_checkpoint(checkpoint, filename=f"checkpoint_epoch{epoch+1}.pth.tar")
            print(f"Checkpoint saved for epoch {epoch+1}")
        except Exception as e:
            print(f"Error saving checkpoint: {e}")

print("\nTraining Completed!")

# Final evaluation
print("Calculating final mAP...")
model.eval()
try:
    pred_boxes, true_boxes = get_bboxes(
        val_loader, 
        model, 
        iou_threshold=0.5, 
        confidence_threshold=0.5, 
        S=S, 
        B=B, 
        C=C, 
        device=device
    )

    final_map = mean_average_precision(
        pred_boxes, 
        true_boxes, 
        iou_threshold=0.5, 
        num_classes=C
    )

    print(f"Final mAP@0.5: {final_map:.4f}")

    print("Plotting final predictions...")
    plot_predictions(model, val_loader, device, S, B, C, num_images=5)
    
except Exception as e:
    print(f"Error in final evaluation: {e}")