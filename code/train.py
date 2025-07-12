import torch
from model import YOLO
from dataset import dataset_p
from loss import YOLOloss
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


seed = 123
torch.manual_seed(seed)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

#hyperparams
S = 7
B = 2
C = 20
batch_size = 16 # 64
epochs = 50 # 135
# momentum = 0.9
weight_decay = 0.0005
save_model = True
learning_rate = 1e-3

"""
def learning_rate(epoch):
    if epoch < 1:
        return 1e-3
    elif epoch < 75: # 1-74
        return 1e-2
    elif epoch < 105: # 75-104
        return 1e-3
    else: # 105-135
        return 1e-4
"""

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
        
        # get prediction boxes for this image
        boxes = pred_boxes[i]
        plot_image(img, boxes)
    
    model.train()
     
#
model = YOLO(S=S, B=B, C=C).to(device)
loss_func = YOLOloss(S=S, B=B, C=C)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# loading dataset
train_dataset = dataset_p(S=S, B=B, C=C)
train_loader = DataLoader(
    train_dataset,
    batch_size = batch_size,
    shuffle = True,
    drop_last = True
)

# take first 100 samples for validation; mAP calculation
val_dataset = dataset_p(S=S, B=B, C=C)
val_indices = list(range(100))
val_subset = torch.utils.data.Subset(val_dataset, val_indices)
val_loader = DataLoader(
    val_subset,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=False,
    drop_last=False
)

# train loop
for epoch in range(epochs):
    print(f"epoch {epoch + 1}/{epochs}")
    model.train()
    epoch_loss = 0

    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)

        predictions = model(x)
        loss = loss_func(predictions, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}\n")

    if (epoch + 1) % 5 == 0:
        print("Calculating mAP...")
        model.eval()
        
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
        model.train()

    if save_model:
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, filename=f"checkpoint_epoch{epoch+1}.pth.tar")

print("Traning Completed!")

print("Calculating final mAP...")
model.eval()
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