import torch
import torch.nn as nn
import torchvision.models as models
from utils import intersection_over_union

class YOLOloss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YOLOloss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = 5.0  # constant
        self.lambda_noobj = 0.5  # constant

    def forward(self, predictions, targets):
        batch_size = predictions.size(0)
        # Reshape
        predictions = predictions.view(batch_size, self.S, self.S, self.C + self.B * 5)
        # [..., class1, class2, ..., classC, x1, y1, w1, h1, c1, x2, y2, w2, h2, c2]

        # Splitting
        class_pred = predictions[:, :, :, :self.C]
        bbox_pred = predictions[:, :, :, self.C:]

        # Initialize Losses
        coordinate_loss = 0.0
        class_prob_loss = 0.0
        obj_confidence_loss = 0.0
        no_obj_loss = 0.0
        # total_loss = ( self.lambda_coord * coord_loss + obj_confidence_loss + self.lambda_noobj * no_obj_conf_loss + class_prob_loss )

        obj_mask = targets[:, :, :, self.C] == 1  # Shape: (batch_size, S, S)
        
        for batch_index in range(batch_size):
            for i in range(self.S):
                for j in range(self.S):
                    if obj_mask[batch_index, i, j]:
                        # Object exists in this cell

                        # Get both the bounding boxes; 1 and 2
                        # Get the target box
                        # Select the box with the higher IOU score
                        # Calculate Coordinate Loss, Object Confidence Loss, Class Prob Loss
                        # Sum of losses (completes loss function)

                        #[0....20,x1,y1,w1,h1,conf1,x2,y2,w2,h2,conf2]
                        target_cell = targets[batch_index, i, j]
                        pred_cell = predictions[batch_index, i, j] 
                        
                        # Extract bounding boxes
                        bbox1 = pred_cell[self.C : self.C + 5]  # [x1, y1, w1, h1, conf1]
                        bbox2 = pred_cell[self.C + 5 : self.C + 10]  # [x2, y2, w2, h2, conf2]
                        target_box = target_cell[self.C : self.C + 5]  # [x, y, w, h, 1]

                        # Calculate IOU for both boxes
                        with torch.no_grad():
                            iou_bbox1 = intersection_over_union(bbox1[:4], target_box[:4])
                            iou_bbox2 = intersection_over_union(bbox2[:4], target_box[:4])
                        
                        # Select responsible box based on higher IOU
                        if iou_bbox1 > iou_bbox2:
                            responsible_box = bbox1
                            responsible_idx = 0
                        else:
                            responsible_box = bbox2
                            responsible_idx = 1

                        # Coordinate loss (x, y)
                        coordinate_loss += self.lambda_coord * (
                            (responsible_box[0] - target_box[0])**2 + 
                            (responsible_box[1] - target_box[1])**2
                        )
                        
                        # Coordinate loss (w, h) - use absolute values and add small epsilon
                        eps = 1e-6
                        pred_w = torch.clamp(responsible_box[2], min=eps)
                        pred_h = torch.clamp(responsible_box[3], min=eps)
                        target_w = torch.clamp(target_box[2], min=eps)
                        target_h = torch.clamp(target_box[3], min=eps)
                        
                        coordinate_loss += self.lambda_coord * (
                            (torch.sqrt(pred_w) - torch.sqrt(target_w))**2 + 
                            (torch.sqrt(pred_h) - torch.sqrt(target_h))**2
                        )

                        # Object confidence loss (only for responsible box)
                        obj_confidence_loss += (responsible_box[4] - target_box[4])**2

                        # Class probability loss
                        pred_classes = class_pred[batch_index, i, j]
                        target_classes = target_cell[:self.C]
                        class_prob_loss += torch.sum((pred_classes - target_classes)**2)

                        # No object confidence loss for the non-responsible box
                        if responsible_idx == 0:
                            no_obj_loss += self.lambda_noobj * (bbox2[4] - 0)**2
                        else:
                            no_obj_loss += self.lambda_noobj * (bbox1[4] - 0)**2

                    else:
                        # No object in this cell - penalize both boxes
                        bbox1_conf = bbox_pred[batch_index, i, j, 4]
                        bbox2_conf = bbox_pred[batch_index, i, j, 9]
                        no_obj_loss += self.lambda_noobj * (bbox1_conf**2 + bbox2_conf**2)

        # Sum of losses and normalize by batch size
        total_loss = (coordinate_loss + obj_confidence_loss + no_obj_loss + class_prob_loss) / batch_size
        return total_loss


# Alternative vectorized implementation (more efficient)
class YOLOlossVectorized(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YOLOlossVectorized, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = 5.0
        self.lambda_noobj = 0.5

    def forward(self, predictions, targets):
        batch_size = predictions.size(0)
        predictions = predictions.view(batch_size, self.S, self.S, self.C + self.B * 5)
        
        # Split predictions
        class_pred = predictions[:, :, :, :self.C]
        bbox_pred = predictions[:, :, :, self.C:]
        
        # Object mask
        obj_mask = targets[:, :, :, self.C] == 1
        
        # Initialize losses
        total_loss = 0.0
        
        if obj_mask.sum() > 0:  # Only compute if there are objects
            # Get indices where objects exist
            obj_indices = torch.where(obj_mask)
            
            for idx in range(len(obj_indices[0])):
                b, i, j = obj_indices[0][idx], obj_indices[1][idx], obj_indices[2][idx]
                
                target_cell = targets[b, i, j]
                pred_cell = predictions[b, i, j]
                
                # Extract boxes
                bbox1 = pred_cell[self.C:self.C+5]
                bbox2 = pred_cell[self.C+5:self.C+10]
                target_box = target_cell[self.C:self.C+5]
                
                # Calculate IOUs
                with torch.no_grad():
                    iou1 = intersection_over_union(bbox1[:4], target_box[:4])
                    iou2 = intersection_over_union(bbox2[:4], target_box[:4])
                
                # Select responsible box
                if iou1 > iou2:
                    responsible_box = bbox1
                    non_responsible_box = bbox2
                else:
                    responsible_box = bbox2
                    non_responsible_box = bbox1
                
                # Coordinate loss
                coord_loss = self.lambda_coord * (
                    (responsible_box[0] - target_box[0])**2 + 
                    (responsible_box[1] - target_box[1])**2
                )
                
                # Width/Height loss with clamping
                eps = 1e-6
                pred_w = torch.clamp(responsible_box[2], min=eps)
                pred_h = torch.clamp(responsible_box[3], min=eps)
                target_w = torch.clamp(target_box[2], min=eps)
                target_h = torch.clamp(target_box[3], min=eps)
                
                coord_loss += self.lambda_coord * (
                    (torch.sqrt(pred_w) - torch.sqrt(target_w))**2 + 
                    (torch.sqrt(pred_h) - torch.sqrt(target_h))**2
                )
                
                # Object confidence loss
                obj_conf_loss = (responsible_box[4] - target_box[4])**2
                
                # Class loss
                class_loss = torch.sum((class_pred[b, i, j] - target_cell[:self.C])**2)
                
                # No object loss for non-responsible box
                no_obj_loss = self.lambda_noobj * non_responsible_box[4]**2
                
                total_loss += coord_loss + obj_conf_loss + class_loss + no_obj_loss
        
        # No object loss for cells without objects
        no_obj_mask = ~obj_mask
        if no_obj_mask.sum() > 0:
            no_obj_indices = torch.where(no_obj_mask)
            for idx in range(len(no_obj_indices[0])):
                b, i, j = no_obj_indices[0][idx], no_obj_indices[1][idx], no_obj_indices[2][idx]
                bbox1_conf = bbox_pred[b, i, j, 4]
                bbox2_conf = bbox_pred[b, i, j, 9]
                total_loss += self.lambda_noobj * (bbox1_conf**2 + bbox2_conf**2)
        
        return total_loss / batch_size


# Test the model
def test_yolo_loss(S=7, B=2, C=20):
    batch_size = 2

    # Dummy predictions
    predictions = torch.rand((batch_size, S, S, C + B * 5), requires_grad=True)

    # Dummy targets
    targets = torch.zeros((batch_size, S, S, C + B * 5))
    for b in range(batch_size):
        i, j = 3, 4
        targets[b, i, j, 0] = 1  # class 0
        targets[b, i, j, C + 0] = 0.5  # x
        targets[b, i, j, C + 1] = 0.5  # y
        targets[b, i, j, C + 2] = 1.0  # w
        targets[b, i, j, C + 3] = 1.0  # h
        targets[b, i, j, C + 4] = 1.0  # object exists

    loss_fn = YOLOloss(S=S, B=B, C=C)
    loss = loss_fn(predictions, targets)

    print(f"Loss: {loss.item():.4f}")
    loss.backward()
    print("Working")

if __name__ == "__main__":
    test_yolo_loss()

# Loss: 16.4769
# Working