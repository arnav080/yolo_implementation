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
        self.lambda_coord = 5 # constant
        self.lambda_noobj = 0.5 # constant

    def forward(self, predictions, targets):
        batch_size = predictions.size(0)
        # Reshape
        predictions = predictions.view(batch_size, self.S, self.S, self.C + self.B * 5)
        # [..., class1, class2, ..., classC, x1, y1, w1, h1, c1, x2, y2, w2, h2, c2]

        # Splitting
        class_pred = predictions[:, :, :, :self.C]
        bbox_pred = predictions[:, :, :, self.C:]

        # Initialize Losses
        coordinate_loss = 0
        coordinate_loss_xy = 0
        coordinate_loss_wh = 0
        class_prob_loss = 0
        obj_confidence_loss = 0
        no_obj_loss = 0
        total_loss = 0

        # total_loss = ( self.lambda_coord * coord_loss + obj_confidence_loss + self.lambda_noobj * no_obj_conf_loss + class_prob_loss ) 

        for batch_index in range(batch_size):
            for i in range(self.S):
                for j in range(self.S):
                    target_cell = targets[batch_index, i, j]

                    # Object exists
                    if target_cell[self.C] == 1: 
                        # Get both the bounding boxes; 1 and 2
                        # Get the target box
                        # Select the box with the higher IOU score
                        # Calculate Coordinate Loss, Object Confidence Loss, Class Prob Loss
                        # Sum of losses (completes loss function)
                        
                        pred_cell = predictions[batch_index, i, j] # [0....20,x1,y1,w1,h1,conf1,x2,y2,w2,h2,conf2]
                        bbox1 = pred_cell[self.C : self.C + 5] # [x1, y1, w1, h1, conf1]
                        bbox2 = pred_cell[self.C + 5 : self.C + 10] # [x2, y2, w2, h2, conf2]
                        target_box = target_cell[self.C : self.C + 5] # [x, y, w, h, 1]

                        responsible_box = 0

                        # Calculating IOU for both boxes and comparing to derive(get) the responsible_box
                        iou_bbox1 = intersection_over_union(bbox1[:4], target_box[:4])
                        iou_bbox2 = intersection_over_union(bbox2[:4], target_box[:4])

                        # The function for calculating iou, shifted it from here to utils.py
                        # for better code structure

                        if iou_bbox1 > iou_bbox2:
                            responsible_box = bbox1
                            
                        else:
                            responsible_box = bbox2

                        # Coordinate loss
                        coordinate_loss_xy = (responsible_box[0] - target_box[0])**2 + (responsible_box[1] - target_box[1])**2
                        coordinate_loss_wh = (torch.sqrt(responsible_box[2]) - torch.sqrt(target_box[2]))**2 + (torch.sqrt(responsible_box[3]) - torch.sqrt(target_box[3]))**2
                        coordinate_loss += self.lambda_coord * coordinate_loss_xy + self.lambda_coord * coordinate_loss_wh

                        # Object confidence loss
                        obj_confidence_loss += (responsible_box[4] - target_box[4])**2

                        # Class probability loss
                        pred_classes = class_pred[batch_index, i, j]
                        target_classes = target_cell[:self.C]
                        class_prob_loss += torch.sum((pred_classes - target_classes)**2)

                    # for no object confidence loss
                    else:
                        bbox1_conf = bbox_pred[batch_index, i, j, 4]
                        bbox2_conf = bbox_pred[batch_index, i, j, 9]
                        no_obj_loss += self.lambda_noobj * ((bbox1_conf - 0)**2 + (bbox2_conf - 0)**2)

        # Sum of losses
        total_loss = (coordinate_loss + obj_confidence_loss + no_obj_loss + class_prob_loss) / batch_size
        return total_loss

    
    
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
