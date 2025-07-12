import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def intersection_over_union(box1, box2):

        # converting coords
        box1_x1 = box1[0] - box1[2] / 2
        box1_y1 = box1[1] - box1[3] / 2
        box1_x2 = box1[0] + box1[2] / 2
        box1_y2 = box1[1] + box1[3] / 2

        box2_x1 = box2[0] - box2[2] / 2
        box2_y1 = box2[1] - box2[3] / 2
        box2_x2 = box2[0] + box2[2] / 2
        box2_y2 = box2[1] + box2[3] / 2

        # intersection
        inter_x1 = torch.max(box1_x1, box2_x1)
        inter_y1 = torch.max(box1_y1, box2_y1)
        inter_x2 = torch.min(box1_x2, box2_x2)
        inter_y2 = torch.min(box1_y2, box2_y2)
        inter_w = torch.clamp(inter_x2 - inter_x1, 0)
        inter_h = torch.clamp(inter_y2 - inter_y1, 0)
        inter_area = inter_w * inter_h

        # Union area
        box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
        box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
        union_area = box1_area + box2_area - inter_area

        iou = inter_area / (union_area + 1e-6)
        return iou



def pred_to_boxes(predictions, S=7, C=20, B=2, img_size=448):
    batch_size = predictions.shape[0]
    predictions = predictions.view(batch_size, S, S, C + B * 5)
    
    output_boxes = []

    for batch in range(batch_size):
        boxes = []
        for i in range(S):
            for j in range(S):
                cell = predictions[batch, i, j]
                class_prob = cell[:C]
                bbox1 = cell[C:C+5]
                bbox2 = cell[C+5:C+10]

                if bbox1[4] > bbox2[4]:
                    main_bbox = bbox1
                else:
                    main_bbox = bbox2

                confidence = main_bbox[4]
                x, y, w, h = main_bbox[:4]

                x_full = (j + x) / S
                y_full = (i + y) / S

                x_img = x_full * img_size
                y_img = y_full * img_size
                box_w_img = w * img_size
                box_h_img = h * img_size

                x1 = x_img - box_w_img / 2
                y1 = y_img - box_h_img / 2
                x2 = x_img + box_w_img / 2
                y2 = y_img + box_h_img / 2

                class_id = torch.argmax(class_prob).item()
                boxes.append([x1.item(), y1.item(), x2.item(), y2.item(), class_id, confidence.item(), ])
        output_boxes.append(boxes)
    
    return output_boxes



def plot_image(img, boxes):

    fig, ax = plt.subplots(1)
    ax.imshow(img.permute(1, 2, 0))

    for box in boxes:
        class_id, conf, x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f"Class {class_id}: {conf:.2f}", color="white",
                fontsize=8, bbox=dict(facecolor="red", edgecolor="red", alpha=0.7))

    plt.axis("off")
    plt.show()


def non_max_suppression(boxes, confidence_threshold=0.5, iou_threshold=0.5):
    
    boxes_sorted = sorted(boxes, key=lambda x: x[5], reverse=True) # boxes sorted in desc order of confidence
    iou_box_list = [] # list of boxes after of higher confidence values than threshold
    new_box_list = [] # final list

    # removing boxes with low confidence than the threshold value
    for box in boxes_sorted:
        if box[5] > confidence_threshold:
            iou_box_list.append(box)
        else:
            pass
    
    # loop over till the list is emptied
    while len(iou_box_list) > 0:
        current_box = iou_box_list.pop(0)
        new_box_list.append(current_box) # append the highset conf. box to our new list

        # incase boxes are from the same class, compare iou's
        for box in iou_box_list:
            if current_box[4] == box[4]:
                iou = intersection_over_union(current_box[:4], box[:4])
                if iou > iou_threshold:
                    iou_box_list.remove(box)
    return new_box_list # list after non max sup



def get_bboxes(loader, model, iou_threshold=0.5, confidence_threshold=0.5, S=7, B=2, C=20, device="cpu"):
    all_pred_boxes = []
    all_true_boxes = []

    train_idx = 0

    with torch.no_grad():
        for batch_idx, (x, labels) in enumerate(loader):
            x = x.to(device)
            labels = labels.to(device)
            predictions = model(x)

            batch_size = x.shape[0]
            pred_boxes = pred_to_boxes(predictions, S, C, B)
            true_boxes = pred_to_boxes(labels, S, C, B)

            for idx in range(batch_size):
                nms_boxes = non_max_suppression(pred_boxes[idx], confidence_threshold, iou_threshold)
                for box in nms_boxes:
                    all_pred_boxes.append([train_idx] + box)
                for box in true_boxes[idx]:
                    if box[5] > 0:
                        all_true_boxes.append([train_idx] + box)
                train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes

def mean_average_precision(pred_boxes, ground_truth_boxes, iou_threshold=0.5, num_classes=20):
    average_precisions = []

    for c in range(num_classes):
        detections = [box for box in pred_boxes if box[5] == c]
        ground_truths = [box for box in ground_truth_boxes if box[5] == c]

        amount_bboxes = {}
        for gt in ground_truths:
            img_idx = gt[0]
            if img_idx in amount_bboxes:
                amount_bboxes[img_idx] += 1
            else:
                amount_bboxes[img_idx] = 1

        for key in amount_bboxes:
            amount_bboxes[key] = torch.zeros(amount_bboxes[key])

        detections.sort(key=lambda x: x[6], reverse=True)
        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))

        total_true_bboxes = len(ground_truths)

        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            img_idx = detection[0]
            best_iou = 0
            best_gt_idx = -1
            gt_for_img = [gt for gt in ground_truths if gt[0] == img_idx]

            for gt_idx, gt in enumerate(gt_for_img):
                iou = intersection_over_union(torch.tensor(detection[1:5]), torch.tensor(gt[1:5]))
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou > iou_threshold:
                if amount_bboxes[img_idx][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[img_idx][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + 1e-6)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + 1e-6)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))

        average_precision = torch.trapz(precisions, recalls)
        average_precisions.append(average_precision)

    return sum(average_precisions) / len(average_precisions)


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    print("saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])