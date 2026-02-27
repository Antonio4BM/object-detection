import torch


def intersection_over_union(boxes_preds, boxes_targets, bbox_format="midpoint"):

    if bbox_format == "midpoint":
        box1_x1 = boxes_preds[:, 0:1] - boxes_preds[:, 2:3] / 2
        box1_y1 = boxes_preds[:, 1:2] - boxes_preds[:, 3:4] / 2
        box1_x2 = boxes_preds[:, 0:1] + boxes_preds[:, 2:3] / 2
        box1_y2 = boxes_preds[:, 1:2] + boxes_preds[:, 3:4] / 2
    
        box2_x1 = boxes_targets[:, 0:1] - boxes_targets[:, 2:3] / 2
        box2_y1 = boxes_targets[:, 1:2] - boxes_targets[:, 3:4] / 2
        box2_x2 = boxes_targets[:, 0:1] + boxes_targets[:, 2:3] / 2 
        box2_y2 = boxes_targets[:, 1:2] + boxes_targets[:, 3:4] / 2
        
    elif bbox_format == "corners":
        box1_x1 = boxes_preds[:, 0:1]
        box1_y1 = boxes_preds[:, 1:2]
        box1_x2 = boxes_preds[:, 2:3]
        box1_y2 = boxes_preds[:, 3:4]
    
        box2_x1 = boxes_targets[:, 0:1]
        box2_y1 = boxes_targets[:, 1:2]
        box2_x2 = boxes_targets[:, 2:3]
        box2_y2 = boxes_targets[:, 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2-x1).clamp(0) * (y2-y1).clamp(0)

    box1_area = (box1_x2 - box1_x1).clamp(0) * (box1_y2 - box1_y1).clamp(0)
    box2_area = (box2_x2 - box2_x1).clamp(0) * (box2_y2 - box2_y1).clamp(0)

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def non_max_suppression(
    bboxes,
    iou_threshold,
    prob_threshold,
    box_format="corners"
):
    bboxes = [box for box in bboxes if box[1] > prob_threshold] # filter bounding boxes greater than a threshold
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format
            ) < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms

    