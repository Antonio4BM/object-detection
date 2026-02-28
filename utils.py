import torch
from torch import Tensor
from collections import Counter


def intersection_over_union(
    boxes_preds: Tensor,
    boxes_targets: Tensor,
    bbox_format: str = "midpoint"
):

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
    bboxes: Tensor,
    iou_threshold: float,
    prob_threshold: float,
    box_format:str = "corners"
):
    bboxes = [box for box in bboxes if box[1] > prob_threshold] # filter bounding boxes greater than a threshold
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]  # check if it isn't the same class or if it is the IoU is lower than threshold, meaning it is another object
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format
            ) < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def mean_average_precision(
    pred_boxes: Tensor,
    true_boxes: Tensor,
    iou_threshold: float,
    box_format: str="corners",
    num_classes: int=20
):

    average_precisions = []
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        total_boxes_per_image = Counter([gtb[0] for gtb in ground_truths]) # track the boxes we have covered

        for key, value in total_boxes_per_image.items():
            total_boxes_per_image[key] = torch.zeros(value)

        detections.sort(key, lambda x: x[2], reverse=True)
        true_positives = torch.zeros((len(detections)))
        false_positives = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truth)


        for detection_index, detection in enumerate(detections):
            ground_truth_image = [
                bbox for bbox in ground_truths if bbox[0] == detection[0] # filter ground truth on the same image
            ]

            num_ground_truths = len(ground_truth_image)
            best_iou = 0.0

            for ground_truth_index, gt in enumerate(ground_truth_image):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_index = ground_truth_index


            if best_iou > iou_threshold:
                if total_boxes_per_image[detections[0]][best_gt_index] == 0:
                    true_positives[detection_index] = 1
                    total_boxes_per_image[detections[0]][best_gt_index] = 1
                else:
                    false_positives[detection_index] = 1

            else:
                false_positives[detection_index] = 1


        true_positives_cumulative_sum = torch.cumsum(true_positives, index=0)
        false_positves_cumulative_sum = torch.cumsum(false_positives,index=0)

        recalls = true_positives_cumulative_sum / (total_true_bboxes + epsilon)
        precisions = true_positives_cumulative_sum / (true_positives_cumulative_sum + false_positves_cumulative_sum + epsilon)
        precisions = tensor.cat((torch.tensor[1], precisions)) # add 1 to the start
        recalls = tensor.cat((torch.tensor[0], recalls)) # add 0 to the start

        average_precision.append(torch.trapz(precisions, recalls)) # calcualte are under curve

        return sum(average_precisions) / len(average_precisions)
            
    