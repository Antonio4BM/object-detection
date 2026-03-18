import torch
from torch import Tensor
from collections import Counter
from torch.utils.data import DataLoader


def intersection_over_union(
    boxes_preds: Tensor,
    boxes_targets: Tensor,
    box_format: str = "midpoint",
):
    """
    boxes_preds:   (..., 4)
    boxes_targets: (..., 4)

    Returns:
        IoU with shape (..., 1)
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2

        box2_x1 = boxes_targets[..., 0:1] - boxes_targets[..., 2:3] / 2
        box2_y1 = boxes_targets[..., 1:2] - boxes_targets[..., 3:4] / 2
        box2_x2 = boxes_targets[..., 0:1] + boxes_targets[..., 2:3] / 2
        box2_y2 = boxes_targets[..., 1:2] + boxes_targets[..., 3:4] / 2

    elif box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]

        box2_x1 = boxes_targets[..., 0:1]
        box2_y1 = boxes_targets[..., 1:2]
        box2_x2 = boxes_targets[..., 2:3]
        box2_y2 = boxes_targets[..., 3:4]

    else:
        raise ValueError(f"Unsupported box_format: {box_format}")

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    box1_area = (box1_x2 - box1_x1).abs() * (box1_y2 - box1_y1).abs()
    box2_area = (box2_x2 - box2_x1).abs() * (box2_y2 - box2_y1).abs()

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
        remaining_boxes = []
        for box in bboxes:
            if box[0] != chosen_box[0]:
                remaining_boxes.append(box)
            else:
                iou = intersection_over_union(
                    torch.tensor(chosen_box[2:], dtype=torch.float32),
                    torch.tensor(box[2:], dtype=torch.float32),
                    box_format=box_format,
                ).item()

                if iou < iou_threshold:
                    remaining_boxes.append(box)

        bboxes = remaining_boxes
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

        detections.sort(key=lambda x: x[2], reverse=True)
        true_positives = torch.zeros((len(detections)))
        false_positives = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)


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
                if total_boxes_per_image[detection[0]][best_gt_index] == 0:
                    true_positives[detection_index] = 1
                    total_boxes_per_image[detection[0]][best_gt_index] = 1
                else:
                    false_positives[detection_index] = 1

            else:
                false_positives[detection_index] = 1


        true_positives_cumulative_sum = torch.cumsum(true_positives, dim=0)
        false_positves_cumulative_sum = torch.cumsum(false_positives, dim=0)

        recalls = true_positives_cumulative_sum / (total_true_bboxes + epsilon)
        precisions = true_positives_cumulative_sum / (true_positives_cumulative_sum + false_positves_cumulative_sum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions)) # add 1 to the start
        recalls = torch.cat((torch.tensor([0]), recalls)) # add 0 to the start

        average_precisions.append(torch.trapz(precisions, recalls)) # calcualte are under curve

        return sum(average_precisions) / len(average_precisions)


def get_bboxes(
    loader: DataLoader,
    model: object,
    iou_threshold: float,
    threshold: float,
    pred_format:str ="cells",
    box_format:str ="midpoint",
    device:str ="cuda",
    S: int = 7,
    C: int = 7,
    B: int = 2,
):
    all_pred_boxes = []
    all_true_boxes = []

    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        
        true_bboxes = cellboxes_to_boxes(
            labels, S=S, C=C, B=B, is_predictions=False
        )
        pred_bboxes = cellboxes_to_boxes(
            predictions, S=S, C=C, B=B, is_predictions=True
        )

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                pred_bboxes[idx],
                iou_threshold=iou_threshold,
                prob_threshold=threshold,
                box_format=box_format,
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                if box[1] > 0:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes



def cellboxes_to_boxes(out, S=7, C=7, B=2, is_predictions=True):
    """
    Converts model outputs or labels into a list of boxes per image.

    Returns:
        list of length batch_size
        each element is a list of S*S boxes:
        [class_pred, confidence, x, y, w, h]
    """
    if is_predictions:
        converted = convert_cellboxes(out, S=S, C=C, B=B)
    else:
        converted = convert_label_cellboxes(out, S=S, C=C)

    converted = converted.reshape(out.shape[0], S * S, 6)
    converted[..., 0] = converted[..., 0].long()

    all_bboxes = []
    for ex_idx in range(out.shape[0]):
        bboxes = []
        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes


def convert_label_cellboxes(labels, S=7, C=7):
    """
    labels shape: (batch, S, S, C + 5)
    or flattened: (batch, S*S*(C + 5))

    Returns:
        (batch, S, S, 6) where last dim is:
        [true_class, objectness, x, y, w, h]
    """
    labels = labels.to("cpu")
    batch_size = labels.shape[0]
    labels = labels.reshape(batch_size, S, S, C + 5)

    true_class = labels[..., :C].argmax(-1).unsqueeze(-1)
    objectness = labels[..., C:C+1]
    boxes = labels[..., C+1:C+5]

    cell_x = torch.arange(S).repeat(batch_size, S, 1).unsqueeze(-1)
    cell_y = cell_x.permute(0, 2, 1, 3)

    x = (boxes[..., 0:1] + cell_x) / S
    y = (boxes[..., 1:2] + cell_y) / S
    w_h = boxes[..., 2:4] / S

    converted_bboxes = torch.cat((x, y, w_h), dim=-1)
    converted_labels = torch.cat(
        (true_class, objectness, converted_bboxes), dim=-1
    )

    return converted_labels


def convert_cellboxes(predictions, S=7, C=7, B=2):
    """
    predictions shape:
        (batch, S*S*(C + B*5)) or (batch, S, S, C + B*5)

    Returns:
        (batch, S, S, 6) where last dim is:
        [predicted_class, score, x, y, w, h]
    """

    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, S, S, C + B * 5)

    # Class scores
    class_scores = predictions[..., :C]                     # (N, S, S, C)
    predicted_class = class_scores.argmax(-1, keepdim=True) # (N, S, S, 1)

    # Gather score of predicted class
    predicted_class_score = class_scores.gather(
        -1, predicted_class
    )  # (N, S, S, 1)

    # Box/conf data
    bbox_data = predictions[..., C:].reshape(batch_size, S, S, B, 5)
    confidences = bbox_data[..., 0:1]   # (N, S, S, B, 1)
    boxes = bbox_data[..., 1:5]         # (N, S, S, B, 4)

    # Pick best box by confidence
    best_box_idx = confidences.argmax(dim=3, keepdim=True)  # (N, S, S, 1, 1)

    best_boxes = boxes.gather(
        3, best_box_idx.expand(-1, -1, -1, -1, 4)
    ).squeeze(3)  # (N, S, S, 4)

    best_confidence = confidences.gather(
        3, best_box_idx
    ).squeeze(3)  # (N, S, S, 1)

    # Final score = class score * objectness
    final_score = predicted_class_score * best_confidence   # (N, S, S, 1)

    # Convert from cell-relative to image-relative
    cell_x = torch.arange(S).repeat(batch_size, S, 1).unsqueeze(-1)
    cell_y = cell_x.permute(0, 2, 1, 3)

    x = (best_boxes[..., 0:1] + cell_x) / S
    y = (best_boxes[..., 1:2] + cell_y) / S
    w = best_boxes[..., 2:3] / S
    h = best_boxes[..., 3:4] / S

    converted_bboxes = torch.cat((x, y, w, h), dim=-1)

    converted_preds = torch.cat(
        (predicted_class.float(), final_score, converted_bboxes), dim=-1
    )

    return converted_preds
            
    