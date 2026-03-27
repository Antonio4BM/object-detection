import torch
import torch.nn as nn
from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=7):
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum")

        self.S = S
        self.B = B
        self.C = C

        self.lambda_noobj = 0.5
        self.lambda_coord = 5.0

    def forward(self, predictions, targets):
        """
        predictions: (N, S*S*(C + B*5)) or (N, S, S, C + B*5)
        targets:     (N, S*S*(C + 5))   or (N, S, S, C + 5)

        Target format per cell:
            [class_onehot(C), obj, x, y, w, h]

        Prediction format per cell:
            [class_scores(C), conf1, x1, y1, w1, h1, conf2, x2, y2, w2, h2, ...]
        """

        N = predictions.shape[0]

        predictions = predictions.reshape(N, self.S, self.S, self.C + self.B * 5)
        targets = targets.reshape(N, self.S, self.S, self.C + 5)

        # -----------------------------
        # Extract target pieces
        # -----------------------------
        exists_box = targets[..., self.C:self.C+1]          # (N, S, S, 1)
        target_box = targets[..., self.C+1:self.C+5]        # (N, S, S, 4)
        target_classes = targets[..., :self.C]              # (N, S, S, C)

        # -----------------------------
        # Extract predicted boxes/confidences
        # -----------------------------
        pred_boxes = []
        pred_confs = []

        for b in range(self.B):
            start = self.C + b * 5
            pred_confs.append(predictions[..., start:start+1])      # conf
            pred_boxes.append(predictions[..., start+1:start+5])    # x,y,w,h

        # list -> tensors with box dimension
        # pred_boxes: (N, S, S, B, 4)
        # pred_confs: (N, S, S, B, 1)
        pred_boxes = torch.stack(pred_boxes, dim=3)
        pred_confs = torch.stack(pred_confs, dim=3)

        # -----------------------------
        # IoU between each predicted box and target box
        # -----------------------------
        ious = []
        for b in range(self.B):
            iou = intersection_over_union(
                pred_boxes[..., b, :],
                target_box,
                box_format="midpoint",
            )  # expected shape (N, S, S, 1) or (N, S, S)
            if iou.dim() == 3:
                iou = iou.unsqueeze(-1)
            ious.append(iou)

        # (N, S, S, B)
        ious = torch.cat(ious, dim=-1)

        iou_maxes, best_box = torch.max(ious, dim=-1, keepdim=True)   # best_box: (N, S, S, 1)

        # -----------------------------
        # Select the best predicted box per cell
        # -----------------------------
        best_box_expanded = best_box.unsqueeze(-1).expand(-1, -1, -1, -1, 4)
        best_pred_box = pred_boxes.gather(3, best_box_expanded).squeeze(3)   # (N, S, S, 4)

        best_conf_idx = best_box.unsqueeze(-1)  # (N, S, S, 1, 1)
        best_pred_conf = pred_confs.gather(3, best_conf_idx).squeeze(3)      # (N, S, S, 1)

        # -----------------------------
        # Box coordinate loss
        # -----------------------------
        box_predictions = exists_box * best_pred_box
        box_targets = exists_box * target_box

        box_predictions_xy = box_predictions[..., 0:2]
        box_predictions_wh = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4]) + 1e-6
        )

        box_targets_xy = box_targets[..., 0:2]
        box_targets_wh = torch.sqrt(box_targets[..., 2:4] + 1e-6)

        box_predictions = torch.cat([box_predictions_xy, box_predictions_wh], dim=-1)
        box_targets = torch.cat([box_targets_xy, box_targets_wh], dim=-1)

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        # -----------------------------
        # Objectness loss
        # -----------------------------
        object_loss = self.mse(
            torch.flatten(exists_box * best_pred_conf),
            torch.flatten(exists_box),
        )

        # -----------------------------
        # No-object loss
        # Penalize all box confidences in cells without objects
        # -----------------------------
        no_object_loss = 0
        for b in range(self.B):
            no_object_loss += self.mse(
                torch.flatten((1 - exists_box) * pred_confs[..., b, :], start_dim=1),
                torch.flatten((1 - exists_box) * torch.zeros_like(exists_box), start_dim=1),
            )

        # -----------------------------
        # Class loss
        # Only for cells containing objects
        # -----------------------------
        class_predictions = predictions[..., :self.C]
        class_loss = self.mse(
            torch.flatten(exists_box * class_predictions, end_dim=-2),
            torch.flatten(exists_box * target_classes, end_dim=-2),
        )

        loss = (
            self.lambda_coord * box_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )

        return loss  