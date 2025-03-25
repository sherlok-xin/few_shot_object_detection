import numpy as np
import torch
from torchvision.ops import nms
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score, confusion_matrix

def evaluate_detection(model, dataloader, device='cuda'):
    model.eval()
    all_preds = []
    all_targets = []

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            outputs = model(images)

        # 假设outputs包含预测的边界框和分类结果
        preds = outputs['boxes']
        scores = outputs['scores']
        labels = outputs['labels']

        # 非极大值抑制 (NMS)
        keep = nms(preds, scores, 0.5)
        preds = preds[keep]
        scores = scores[keep]
        labels = labels[keep]

        all_preds.append((preds, scores, labels))
        all_targets.append(targets)

    # 计算评估指标
    precision, recall, _ = precision_recall_curve(all_targets, all_preds)
    ap = average_precision_score(all_targets, all_preds)
    fpr, tpr, _ = roc_curve(all_targets, all_preds)
    roc_auc = auc(fpr, tpr)
    conf_matrix = confusion_matrix(all_targets, all_preds)
    mAP = np.mean([average_precision_score(t, p) for t, p in zip(all_targets, all_preds)])
    accuracy = np.mean(np.array(all_preds) == np.array(all_targets))

    iou_scores = []
    for preds, targets in zip(all_preds, all_targets):
        iou = calculate_iou(preds, targets)
        iou_scores.append(iou)

    return {
        'precision': precision,
        'recall': recall,
        'ap': ap,
        'roc_auc': roc_auc,
        'confusion_matrix': conf_matrix,
        'mAP': mAP,
        'accuracy': accuracy,
        'iou': np.mean(iou_scores)
    }

def calculate_iou(preds, targets):
    # 计算IoU
    ious = []
    for pred, target in zip(preds, targets):
        xA = max(pred[0], target[0])
        yA = max(pred[1], target[1])
        xB = min(pred[2], target[2])
        yB = min(pred[3], target[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (pred[2] - pred[0] + 1) * (pred[3] - pred[1] + 1)
        boxBArea = (target[2] - target[0] + 1) * (target[3] - target[1] + 1)

        iou = interArea / float(boxAArea + boxBArea - interArea)
        ious.append(iou)
    return np.mean(ious)
