import torch

from torchvision.ops import batched_nms

SCORE_THRESH = 0.1
import numpy as np

# Use GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def convert_to_x1y1x2y2(bbox, img_width, img_height):
    """
    Convert bounding boxes from [cx, cy, w, h] to [x1, y1, x2, y2] format.

    Args:
        bbox (np.array): NumPy array of bounding boxes in the format [cx, cy, w, h].
        img_width (int): Width of the image.
        img_height (int): Height of the image.

    Returns:
        np.array: NumPy array of bounding boxes in the format [x1, y1, x2, y2].
    """
    cx, cy, w, h = bbox
    x1 = (cx - w / 2) * img_width
    y1 = (cy - h / 2) * img_height
    x2 = (cx + w / 2) * img_width
    y2 = (cy + h / 2) * img_height
    return np.array([x1, y1, x2, y2])

def apply_NMS(boxes, scores, labels, total_scores, iou=0.5):
    indexes_to_keep = batched_nms(torch.stack([torch.FloatTensor(box) for box in boxes], dim=0),
                       torch.FloatTensor(scores),
                       torch.IntTensor(labels),
                       iou)
    
    filtered_boxes = []
    filtered_scores = []
    filtered_labels = []
    filtered_total_scores = []
    deleted_boxes = []
    deleted_scores = []
    deleted_labels = []
    deleted_total_scores = []
    
    for x in range(len(boxes)):
        if x in indexes_to_keep:
            filtered_boxes.append(boxes[x])
            filtered_scores.append(scores[x])
            filtered_labels.append(labels[x])
            filtered_total_scores.append(total_scores[x])
        else:
            deleted_boxes.append(boxes[x])
            deleted_scores.append(scores[x])
            deleted_labels.append(labels[x])
            deleted_total_scores.append(total_scores[x])
    
    return filtered_boxes, filtered_scores, filtered_labels, filtered_total_scores

def evaluate_image(model, processor, im, vocabulary, MAX_PREDICTIONS=100, nms=False):
    global skipped_categories
    # preparing the inputs
    inputs = processor(text=vocabulary, images=im, return_tensors="pt", padding=True).to(device)
    
    # if the tokens length is above 16, the model can't handle them
    if  inputs['input_ids'].shape[1] > 16:
        skipped_categories += 1
        return None
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        
        
    # Get prediction logits
    logits = torch.max(outputs['logits'][0], dim=-1)
    scores = torch.sigmoid(logits.values).cpu().detach().numpy()
    all_scores = torch.sigmoid(outputs['logits'][0]).cpu().detach().numpy()
    
    # Get prediction labels and boundary boxes
    labels = logits.indices.cpu().detach().numpy()
    boxes = outputs['pred_boxes'][0].cpu().detach().numpy()    
        
    scores_filtered = []
    labels_filtered = []
    boxes_filtered = []
    total_scores_filtered = []
    height = im.shape[0]
    width = im.shape[1]
    
    boxes = [convert_to_x1y1x2y2(box, width, height) for box in boxes]
    # Combine the lists into tuples using zip
    if nms:
        # apply NMS
        boxes, scores, labels, all_scores = apply_NMS(boxes, scores, labels, all_scores)
    data = list(zip(scores, boxes, labels, all_scores))

    # Sort the combined data based on the first element of each tuple (score) in decreasing order
    sorted_data = sorted(data, key=lambda x: x[0], reverse=True)
    
    
    # filtering the predictions with low confidence
    for score, box, label, total_scores in sorted_data[:MAX_PREDICTIONS]:
        scores_filtered.append(score)
        labels_filtered.append(label)
        # boxes_filtered.append(convert_to_x1y1x2y2(box, width, height))
        boxes_filtered.append(box)
        total_scores_filtered.append(total_scores)
    
    return {
        'scores': scores_filtered,
        'labels': labels_filtered,
        'boxes': boxes_filtered,
        'total_scores': total_scores_filtered
    }

