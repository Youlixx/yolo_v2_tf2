import numpy as np


def _compute_iou(box_true: np.ndarray, boxes_predicted: np.ndarray) -> np.ndarray:
    """
    Computes the IOU between a ground truth box and all the predicted boxes for the given image and class.

    Parameters
    ----------
    box_true: np.ndarray
        The ground truth box.
    boxes_predicted: np.ndarray
        The predicted boxes.

    Returns
    -------
    iou_scores: np.ndarray
        The IOU scores between the ground truth and each predicted box.
    """

    box_true_wh = box_true[2:4] - box_true[0:2]  # (2)
    boxes_predicted_wh = boxes_predicted[..., 2:4] - boxes_predicted[..., :2]  # (NB x 2)

    intersection_x0_y0 = np.maximum(box_true[:2], boxes_predicted[..., :2])  # (NB x 2)
    intersection_x1_y1 = np.minimum(box_true[2:4], boxes_predicted[..., 2:4])  # (NB x 2)
    intersection_wh = np.maximum(intersection_x1_y1 - intersection_x0_y0, 0.)  # (NB x 2)

    overlaps = intersection_wh[..., 0] * intersection_wh[..., 1]  # (NB)

    box_true_area = box_true_wh[0] * box_true_wh[1]  # (1)
    boxes_predicted_area = boxes_predicted_wh[..., 0] * boxes_predicted_wh[..., 1]  # (NB)
    iou_scores = overlaps / (boxes_predicted_area + box_true_area - overlaps + 1e-6)  # (NB)

    return iou_scores


def _classify_boxes(
        boxes_true: np.ndarray, boxes_predicted: np.ndarray, iou_threshold: float
) -> (np.ndarray, int):
    """
    This function classifies each predicted box as true or false positive. Each ground truth box have to be matched to
    a single predicted box, and the boxes are matched using the IOU (i.e. if the IOU is greater than the threshold,
    the predicted box is considered as a true positive). However, if multiple predicted boxes are associated with a
    same ground truth box, only the one with the highest probability is considered as a true positive and the others
    are classified as false positives. Predicted boxes that do not have a matching ground truth box are classified as
    false negatives as well. The labels are returned as a 1D vector of booleans corresponding to each predicted boxes.
    The number of false negatives is can be deduced from the number of true positives.

    Parameters
    ----------
    boxes_true: np.ndarray
        The ground truth boxes.
    boxes_predicted: np.ndarray
        The predicted boxes.
    iou_threshold: float
        The IOU threshold used to determine whether a predicted box is a true positive or a false positive.

    Returns
    -------
    boxes_positives: np.ndarray
        The label vector.
    """

    classified = np.zeros((len(boxes_predicted)), dtype=bool)  # (NB)
    boxes_positives = np.zeros((len(boxes_predicted)), dtype=bool)  # (NB)

    # We check if each ground truth box has a matching predicted box
    for i in range(len(boxes_true)):
        box_true = boxes_true[i]

        # We compute the IOU between the ground truth box and each predicted box. We only keep the IOU scores of the
        # predicted boxes that have not been matched yet by setting to 0 the scores of the matched boxes.
        iou_scores = _compute_iou(box_true, boxes_predicted)  # (NB)
        iou_scores[classified] = 0

        # If at least one predicted box has a matching IOU score, then the ground truth box is a true positive, and
        # the ground truth box is considered as a false negative otherwise.
        if len(iou_scores) > 0 and np.max(iou_scores) >= iou_threshold:
            best_index = np.argmax(iou_scores)

            # We classify the best predicted box as a true positive and set all the remaining matching predicted boxes
            # as false positives.
            boxes_positives[best_index] = True
            classified[best_index] = True
            iou_scores[classified] = 0

            while np.max(iou_scores) >= iou_threshold:
                false_index = np.argmax(iou_scores)

                classified[false_index] = True
                iou_scores[false_index] = 0

    return boxes_positives


def _compute_ap(boxes_true_all: list, boxes_predicted_all: list, label: int, iou_threshold: float) -> float:
    """
    Computes the area under the precision-recall curve for a given class. To compute the exact precision-recall curve,
    we sample the precision and recall for each possible threshold. This is done by starting with a threshold of 1 and
    decreasing it to the highest predicted probability below the threshold at each iteration. By doing this way, each
    time the threshold is decreased, a single predicted box is added to the considered boxes, and it is either
    classified as a true positive or false positive, allowing to compute the precision and recall.
     - If the predicted box is a true positive, then both the precision and recall increase.
     - If the predicted box is a false positive, then the precision decreases, but the recall remains the same.
    This means that the precision-recall curve have a saw-like shape: either both the precision and the recall increase
    or the precision decreases while the recall remains the same. Points located on the peaks of the curve are
    objectively obtained at the best threshold values for the given class. The area under the curve is therefore
    computed only using these peak values.

    Parameters
    ----------
    boxes_true_all: list
        The ground truth boxes.
    boxes_predicted_all: list
        The predicted boxes.
    label: int
        The class label.
    iou_threshold: float
        The IOU threshold used to determine whether a predicted box is a true positive or a false positive.

    Returns
    -------
    area_under_curve: float
        The area under the precision-recall curve.
    """

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # These list track the labels of each predicted box and their predicted objectness.
    boxes_positives = []
    probabilities = []

    for index in range(len(boxes_true_all)):
        boxes_true = np.array(boxes_true_all[index])
        boxes_predicted = np.array(boxes_predicted_all[index])

        if len(boxes_predicted) == 0:
            continue

        # Only the boxes associated with the given class are considered
        boxes_true = boxes_true[boxes_true[..., 4] == label]
        boxes_predicted = boxes_predicted[boxes_predicted[..., 4] == label]

        probabilities.append(boxes_predicted[..., 5])

        # The predicted boxes are classified as true or false positives
        boxes_positives.append(_classify_boxes(boxes_true, boxes_predicted, iou_threshold))

        # As our first threshold is 1, we consider all the predicted boxes as false negatives.
        false_negatives += len(boxes_true)

    boxes_positives = np.concatenate(boxes_positives)
    probabilities = np.concatenate(probabilities)

    if len(boxes_positives) == 0:
        return 0

    # We initialize the precision-recall curve with zeros.
    precision_recall = np.zeros((len(boxes_positives), 2))

    # The boxes are sorted by decreasing probability. We iteratively consider more and more predicted boxes in the
    # calculation as the threshold decreases.
    indices = np.argsort(probabilities)[::-1]

    for k in range(len(indices)):
        index = indices[k]

        # If the box is a true positive, we increase the number of true positives and decrease the number of false
        # negatives.
        if boxes_positives[index]:
            true_positives += 1
            false_negatives -= 1

        # If the box is a false positive, we increase the number of false positives.
        else:
            false_positives += 1

        # We update the precision-recall curve.
        precision_recall[k, 0] = true_positives / (true_positives + false_positives)
        precision_recall[k, 1] = true_positives / (true_positives + false_negatives)

    precision_recall_reverse = precision_recall[::-1, 0]
    area_under_curve = 0
    last_peak = 0

    while np.max(precision_recall[..., 0]) > 0:
        # We find the last highest peak of the precision-recall curve.
        peak_index = len(indices) - np.argmax(precision_recall_reverse) - 1

        # We compute the area under the curve between the last peak and the current peak.
        area_under_curve += (precision_recall[peak_index, 1] - last_peak) * precision_recall[peak_index, 0]
        last_peak = precision_recall[peak_index, 1]

        # We remove the last peak from the precision-recall curve by setting its precision to 0.
        precision_recall[precision_recall[..., 1] <= precision_recall[peak_index, 1], 0] = 0

    return area_under_curve


def compute_map(boxes_true_all: list, boxes_predicted_all: list, classes: int, iou_threshold: float) -> float:
    """
    Computes the mean average precision (mAP) given the ground truth and the predicted boxes.

    Parameters
    ----------
    boxes_true_all: list
        The ground truth boxes.
    boxes_predicted_all: list
        The predicted boxes.
    classes: int
        The number of classes.
    iou_threshold: float
        The IOU threshold used to determine whether a predicted box is a true positive or a false positive.

    Returns
    -------
    mean_average_precision: float
        The mean average precision.
    """

    mean_average_precision = 0

    for label in range(classes):
        mean_average_precision += _compute_ap(boxes_true_all, boxes_predicted_all, label, iou_threshold)

    return mean_average_precision / classes
