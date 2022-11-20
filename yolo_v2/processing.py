import numpy as np


def _compute_priors_clusters(boxes: np.ndarray, cluster_count: int, max_iter: int = 10000) -> np.ndarray:
    """
    Computes the priors of the YOLOv2 model by clustering the dimensions of the bounding boxes using naive K-means with
    the IOU distance.

    The algorithm is initialized randomly, therefore the results may differ from run to run. You can set the seed of
    np.random to get reproducible results.

    Parameters
    ----------
    boxes: np.ndarray
        The dimensions of the bounding boxes.
    cluster_count: int
        The number of clusters / priors.
    max_iter: int
        The maximum number of iterations for the K-means algorithm.

    Returns
    -------
    priors: np.ndarray
        The priors of the YOLOv2 model.
    """

    boxes_wh = np.expand_dims(boxes, axis=1)  # (NB x 1 x 2)

    # First, we initialize the centroids randomly.
    centers = boxes[np.random.choice(len(boxes), size=cluster_count, replace=False)]  # (B x 2)

    for _ in range(max_iter):
        # As we use the IOU distance, we need to compute the intersection and union areas between the boxes and the
        # centers. The IOU distance is computed as if the two boxes are centered between each other.
        intersection_wh = np.minimum(boxes_wh, centers)  # (NB x B x 2)
        intersection_area = intersection_wh[..., 0] * intersection_wh[..., 1]  # (NB x B)

        boxes_area = boxes_wh[..., 0] * boxes_wh[..., 1]  # (NB x 1)
        center_area = centers[..., 0] * centers[..., 1]  # (B)
        union_area = boxes_area + center_area - intersection_area  # (NB x B)

        # The IOU distance is computed as d(x, y) = 1 - IOU(x, y)
        iou_distances = 1 - intersection_area / union_area  # (NB x B)

        # We update the centers by taking the mean dimension of each cluster
        closest_center = np.argmin(iou_distances, axis=-1)  # (NB)
        new_centers = np.array([np.mean(boxes[closest_center == k], axis=0) for k in range(cluster_count)])  # (B x 2)

        # As long as we update the centers, we continue the loop.
        if np.all(centers == new_centers):
            break

        centers = new_centers  # (B x 2)

    return centers


def compute_priors(all_boxes: list, cluster_count: int, scaling: int) -> np.ndarray:
    """
    This function computes the priors of the YOLOv2 model by clustering the dimensions of the bounding boxes using
    K-means with the IOU distance. The priors should be computed on the training set, and kept the same for the tests.
    Increasing the number of clusters increases the bias of the model, and therefore can lead to underfitting.

    Parameters
    ----------
    all_boxes: list
        A list of lists of bounding boxes. Each list of boxes represents a single image.
    cluster_count: int
        The number of clusters / priors.
    scaling: int
        The scaling factor used to transform the bounding boxes to the YOLOv2 model format.

    Returns
    -------
    priors: np.ndarray
        The priors of the YOLOv2 model.
    """

    dimensions = []

    for boxes in all_boxes:
        for box in boxes:
            dimensions.append([box[2] - box[0], box[3] - box[1]])

    dimensions = np.array(dimensions)

    priors = _compute_priors_clusters(dimensions, cluster_count) / scaling
    priors = priors.astype(np.float32)

    return priors


def _to_one_hot(labels: np.ndarray, classes: int) -> np.ndarray:
    """
    Converts a label vector to a one-hot matrix.

    Parameters
    ----------
    labels: np.ndarray
        The label vector.
    classes: int
        The number of classes.

    Returns
    -------
    one_hot: np.ndarray
        The one-hot matrix.
    """

    one_hot = np.zeros((len(labels), classes))  # (NB x C)
    one_hot[np.arange(len(labels)), labels] = 1  # (NB x C)

    return one_hot


def _to_yolo_format(boxes: np.ndarray, scaling: int) -> np.ndarray:
    """
    Converts a list of bounding boxes from the standard format to the YOLOv2 format.

    More precisely, in standard format, the boxes are formatted relative to the input image as follows
     - x[..., 0] = x0: the left-most coordinate of the bounding box
     - x[..., 1] = y0: the top-most coordinate of the bounding box
     - x[..., 2] = x1: the right-most coordinate of the bounding box
     - x[..., 3] = y1: the bottom-most coordinate of the bounding box

    And in the YOLOv2 format, the boxes are formatted relative to the output grid as follows
     - y[..., 0] = x: the x coordinate of the center of the bounding box
     - y[..., 1] = y: the y coordinate of the center of the bounding box
     - y[..., 2] = w: the width of the bounding box
     - y[..., 3] = h: the height of the bounding box

    Parameters
    ----------
    boxes: np.ndarray
        The bounding boxes in the standard format.
    scaling: int
        The scaling factor used to transform the bounding boxes to the YOLOv2 model format.

    Returns
    -------
    yolo_boxes: np.ndarray
        The bounding boxes in the YOLOv2 format.
    """

    # In the standard format, the coordinates of the bounding boxes are relative to the input image, meaning that
    # x is in [0, W[ and y is in [0, H[.
    boxes_x0_y0 = boxes[..., :2]  # (NB x 2)
    boxes_x1_y1 = boxes[..., 2:]  # (NB x 2)

    boxes_wh = boxes_x1_y1 - boxes_x0_y0  # (NB x 2)
    boxes_xy = boxes_x0_y0 + boxes_wh / 2  # (NB x 2)

    # However, the model outputs coordinates relative to the output grid, meaning that x is in [0, WG[ and y is
    # in [0, HG[, thus we have to divide the computed coordinates by the scaling factor.
    boxes_yolo = np.concatenate([boxes_xy, boxes_wh], axis=-1)  # (NB x 4)
    boxes_yolo = boxes_yolo / scaling

    return boxes_yolo


def _find_best_prior(boxes: np.ndarray, priors: np.ndarray) -> np.ndarray:
    """
    Finds the best corresponding prior for each bounding box. The best prior is the one with the smallest IOU distance
    with the bounding box.

    Parameters
    ----------
    boxes: np.ndarray
        The bounding boxes in the YOLOv2 format.
    priors: np.ndarray
        The priors of the YOLOv2 model.

    Returns
    -------
    best_priors: np.ndarray
        A vector of indices of the best prior for each bounding box.
    """

    boxes_wh = boxes[..., 2:]  # (NB x 2)
    boxes_wh = np.expand_dims(boxes_wh, axis=1)  # (NB x 1 x 2)

    # As we use the IOU distance, we need to compute the intersection and union areas between the boxes and the priors.
    # The IOU distance is computed as if the two boxes are centered between each other.
    intersection_wh = np.minimum(boxes_wh, priors)  # (NB x B x 2)
    intersection_area = intersection_wh[..., 0] * intersection_wh[..., 1]  # (NB x B)

    boxes_area = boxes_wh[..., 0] * boxes_wh[..., 1]  # (NB x 1)
    prior_area = priors[..., 0] * priors[..., 1]  # (B)
    union_area = boxes_area + prior_area - intersection_area  # (NB x B)

    iou_scores = intersection_area / union_area  # (NB x B)

    # The selected prior for each bounding box is the one with the highest IOU score.
    best_prior = np.argmax(iou_scores, axis=-1)  # (NB)

    return best_prior


def _to_tensor_format(
        boxes: np.ndarray, priors: np.ndarray, classes: int, scaling: int, width: int, height: int
) -> np.ndarray:
    """
    Converts a list of bounding boxes in the standard format into a ground truth output tensor.

    Parameters
    ----------
    boxes: np.ndarray
        The bounding boxes in the standard format.
    priors: np.ndarray
        The priors of the YOLOv2 model.
    classes: int
        The number of classes.
    scaling: int
        The scaling factor used to transform the bounding boxes to the YOLOv2 model format.
    width: int
        The width of the input image.
    height: int
        The height of the input image.

    Returns
    -------
    y_true: np.ndarray
        The ground truth tensor.
    """

    # We initialize the ground truth tensor with zeros. Note that this tensor is essentially a sparse tensor, where only
    # the cells corresponding to the bounding boxes are non-zero. This means that by default, the true "objectness" is
    # set to zero for every cell that doesn't contain a box and therefor, only the "no objectness" loss will be applied
    # to these cells. The other cells should have a true "objectness" of 1 and in this case, all the other losses will
    # be applied. This tensor can be seen as a 3D grid of size HG x WG x B, where each cell corresponds to a box.
    y_true = np.zeros((height // scaling, width // scaling, len(priors), 5 + classes))  # (HG x WG x B x (5+C))

    # If there are no bounding boxes, on the input image, we return an empty ground truth tensor.
    if len(boxes) == 0:
        return y_true

    # First, we convert the bounding boxes from the standard format to the YOLOv2 format and the class labels to
    # one-hot vectors.
    boxes_yolo = _to_yolo_format(boxes[..., :4], scaling)  # (NB x 4)
    labels_one_hot = _to_one_hot(boxes[..., 4], classes)  # (NB x C)

    # Then, we find the best corresponding prior for each bounding box. This variable acts as a z index for the grid.
    best_prior = _find_best_prior(boxes_yolo, priors)  # (NB)

    # The bounding boxes should be placed at their corresponding grid cell by rounding down the coordinates.
    grid_coordinates = np.floor(boxes_yolo[..., :2]).astype(np.uint32)  # (NB x 2)
    grid_x = grid_coordinates[..., 0]  # (NB)
    grid_y = grid_coordinates[..., 1]  # (NB)

    # We construct the expected bounding box output using the previously converted bounding box, the one-hot vector, and
    # we set the true "objectness" to 1 as the cell actually contains a box.
    expected_values = np.concatenate([boxes_yolo, np.ones((len(boxes), 1)), labels_one_hot], axis=-1)  # (NB x (5+C))

    y_true[grid_y, grid_x, best_prior] = expected_values  # (HG x WG x B x (5+C))

    return y_true


def boxes_to_tensor(
        all_boxes: list, priors: np.ndarray, classes: int, scaling: int, width: int, height: int
) -> np.ndarray:
    """
    Converts the dataset bounding boxes into ground truth output tensors. The input list contains a list of bounding
    boxes in the standard format for each input image.

    Parameters
    ----------
    all_boxes: list
        A list of bounding boxes in the standard format for each input image.
    priors: np.ndarray
        The priors of the YOLOv2 model.
    classes: int
        The number of classes.
    scaling: int
        The scaling factor used to transform the bounding boxes to the YOLOv2 model format.
    width: int
        The width of the input image.
    height: int
        The height of the input image.

    Returns
    -------
    y_true: np.ndarray
        The ground truth tensors.
    """

    y_train = []

    for boxes in all_boxes:
        boxes = np.array(boxes)  # (NB x 5)
        y_true = _to_tensor_format(boxes, priors, classes, scaling, width, height)  # (HG x WG x B x (5+C))

        y_train.append(y_true)

    y_train = np.stack(y_train)  # (NI x HG x WG x B x (5+C))

    return y_train


def _to_standard_format(output_tensor: np.ndarray, scaling: int, alpha: float) -> np.ndarray:
    """
    Extracts the predicted bounding boxes from the output tensor and turns them into standard format.

    The bounding boxes are returned with their predicted classes and probabilities as follows:
     - x[..., 0] = x0: the left-most coordinate of the bounding box
     - x[..., 1] = y0: the top-most coordinate of the bounding box
     - x[..., 2] = x1: the right-most coordinate of the bounding box
     - x[..., 3] = y1: the bottom-most coordinate of the bounding box
     - x[..., 4] = c: the predicted class of the bounding box
     - x[..., 5] = p: the predicted probability of the bounding box

    Parameters
    ----------
    output_tensor: np.ndarray
        The output tensor.
    scaling: int
        The scaling factor used to transform the bounding boxes to the YOLOv2 model format.
    alpha: float
        The objectness threshold (i.e. if the predicted objectness is lower than alpha, the bounding box is discarded).

    Returns
    -------
    boxes: np.ndarray
        The bounding boxes in the standard format.
    """

    # First, we split the predicted tensors to retrieve the coordinates and the conditional probabilities.
    predicted_xy = output_tensor[..., :2]  # (HG x WG x B x 2)
    predicted_wh = output_tensor[..., 2:4]  # (HG x WG x B x 2)
    predicted_objectness = output_tensor[..., 4]  # (HG x WG x B)
    predicted_probabilities = output_tensor[..., 5:]  # (HG x WG x B x C)

    # We can retrieve the predicted label with its probability by taking the index of the maximum probability.
    predicted_labels = np.argmax(predicted_probabilities, axis=-1)  # (HG x WG x B)
    predicted_labels_probability = np.max(predicted_probabilities, axis=-1)  # (HG x WG x B)

    # Then, we convert the coordinates to the standard format.
    predicted_x0_y0 = predicted_xy - predicted_wh / 2  # (HG x WG x B x 2)
    predicted_x1_y1 = predicted_xy + predicted_wh / 2  # (HG x WG x B x 2)

    # As the coordinates are expressed relative to the output grid, we have to scale them back to the input image.
    boxes = scaling * np.concatenate([predicted_x0_y0, predicted_x1_y1], axis=-1)  # (HG x WG x B x 4)
    boxes = boxes.astype(np.uint32)  # (HG x WG x B x 4)

    # We discard the bounding boxes that are below the objectness threshold.
    confidence = predicted_objectness * predicted_labels_probability  # (HG x WG x B)

    boxes = boxes[confidence > alpha]  # (NB x 4)
    labels = predicted_labels[confidence > alpha]  # (NB)
    probabilities = predicted_labels_probability[confidence > alpha]  # (NB)

    # Finally, we concatenate the bounding boxes and the corresponding labels and probabilities.
    boxes_standard = np.concatenate([boxes, np.stack([labels, probabilities], axis=-1)], axis=-1)  # (NB x 6)

    return boxes_standard


def _non_maximum_suppression(boxes: np.ndarray, beta: float) -> list:
    """
    Performs non-maximum suppression on the input bounding boxes using their predicted probability. When the model
    predicts multiple bounding boxes for a single object on the input image, the redundant bounding boxes should be
    discarded. The NMS algorithm only keeps the bounding box with the highest predicted probability out of the
    overlapping bounding boxes. Two boxes are considered overlapping if their intersection is greater than the threshold
    beta.

    Parameters
    ----------
    boxes: np.ndarray
        The bounding boxes in the standard format.
    beta: float
        The overlap threshold.

    Returns
    -------
    boxes: list
        The list of indices of the bounding boxes that should be kept.
    """

    pick = []

    if len(boxes) == 0:
        return pick

    predicted_x0_y0 = boxes[..., :2]  # (NB x 2)
    predicted_x1_y1 = boxes[..., 2:4]  # (NB x 2)
    probabilities = boxes[..., 5]  # (NB)

    predicted_wh = predicted_x1_y1 - predicted_x0_y0  # (NB x 2)
    predicted_area = predicted_wh[..., 0] * predicted_wh[..., 1] + 1e-6  # (NB)

    # The indices of the boxes are sorted by their predicted probability, so the best boxes are at the end of the array.
    indices = np.argsort(probabilities)  # (NB)

    while len(indices) > 0:
        # We keep the bounding box with the highest predicted probability.
        index = indices[-1]
        pick.append(index)

        indices = indices[:-1]

        # The following operation consists in removing the other bounding boxes that overlap with the selected bounding
        # box. Note that the considered boxes must be of the same class. Two boxes are considered overlapping if their
        # IOU is greater than the threshold beta.
        indices_class = indices[boxes[index, 4] == boxes[..., 4][indices]]  # (~NB)

        intersection_x0_y0 = np.maximum(predicted_x0_y0[index], predicted_x0_y0[indices_class])  # (~NB x 2)
        intersection_x1_y1 = np.minimum(predicted_x1_y1[index], predicted_x1_y1[indices_class])  # (~NB x 2)

        intersection_wh = intersection_x1_y1 - intersection_x0_y0  # (~NB x 2)
        intersection_wh = np.maximum(intersection_wh, 0.)  # (~NB x 2)

        intersection_areas = intersection_wh[..., 0] * intersection_wh[..., 1]  # (~NB)
        union_areas = predicted_area[index] + predicted_area[indices_class] - intersection_areas  # (~NB)

        iou_scores = intersection_areas / union_areas  # (~NB)

        # We discard the bounding boxes that have an IOU greater than the threshold beta.
        indices_to_remove = indices_class[iou_scores > beta]
        indices = np.setdiff1d(indices, indices_to_remove)

    return pick


def tensor_to_boxes(output_tensors: np.ndarray, scaling: int, alpha: float, beta: float) -> list:
    """
    Converts back a list of output tensors into a list of bounding boxes corresponding to the input images.

    Parameters
    ----------
    output_tensors: np.ndarray
        The output tensors.
    scaling: int
        The scaling factor used to transform the bounding boxes to the YOLOv2 model format.
    alpha: float
        The objectness threshold (i.e. if the predicted objectness is lower than alpha, the bounding box is discarded).
    beta: float
        The overlap threshold used for the non-maximum suppression.

    Returns
    -------
    boxes: list
        The list of bounding boxes corresponding to the input images.
    """

    all_boxes = []

    for output_tensor in output_tensors:
        boxes = _to_standard_format(output_tensor, scaling, alpha)
        pick = _non_maximum_suppression(boxes, beta)

        boxes_reduced = boxes[pick]

        all_boxes.append(boxes_reduced.tolist())

    return all_boxes
