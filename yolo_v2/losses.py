from keras.losses import Loss

import tensorflow as tf


class YoloDetectionLoss(Loss):
    """
    The YOLOv2 detection loss.
    """

    def __init__(self, **kwargs):
        """
        Initializes the detection loss.
        """

        super(YoloDetectionLoss, self).__init__(**kwargs)

    def call(self, y_true: tf.Tensor, y_predicted: tf.Tensor) -> tf.Tensor:
        """
        Computes the YOLOv2 loss function between the ground truth and the predicted tensors.

        Both tensors are expected to have the shape (BS x HG x WG x B x (5+C)) and formatted as following:
         - y[..., 0] = x: the x position of the center of the box relative to the output grid.
         - y[..., 1] = y: the y position of the center of the box relative to the output grid.
         - y[..., 2] = w: the width of the box relative to the output grid.
         - y[..., 3] = h: the height of the box relative to the output grid.
         - y[..., 4] = P: the "objectness" score of the box (i.e. the probability that the box contains an object).
         - y[..., 5:] = C: the conditional probability vector of the box belonging to each class.

        Parameters
        ----------
        y_true: tf.Tensor
            The ground truth tensor.
        y_predicted: tf.Tensor
            The predicted tensor.

        Returns
        -------
        loss: tf.Tensor
            The loss tensor.
        """

        # First, we split the predicted tensors to retrieve the coordinates and the conditional probabilities.
        predicted_xy = y_predicted[..., :2]  # (BS x HG x WG x B x 2)
        predicted_wh = y_predicted[..., 2:4]  # (BS x HG x WG x B x 2)
        predicted_objectness = y_predicted[..., 4]  # (BS x HG x WG x B)
        predicted_probabilities = y_predicted[..., 5:]  # (BS x HG x WG x B x C)

        # Then, we do the same to the ground truth tensors.
        true_xy = y_true[..., :2]  # (BS x HG x WG x B x 2)
        true_wh = y_true[..., 2:4]  # (BS x HG x WG x B x 2)
        true_objectness = y_true[..., 4]  # (BS x HG x WG x B)
        true_probabilities = y_true[..., 5:]  # (BS x HG x WG x B x C)

        # In the following operations, true_objectness is a tensor containing ones for the cells that contain an object.
        # Position error: a mere square error between the predicted and ground truth positions.
        diff_xy = tf.square(predicted_xy - true_xy)  # (BS x HG x WG x B x 2)
        diff_xy = tf.reduce_sum(diff_xy, axis=-1)  # (BS x HG x WG x B)
        diff_xy = diff_xy * true_objectness  # (BS x HG x WG x B)

        # Dimension error: a square error between the square roots of the predicted and ground truth dimensions.
        diff_wh = tf.square(tf.sqrt(predicted_wh) - tf.sqrt(true_wh))  # (BS x HG x WG x B x 2)
        diff_wh = tf.reduce_sum(diff_wh, axis=-1)  # (BS x HG x WG x B)
        diff_wh = diff_wh * true_objectness  # (BS x HG x WG x B)

        # The following operations consists in determining the IOU between the predicted and ground truth boxes.
        # First, we compute the position of top-left and bottom-right corners of the predicted boxes.
        predicted_x0_y0 = predicted_xy - predicted_wh / 2  # (BS x HG x WG x B x 2)
        predicted_x1_y1 = predicted_xy + predicted_wh / 2  # (BS x HG x WG x B x 2)

        # Same goes for the ground truth boxes.
        true_x0_y0 = true_xy - true_wh / 2  # (BS x HG x WG x B x 2)
        true_x1_y1 = true_xy + true_wh / 2  # (BS x HG x WG x B x 2)

        # Then we compute the coordinates of the intersection between the predicted and ground truth boxes.
        intersection_x0_y0 = tf.maximum(predicted_x0_y0, true_x0_y0)  # (BS x HG x WG x B x 2)
        intersection_x1_y1 = tf.minimum(predicted_x1_y1, true_x1_y1)  # (BS x HG x WG x B x 2)

        # Using the coordinates, we can deduce the dimensions of the intersection.
        # If the intersection is empty, at least one of the dimension will be negative. By setting it to zero, the
        # intersection area will be zero.
        intersection_wh = intersection_x1_y1 - intersection_x0_y0  # (BS x HG x WG x B x 2)
        intersection_wh = tf.maximum(intersection_wh, 0.)  # (BS x HG x WG x B x 2)

        # Then, we compute the intersection area between the predicted and ground truth boxes.
        intersection_area = intersection_wh[..., 0] * intersection_wh[..., 1]  # (BS x HG x WG x B)

        # To compute the IOU we also need to compute the union area.
        predicted_area = predicted_wh[..., 0] * predicted_wh[..., 1]  # (BS x HG x WG x B)
        true_area = true_wh[..., 0] * true_wh[..., 1]  # (BS x HG x WG x B)
        union_area = predicted_area + true_area - intersection_area  # (BS x HG x WG x B)

        # Finally, we compute the IOU between the predicted and ground truth boxes.
        iou_scores = intersection_area / union_area  # (BS x HG x WG x B)

        # Objectness error: a square error between the predicted objectness and the IOU between the boxes.
        # This means that the objectness will tend to quantify the quality of the predicted box.
        diff_objectness = tf.square(predicted_objectness - iou_scores)  # (BS x HG x WG x B)
        diff_objectness = diff_objectness * true_objectness  # (BS x HG x WG x B)

        # No objectness error: if the predicted box does not contain an object, the objectness should tend to zero.
        diff_no_object = tf.square(predicted_objectness)  # (BS x HG x WG x B)
        diff_no_object = diff_no_object * (1 - true_objectness)  # (BS x HG x WG x B)

        # Classification error: a square error between the predicted and ground truth conditional probabilities.
        # Note that any kind of classification loss can be used such as the binary cross-entropy.
        diff_classification = tf.square(predicted_probabilities - true_probabilities)  # (BS x HG x WG x B x C)
        diff_classification = tf.reduce_sum(diff_classification, axis=-1)  # (BS x HG x WG x B)
        diff_classification = diff_classification * true_objectness  # (BS x HG x WG x B)

        # The total loss is the weighted sum of all the previously computed errors.
        diff = 5 * diff_xy + 5 * diff_wh + diff_objectness + diff_no_object + diff_classification  # (BS x HG x WG x B)
        diff = tf.reduce_sum(diff, axis=(1, 2, 3))  # (BS)

        return diff  # (BS)
