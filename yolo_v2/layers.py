from keras.layers import Layer, Convolution2D

import tensorflow as tf
import numpy as np


class YoloDetectionHead(Layer):
    """
    The detection layer is a simple point-wise convolutional layer with the YOLOv2 activation function.
    It takes a 4D tensor as input and outputs a 5D tensor of shape (BS x HG x WG x B x (5+C)).
    """

    def __init__(self, priors: np.ndarray, classes: int, **kwargs):
        """
        Initializes the detection layer.

        Parameters
        ----------
        priors: np.ndarray
            The prior boxes used for the detection.
        classes: int
            The number of classes to detect.
        """

        super(YoloDetectionHead, self).__init__(**kwargs)

        self._priors = tf.constant(priors, dtype=tf.float32)

        self._prior_count = len(priors)
        self._classes = classes

        self._convolution_head = Convolution2D(len(priors) * (5 + classes), (1, 1), strides=1, padding="same")

        self._grid_h = 0
        self._grid_w = 0
        self._output_shape = None

    def build(self, input_shape: tuple) -> None:
        """
        Builds the layer.

        The width and height of the input tensor have to be known in order to calculate the output shape.

        Parameters
        ----------
        input_shape: tuple
            The shape of the input tensor.
        """

        super(YoloDetectionHead, self).build(input_shape)

        self._grid_h, self._grid_w = input_shape[1:3]

        if self._grid_h is None or self._grid_w is None:
            raise ValueError("The input shape must have a known height and width.")

        self._output_shape = (-1, self._grid_h, self._grid_w, self._prior_count, 5 + self._classes)

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        """
        When called, this layers applies the point-wise convolution and the YOLOv2 activation function.

        The format of the tensor before the activation function is the following
         - x[..., 0] = x: the x position of the center of the box relative to the associated cell.
         - x[..., 1] = y: the y position of the center of the box relative to the associated cell.
         - x[..., 2] = w: the width of the box relative to the associated prior.
         - x[..., 3] = h: the height of the box relative to the associated prior.
         - x[..., 4] = L: the "objectness" score of the box (pre-activation logit).
         - x[..., 5:] = Lc: the conditional probability of the box belonging to each class (pre-activation logit).

        The format of the tensor after the activation function is the following
         - y[..., 0] = x: the x position of the center of the box relative to the output grid.
         - y[..., 1] = y: the y position of the center of the box relative to the output grid.
         - y[..., 2] = w: the width of the box relative to the output grid.
         - y[..., 3] = h: the height of the box relative to the output grid.
         - y[..., 4] = P: the "objectness" score of the box (i.e. the probability that the box contains an object).
         - y[..., 5:] = C: the conditional probability vector of the box belonging to each class.

        Parameters
        ----------
        inputs: tf.Tensor
            The input tensor pre point-wise convolution.

        Returns
        -------
        outputs: tf.Tensor
            The output tensor of shape (BS x HG x WG x B x (5+C)).
        """

        # First, we apply the point-wise convolution.
        x = self._convolution_head(inputs)  # (BS x GH x GW x B*(5 +C))
        x = tf.reshape(x, shape=self._output_shape)  # (BS x GH x GW x B x (5+C))

        # The predicted coordinates are local to the cell of the prediction, meaning it is a value between 0 and 1,
        # positioning the center of the box within that cell. The goal of the activation function is to get the
        # coordinates of the box in relative to the grid (i.e. x in [0 WG[ and y in [0 HG[).
        # To do this, first we create a tensor in which each cell contains its own coordinates within the grid.
        cell_indices_w = tf.range(self._grid_w, dtype=tf.float32)  # (WG)
        cell_indices_h = tf.range(self._grid_h, dtype=tf.float32)  # (HG)
        cell_indices = tf.stack(tf.meshgrid(cell_indices_w, cell_indices_h), axis=-1)  # (HG x WG x 2)
        cell_indices = tf.reshape(cell_indices, shape=(1, self._grid_h, self._grid_w, 1, 2))  # (1 x HG x WG x 1 x 2)

        # Then, we add the coordinates of the cell to the coordinates of the box after applying the sigmoid function.
        predicted_xy = x[..., :2]  # (BS x HG x WG x B x 2)
        predicted_xy = tf.nn.sigmoid(predicted_xy)  # (BS x HG x WG x B x 2)
        predicted_xy = cell_indices + predicted_xy  # (BS x HG x WG x B x 2)

        # The predicted dimension of the box is relative to the associated prior. The activation function used here is
        # the exponential function, meaning that a predicted size of 0 pre-activation will give the size of the prior.
        predicted_wh = x[..., 2:4]  # (BS x HG x WG x B x 2)
        predicted_wh = tf.exp(predicted_wh)  # (BS x HG x WG x B x 2)
        predicted_wh = self._priors * predicted_wh  # (BS x HG x WG x B x 2)

        # The predicted objectness is merely obtained by applying the sigmoid function to the logit.
        # This probability indicates whether the box actually contains an object.
        # The loss function used by YOLOv2 will make this value quantify the quality of the box.
        predicted_objectness = x[..., 4]  # (BS x HG x WG x B)
        predicted_objectness = tf.nn.sigmoid(predicted_objectness)  # (BS x HG x WG x B)
        predicted_objectness = tf.expand_dims(predicted_objectness, axis=-1)  # (BS x HG x WG x B x 1)

        # Finally, the conditional probability vector is obtained by applying the softmax function.
        predicted_probabilities = x[..., 5:]  # (BS x HG x WG x B x C)
        predicted_probabilities = tf.nn.softmax(predicted_probabilities, axis=-1)  # (BS x HG x WG x B x C)

        # The output tensor is then assembled by concatenating the previously computed tensors.
        y = tf.concat([predicted_xy, predicted_wh, predicted_objectness, predicted_probabilities], axis=-1)

        return y  # (BS x HG x WG x B x (5+C))
