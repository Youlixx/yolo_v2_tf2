from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, LeakyReLU, BatchNormalization

from yolo_v2 import YoloDetectionHead

import numpy as np


def build_model(width: int, height: int, priors: np.ndarray, classes: int) -> Model:
    """
    Builds the YOLOv2 model. The YOLOv2 model can be split in two different stages : a feature extractor and a detection
    head. The feature extractor consists of several convolutional blocks used to compute a feature map representing the
    input image. Any kind of feature extractor, such as a Resnet can be used. Usually, a pretrained feature extractor is
    used to speed up the learning phase. The detection head is a simple 1x1 convolution layer used to predict the
    bounding boxes of the objects present on the image.

    Considering that the dataset used here is very simple, transfer-learning is not required and a custom feature
    extractor is used instead.

    The architecture of the feature extractor is the following:
     - Input (BS x H x W x 1)
     - Convolution2D + BatchNorm + LeakyReLU + MaxPooling2D (BS x H/2 x W/2 x 16)
     - Convolution2D + BatchNorm + LeakyReLU + MaxPooling2D (BS x H/4 x W/4 x 32)
     - Convolution2D + BatchNorm + LeakyReLU + MaxPooling2D (BS x H/8 x W/8 x 64)
     - Convolution2D + BatchNorm + LeakyReLU + MaxPooling2D (BS x HG x WG x 128)
     - Convolution2D + BatchNorm + LeakyReLU (BS x HG x WG x 256)

    Then the feature map goes through the detection head:
     - Convolution2D (BS x HG x WG x B*(5+C))
     - Activation function (BS x HG x WG x B x 5+C)

    Parameters
    ----------
    width: int
        The width of the input image.
    height: int
        The height of the input image.
    priors: np.ndarray
        The prior boxes used for the detection.
    classes: int
        The number of classes to detect.

    Returns
    -------
    model: Model
        The YOLOv2 model.
    """

    network_input = Input(shape=(height, width, 1))

    # Feature extractor
    # Convolutional block 1  (BS x H x W x 1) -> (BS x H/2 x W/2 x 16)
    network = Convolution2D(16, (3, 3), strides=1, padding="same", use_bias=False)(network_input)  # (BS x H x W x 16)
    network = BatchNormalization()(network)  # (BS x H x W x 16)
    network = LeakyReLU(alpha=0.1)(network)  # (BS x H x W x 16)
    network = MaxPooling2D(2, padding="valid")(network)  # (BS x H/2 x W/2 x 16)

    # Convolutional block 2  (BS x H/2 x W/2 x 16) -> (BS x H/4 x W/4 x 32)
    network = Convolution2D(32, (3, 3), strides=1, padding="same", use_bias=False)(network)  # (BS x H/2 x W/2 x 32)
    network = BatchNormalization()(network)  # (BS x H/2 x W/2 x 32)
    network = LeakyReLU(alpha=0.1)(network)  # (BS x H/2 x W/2 x 32)
    network = MaxPooling2D(2, padding="valid")(network)  # (BS x H/4 x W/4 x 32)

    # Convolutional block 3  (BS x H/4 x W/4 x 32) -> (BS x H/8 x W/8 x 64)
    network = Convolution2D(64, (3, 3), strides=1, padding="same", use_bias=False)(network)  # (BS x H/4 x W/4 x 64)
    network = BatchNormalization()(network)  # (BS x H/4 x W/4 x 64)
    network = LeakyReLU(alpha=0.1)(network)  # (BS x H/4 x W/4 x 64)
    network = MaxPooling2D(2, padding="valid")(network)  # (BS x H/8 x W/8 x 64)

    # Convolutional block 4  (BS x H/8 x W/8 x 64) -> (BS x HG x WG x 128)
    network = Convolution2D(128, (3, 3), strides=1, padding="same", use_bias=False)(network)  # (BS x H/8 x W/8 x 128)
    network = BatchNormalization()(network)  # (BS x H/8 x W/8 x 128)
    network = LeakyReLU(alpha=0.1)(network)  # (BS x H/8 x W/8 x 128)
    network = MaxPooling2D(2, padding="valid")(network)  # (BS x HG x WG x 128)

    # Convolutional block 5  (BS x HG x WG x 128) -> (BS x HG x WG x 256)
    network = Convolution2D(256, (3, 3), strides=1, padding="same", use_bias=False)(network)  # (BS x HG x WG x 256)
    network = BatchNormalization()(network)  # (BS x HG x WG x 256)
    network = LeakyReLU(alpha=0.1)(network)  # (BS x HG x WG x 256)

    # Detection head
    network = YoloDetectionHead(priors, classes)(network)  # (BS x HG x WG x B x (5+C))

    return Model(network_input, network)
