from keras.optimizers import Adam

from yolo_v2 import YoloDetectionLoss, compute_priors, boxes_to_tensor

from model import build_model

import numpy as np

import argparse
import json
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="train", help="The dataset name.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train the model.")
    parser.add_argument("--batch_size", type=int, default=16, help="The batch size for the training.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="The learning rate for the optimizer.")
    parser.add_argument("--priors", type=int, default=3, help="The number of prior used.")
    parser.add_argument("--scaling", type=int, default=16, help="The scaling factor of the model.")

    args = parser.parse_args()

    dataset = args.dataset
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    prior_count = args.priors
    scaling = args.scaling

    if not os.path.exists("weights"):
        os.makedirs("weights")

    x_train = np.load(f"dataset/x_{dataset}.npy")
    y_train = json.load(open(f"dataset/y_{dataset}.json"))

    height, width = x_train.shape[1:3]

    priors = compute_priors(y_train, prior_count, scaling)
    np.save("weights/priors.npy", priors)

    x_train = np.expand_dims(x_train, axis=-1)
    y_train = boxes_to_tensor(y_train, priors, 10, scaling, width, height)

    model = build_model(width, height, priors, 10)
    model.compile(loss=YoloDetectionLoss(), optimizer=Adam(learning_rate))
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)

    model.save_weights("weights/weights.h5")
