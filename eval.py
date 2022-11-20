from yolo_v2 import tensor_to_boxes, compute_map

from model import build_model

import numpy as np

import argparse
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="test", help="The dataset name.")
    parser.add_argument("--batch_size", type=int, default=16, help="The batch size for used the testing.")
    parser.add_argument("--scaling", type=int, default=16, help="The scaling factor of the model.")
    parser.add_argument("--alpha", type=float, default=0.1, help="The detection threshold.")
    parser.add_argument("--beta", type=float, default=0.3, help="The NMS threshold.")

    args = parser.parse_args()

    dataset = args.dataset
    batch_size = args.batch_size
    scaling = args.scaling
    alpha = args.alpha
    beta = args.beta

    x_test = np.load(f"dataset/x_{dataset}.npy")
    x_test = np.expand_dims(x_test, axis=-1)

    priors = np.load("weights/priors.npy")

    height, width = x_test.shape[1:3]

    model = build_model(width, height, priors, 10)
    model.load_weights("weights/weights.h5")

    y_predicted = model.predict(x_test, batch_size=batch_size)
    y_predicted = tensor_to_boxes(y_predicted, scaling, alpha, beta)

    y_test = json.load(open(f"dataset/y_{dataset}.json"))

    print("mAP over the test set:", compute_map(y_test, y_predicted, 10, 0.5))
