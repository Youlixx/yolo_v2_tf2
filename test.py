from yolo_v2 import tensor_to_boxes, compute_map

from model import build_model

import numpy as np

import argparse
import json
import cv2


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

    scaling_factor = 1024 / height
    scaled_width = int(scaling_factor * width)
    scaled_height = int(scaling_factor * height)

    for index in range(len(x_test)):
        boxes = y_predicted[index]
        image = x_test[index]

        image = cv2.resize(image, (scaled_width, scaled_height), interpolation=cv2.INTER_NEAREST_EXACT)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        for box in boxes:
            x0, y0, x1, y1 = map(lambda x: int(scaling_factor * x), box[:4])
            label = int(box[4])

            if x0 < 0 or x0 > scaled_width or x1 < 0 or x1 > scaled_width \
                    or y0 < 0 or y0 > scaled_height or y1 < 0 or y1 > scaled_height:
                continue

            image = cv2.rectangle(image, (x0, y0), (x1, y1), color=(0, 255, 0), thickness=2)
            image = cv2.putText(image, str(label), (x0 + 5, y0), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)

        cv2.imshow("Detections", image)

        if cv2.waitKey() == 27:
            break
