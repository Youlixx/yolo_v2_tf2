# Object detection using YOLOv2

This implementation of [YOLOv2](https://arxiv.org/abs/1612.08242) was written for a tutorial session given at 
Automatants, the Artificial Intelligence association of CentraleSupélec. A replay of the session is available
[here](https://www.youtube.com/watch?v=8b2oOXX2uuU) (check it out if you can speak French!)

The following packages are required
 - tensorflow
 - opencv-python (used in `test.py` to display the images)

## Dataset

The dataset was synthetically generated, it consists of noisy images with random MNIST digits on it. The goal of the
model is to detect the digits on the image using bounding boxes and classify them into 0-9. The digits are uniformly
distributed over the whole dataset as it is in the original one.

The dataset can be downloaded [here](https://drive.google.com/file/d/1f51LLoxgKkyPmR5YesOFmfmZs2DBpZES/view?usp=share_link).
The files of the archive should be placed in a folder named `dataset` at the root of the project.

## Train

A model can be trained using the following command

`python train.py --datatset train --epochs 20 --batch_size 16 --learning_rate 0.001 --priors 3 --scaling 16`

with the following parameters
 - the name of the dataset (train by default)
 - the number of epochs over which the model will be trained
 - the batch size
 - the learning rate
 - the number of prior used by the model
 - the scaling factor of the model (architecture dependent, should be 16 for the default architecture)

## Test

The model can be tested using the following command

`python test.py --dataset test --batch_size 16 --scaling 16 --alpha 0.1 --beta 0.3`

with the following parameters
 - the name of the dataset (test by default, or test_large which is another test set with large images)
 - the batch size
 - the scaling factor of the model (architecture dependent, should be 16 for the default architecture)
 - the detection threshold alpha
 - the NMS threshold beta

This script will display the images of the test set with the detected bounding boxes

## Evaluation

The model can be evaluated using the mAP metric using the following command

`python eval.py --dataset test --batch_size 16 --scaling 16 --alpha 0.1 --beta 0.3`

with the following parameters
 - the name of the dataset (test by default, or test_large which is another test set with large images)
 - the batch size
 - the scaling factor of the model (architecture dependent, should be 16 for the default architecture)
 - the detection threshold alpha
 - the NMS threshold beta

Note: the current implementation of the mAP is not very accurate. Since the metric is computed across all possible alpha 
values, the NMS should be recomputed each time which would very time expensive. Right now, a simple approximation of the
mAP is given by the script; a more practical way of doing would be by sampling alpha with a fine enough step instead of
using the continuous approach.
