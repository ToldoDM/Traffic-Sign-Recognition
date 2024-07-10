# Traffic Sign Recognition: a performance comparison on challenging weather conditions between CNNs with spatial transformers and ResNet50

## Overview
The task is the classification of a traffic sign among 43 possible classes over real images of traffic signs captured in Germany.

The classification has been done implementing the state-of-the-art models for this task (according to [GTSRB Benchamark](https://benchmark.ini.rub.de/gtsrb_results.html)) and exploring the potentiality of ResNet50 as backbone for transfer learning.

The comparison between the models is presented in our report.
## Features

- **Traffic Sign Recognition:** Classify the traffic signs from images.
- **Convolutational Neural Networks:** Utilize convolutional neural networks for feature extraction.
- **Convolutional Neural Network with Spatial Transformers:** Incorporate spatial transformers to handle geometric transformations and improve accuracy over CNN architecture.
- **Transfer learning using ResNet50:** Training of the last fully connected layers of the backbone keeping the weights for all the other layers locked.

## Dataset
- **GTSRB Dataset:** [German Traffic Sign Recognition Benchmark (GTSRB)](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)
- Description:
  - 43 classes of traffic signs
  - 51840 images of different sizes
  ![](.\images\gtsrb_samples.png) 
- **Weather augmented datasets (Weather50-Weather100):** Adding weather augmentations images to GTSRB dataset with a respectively fixed probability of the augmentation to be applied of 0.5 and 1.
  ![](.\images\aug_example_2.png)

## Models
- **Models** can be found here: https://drive.google.com/drive/folders/1P0Vv7XsjSRJ90Za2QqmW0gQHPsd9ucO1?usp=drive_link

## Classification example of CNN with spatial transformers on a weather augmented dataset
Classification examples of a convolutional neural network with spatial transformers on weather aug-
mented dataset.

![](.\images\weather_classification_examples.png)
The effects starting from the left are: rain, shadow,
sun flare, fog. On the top row the predicted label with its confi-
dence, while on the second row the ground truth class.

## Requirements

- Python 3.11
- PyTorch 2.3.0
- Other libraries listed in `requirements.txt`
