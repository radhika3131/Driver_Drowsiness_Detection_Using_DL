# Driver_Drowsiness_Detection_With_OpenCv_And_Keras

This project aims to address the issue of driver drowsiness by using computer vision techniques to monitor the driver's eyes and alert them if signs of drowsiness are detected.  

In this Python project, we will be using OpenCV to gather the images from the webcam and feed them into a Deep Learning model which will classify whether the person’s eyes are ‘Open’ or ‘Closed’. The approach we will be using for this Python project is as follows :
#### Step 1– Take the image as input from a camera.
#### Step 2 – Detect the face in the image and create a Region of Interest (ROI).
#### Step3– Detect the eyes from ROI and feed it to the classifier.
#### Step4– Classifier will categorize whether eyes are open or closed.
#### Step 5– Calculate the score to check whether the person is drowsy.



#### DataSet Link:! http://mrl.cs.vsb.cz/eyedataset 




We have used *Transfer Learning* - Instead of building a new model from the ground up, we have used  pre-trained "knowledge" from one task and applied it to a similar, but slightly different task.

## Table of Contents
- [Project Features](#project-features)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)
- [Future Work](#future-work)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Project Features
- **Transfer Learning**: The project utilizes the pre-trained InceptionV3 model for efficient feature extraction, which reduces the need for large amounts of training data and speeds up the training process.
- **Custom Classification Layers**: The system adds custom layers on top of the InceptionV3 model to classify eye images into two categories: open and closed.
- **Data Augmentation**: To improve the model's robustness and generalization, various data augmentation techniques are applied, such as rotation, shear, zoom, and shifts.
- **Callbacks for Training Optimization**:
  - The model is saved only when it performs best on the validation set.
  - Training stops early if the model stops improving, to prevent overfitting.
  - The learning rate is reduced when the model's performance plateaus, allowing for finer adjustments during training.

## Installation
To set up the project, follow these steps:

1. **Clone the repository**: Download or clone the repository to your local machine.
2. **Install dependencies**: Install the required Python libraries as listed in the `requirements.txt` file.
3. **Prepare your dataset**: Ensure your dataset is organized into training and testing directories, with subdirectories for each class (e.g., open and closed eyes).

## Data Preparation
The dataset used in this project consists of eye images categorized into two classes: **Open_Eye** and **Closed_Eye**. The images undergo preprocessing, including rescaling and augmentation, to ensure the model is exposed to a wide variety of training conditions. This preprocessing helps the model generalize unseen data better.

### Data Augmentation Techniques:
- **Rescaling**: Normalizes the pixel values to a consistent range.
- **Rotation**: Rotates images to simulate different angles of view.
- **Shear and Zoom**: Applies transformations to the images to enhance robustness.
- **Shifts**: Shifts images horizontally and vertically to simulate different positions.

## Model Architecture
The core of the project is built on the InceptionV3 architecture, a deep convolutional neural network known for its efficiency in image classification tasks. The pre-trained InceptionV3 model is used for feature extraction, while custom layers are added to classify the input images into the desired categories.

### Base Model:
- The InceptionV3 model is used without its top layers, allowing for customization specific to this project.

### Custom Layers:
- The custom layers include a flattening layer to convert the model's output into a vector, followed by dense layers with activation functions for classification.

### Freezing Layers:
- The layers of the InceptionV3 model are frozen to preserve the learned features from the ImageNet dataset, focusing the training process on the new classification layers.

## Training the Model
The model is trained using an adaptive learning rate optimizer, with a loss function suited for multi-class classification tasks. The training process is monitored and optimized using callbacks that save the best-performing model, stop training early if needed, and adjust the learning rate based on the model's performance.

## Evaluation
The model is evaluated on both training and validation datasets to ensure it has learned effectively and can generalize well to new, unseen data. The evaluation metrics include accuracy and loss, which are compared across training and validation sets to monitor overfitting.

## Results
The results of the model training and evaluation are recorded, providing insights into the model's performance. The accuracy and loss metrics on both the training and validation datasets are compared to assess the model's generalization capabilities.

## Usage
To use this project:

1. **Train the Model**: Follow the instructions provided to train the model on your dataset.
2. **Evaluate the Model**: Use the provided tools to evaluate the model's performance.
3. **Deploy the Model**: Integrate the trained model into a real-time system for detecting driver drowsiness.

## Future Work
- **Real-Time Detection**: Extend the system to process live video streams for real-time drowsiness detection.
- **Model Optimization**: Explore different architectures and hyperparameters to further improve model performance.
- **Expanded Dataset**: Increase the size and diversity of the dataset to improve the model's robustness.



## License
This project is licensed under the MIT License. For more details, see the [LICENSE](LICENSE) file.

## Acknowledgments
A special thanks to the creators and maintainers of Keras and TensorFlow, as well as the contributors to the ImageNet dataset, for making such resources available.



