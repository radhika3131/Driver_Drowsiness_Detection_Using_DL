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




## Project Prerequisites
The requirement for this Python project is a webcam through which we will capture images. You need to have Python ) installed on your system, then using pip, you can install the necessary packages.

#### OpenCV – pip install opencv-python (face and eye detection).
#### TensorFlow – pip install tensorflow (keras uses TensorFlow as backend).
#### Keras – pip install keras (to build our classification model).
#### Pygame – pip install pygame (to play alarm sound).




