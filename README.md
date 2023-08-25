# Driver_Drowsiness_Detection_With_OpenCv_And_Keras

This project aims to address the issue of driver drowsiness by using computer vision techniques to monitor the driver's eyes and alert them if signs of drowsiness are detected.  

In this Python project, we will be using OpenCV for gathering the images from webcam and feed them into a Deep Learning model which will classify whether the person’s eyes are ‘Open’ or ‘Closed’. The approach we will be using for this Python project is as follows :
#### Step1– Take image as input from a camera.
#### Step2 – Detect the face in the image and create a Region of Interest (ROI).
#### Step3– Detect the eyes from ROI and feed it to the classifier.
#### Step4– Classifier will categorize whether eyes are open or closed.
#### Step5– Calculate score to check whether the person is drowsy.



#### DataSet Link : ! http://mrl.cs.vsb.cz/eyedataset




We have used *Transfer Learning* - Instead of building a new model from the ground up , we have used  a pre-trained "knowledge" from one task and applying it to a similar, but slightly different task.




## Project Prerequisites
The requirement for this Python project is a webcam through which we will capture images. You need to have Python ) installed on your system, then using pip, you can install the necessary packages.

#### OpenCV – pip install opencv-python (face and eye detection).
#### TensorFlow – pip install tensorflow (keras uses TensorFlow as backend).
#### Keras – pip install keras (to build our classification model).
#### Pygame – pip install pygame (to play alarm sound).


## Output Screensort
![OpenEyesOutput](https://github.com/radhika3131/Driver_Drowsiness_Detection_Using_DL/assets/102825662/40f23dc7-ccde-4e47-97aa-3a0e4fadc3c6)


![CloseEyeOutput](https://github.com/radhika3131/Driver_Drowsiness_Detection_Using_DL/assets/102825662/5882057c-6c5d-4749-a46c-27d90adc40fb)



