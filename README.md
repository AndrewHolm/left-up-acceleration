
# Project Title: left-up-acceleration

# Description:
Binary classification between a swipe left and a swipe up. Data gathered from arduino Nano BLE 33 Sense using the onboard accelerometer. Trained in TensorFlow to perform a binary classification. 
The main purpose was to create a model trained on data gathered on the sensor and to transfer the model back to the sensor and run inference on live data. The goal was to run inference on an inexpensive edge device 
while maintaining the same level of accuracy. The project succesfully determines between a sharp left swipe and upward swipe in real time with the same outcome as the model running in Colab. The project is a proof of concept to show if a model could maintain accuracy on an edge device and serves no functional purpose. 

# Installation: 
The project is being run on an Arduino nano 33 BLE sense. 
To run the project you need Arduino IDE as well as a few libraries installed. These include the TensorFlow Lite for Microcontollers library which is no longer available directly from the arduino IDE. 
There is a step by step installation guide that can be found [here](https://github.com/tensorflow/tflite-micro-arduino-examples). You will also need the Arduino_LSM9DS1 library which can be downloaded directly from the library tab in the Arduino IDE. The main project file as well as the header file containing the model (stored in a C array) are both needed. 
