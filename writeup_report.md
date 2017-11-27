# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolutional neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Writeup_images/normal_image.jpg "Original Image"
[image2]: ./Writeup_images/BGR2RGB.png "BGR to RGB"
[image3]: ./Writeup_images/image_flipped.png "Flipped Image"
[image4]: ./Writeup_images/normalized.png "Normalized"
[image5]: ./Writeup_images/RMSEplot100epochs.png "RMSE"


## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* run_track1_30mph.mp4, run_track2_9mph.mp4 containing videos of the model driving the car autonomously in the simulator

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolutional neural network with 3 convolution layers and 3 fully connected layers  (model.py lines 68-84) 

The model includes RELU layers to introduce nonlinearity (code line 73, 76, 79), and the data is normalized in the model using a Keras lambda layer (code line 69). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 72, 75, 78). 

The model was trained and validated (model.py line 87) on different data sets (model.py line 24) to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 86).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, driving at different speeds and driving on the second track. All this to help the model generalization.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use the original Nvidia architecture and add modifications in order to enhance the model.

I thought this model might be appropriate because it has been proven successful in real life scenarios.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model by adding dropout layers that would help prevent the loss of generalization in the model.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. I recorded extra frames driving through these spots to improve the driving behavior in these cases and re-trained the network.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. As an additional challenge, I tested the vehicle in the second track. It worked for most of the track eventhough some curves were difficult. Extra training data for these sections would be helfpul to deal with the problem.

#### 2. Final Model Architecture

The final model architecture (model.py lines 68-84) consisted of a convolution neural network with the following layers and layer sizes

```sh
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((55,20), (0,0))))
model.add(Convolution2D(24,5,5, subsample=(2,2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
```

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

Subsequently, I converted the images from BGR to RGB to match the format in which images are processed in the drive.py script.

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to come back to the center in case it deviated from the center path.
Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would help further generalize the model for it would help balance the left turn bias in the first track. For example, here is an image that has then been flipped:

![alt text][image3]

After the collection process, I had 108135 training images (216270 after flipping). I then preprocessed this data by a simple normalization (this process is being done as first step of the architecture):

![alt text][image4]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was around 30 as evidenced by the following plot (however the model was not retrained after 100 epochs for there was little change in RMSE):

![alt text][image5]

I used an adam optimizer so that manually training the learning rate wasn't necessary.
