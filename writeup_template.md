

**Behavioral Cloning Project**

The goals / steps of this project are as follows:
* Use a built in simulator to collect data of good driving behavior.
* Build a Convolution Neural Network in Keras that predicts steering angles from images.
* Train and validate the model with a training and validation set.
* Test that the model successfully drives around track one without leaving the road.
* Summarize the results in this report.


[//]: # (Image References)

[image1]: ./examples/nVidia_model.png "Model architecture"
[image2]: ./examples/center_2018_02_03_04_01_27_636.jpg "Center"
[image3]: ./examples/left_2018_02_03_04_01_27_636.jpg "Left turn"
[image4]: ./examples/right_2018_02_03_04_01_27_636.jpg "Right turn"


---
### Files & Code review

#### 1.  Required files and final video

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained CNN
* writeup_report.md summarizing the results
* video.mp4 with the result video of the car driving using the trained model

#### 2. Viewing final video
The final video of the project is generated using the Udacity provided simulator and my drive.py file. The car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. CNN code

The *model.py* file contains the code for training CNN and saving trained weights. The file clearly depicts a pipeline I used for training and validating the model. Hopefully comments help you understand code better.

### Model Architecture and Training Strategy

#### 1. Data collection and preprocessing

The goal of the project was to design a CNN so that car can adjust it's steering angle to drive on a pre-designed track by itself. To acheive this, we collected images by driving the car on the same track. An inbuilt simulator helped us acheive this step. Data collected was saved as images and final csv file with a summary of image and corresponding sterring angles were generated. 

Intitally, I collected images from single lap of the track and started training CNN. Later however, based on the model behaviour, modified data to include flipped images, multiple laps of track. In below section will give you more detailed information regarding the preprocessing steps.

#### 1.  Model architecture 

I began using LENET architecture with single lap of data. One important aspect of this problem is that, the model predicts steering angle that is not discrete. Hence had to modify error function to MSE. However, the model didnot give me appropiate results. Going over few resources, swithced to a Nvidea model. This model consists of CNN with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 89-93). It also includes RELU activation layers at the end of each convolution layer. Furhter, it has three fully connected layers after convolutional layers prior to output layer.

![alt text][image1]


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

Using a single lap of data, I began training above CNN. In this case the input is the image taken from a car on track and output is steering angle. To be more clear, steering angle(SA) = 0 when the car drives straight. SA < 0, when turning left and SA > 0 when turning right. Sample images are shown below.

![alt text][image2]
![alt text][image3]
![alt text][image4]

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model to include few dropouts inbetween flattened layers.
Then I noticed that there was little improvement in validation loss. To further improve validation loss, I generated more data to better train the model. This included flipped images and images from multiple camera angles. Once, I obtained a model using original and augmented data, I ran the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, like taking right turn was not happening smoothly.  This was because most of the images included only left turns and model couldnot train properly for right turns. To improve the driving behavior in these cases, I collected data by driving in opposite direction to include more information of steering right. 

At the end of the process, the vehicle was able to drive autonomously around the track without leaving the road as seen in video.mp4.


#### 3. Improving efficiency of training time

One of the fundamental challenges in this project was the time taken to train our model. Even with a GPU it took 40-45 mins for each epoch. Multiple steps were taken to improve training time.

* The first step was to reduce the size of input image to only include road and lanes information.  
* Next, I reduced the number of images by removing redundant data. This was acheived by observing the distribution of steering angle. Most of the time SA = 0.Hence,I made sure augmented data includes more non zero SA images giving us more information of turns. This did reduce the training time without any impact on validation loss. 
* However, usage of a generator function to parallelise pre processing and model training bought down the training time to 15-25 mins. The genrator function is defined in line 33 in my code. 
* Lastly tried to use Early stopping, relaoading previously trained weights as initial weitghts with adam optimizer. Probably this did help me acheive the final model sooner. But there was not much of reduction in training time for each epoch due to these techniques.


