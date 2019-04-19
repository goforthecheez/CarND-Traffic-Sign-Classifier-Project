# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/examples.png "Random example from each class"
[image2]: ./writeup_images/train_counts.png "Counts per class (training set)"
[image3]: ./writeup_images/valid_counts.png "Counts per class (validation set)"
[iamge4]: ./writeup_images/grayscaled.png "Some grayscaled examples"

[image5]: ./test_images/no_entry.jpg "Traffic Sign 1: No entry"
[image6]: ./test_images/no_vehicles.jpg "Traffic Sign 2: No vehicles"
[image7]: ./test_images/road_work.jpg "Traffic Sign 3: Road work"
[image8]: ./test_images/roundabout_mandatory.jpg "Traffic Sign 4: Roundabout mandatory"
[image9]: ./test_images/speed_limit_120.jpg "Traffic Sign 5: Speed limit (120km/h)"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

* Writeup: You're reading it!
* Here is a link to my [project code](./Traffic_Sign_Classifier.ipynb).
* Here is a link to the [HTML export](./report.html) of the project code.

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34,799.
* The size of the validation set is 4,410.
* The size of test set is 12,630.
* The shape of a traffic sign image is (32, 32).
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

Here are randomly selected examples from each of the 43 classes:

![alt test][image1]

A number of the images were taken under poor lighting conditions, making it challenging for me to discern the type of traffic sign. Better lighting would make the job easier to me, and I suspect improving the brightness and contrast of images would be make it easier for the model to learn, too.

Here is a bar chart showing the distribution of examples across classes in the training set:

![alt text][image2]

And here is another showing the distribution in the validation set:

![alt text][image3]

We see that the distributions are broadly similar, which is good. We also see that the most frequent classes are about 8 times as frequent as the least frequent classes. This may or may not be severe enough of a class imbalance to affect model training.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

My preprocessing consisted of grayscaling the image and applying `(pixel - 128) / 128` normalization.

As a first step, I decided to convert the images to grayscale, which normalizes the intensity of light in the image. In dim images, this markedly increases the contrast between relatively light and relatively dark parts of the image.

Here are some examples of traffic sign images before and after grayscaling:

![alt text][image4]

As a last step, I normalized the image data because grayscale values range between \[0, 255\]. Since I am using ReLU activations, magnitudes matter less, but if I had been using sigmoids, scaling down the values is important to prevent saturation, which results in very small gradients and slow training. Still, ReLUs have an inflection point at 0, so centering the grayscale range around 0 is useful.

(Previously, I had also tried to increase the brightness by adjusting the value component of the images converted to the HSV color space, but light intensity is also affected by hue, so this didn't work as well as I expected. Grayscaling more properly captures light intensity.

Traffic signs also use high-saturation colors in specific hues. I also tried to extract hue and saturation from HSV color space and incorporating these color channels, but these models were extremely slow to train and the paper "Traffic Sign Recognition with Multi-Scale Convolutional Networks" by Sermanet and LeCun suggests that the error values at convergence are suboptimal. In the end, I decided to stick with just grayscaled images.)

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final architecture is similar to the one described in "Traffic Sign Recognition with Multi-Scale Convolutional Networks" by Sermanet and LeCun. That model is almost the same as LeNet-5, the main difference being that the output of both the first convolution-followed-by-max-pooling stage and the second stage are fed into the classifier.

My final model consisted of the following layers:

| Layer         		|     Description	        															| 
|:---------------------:|:-------------------------------------------------------------------------------------:| 
| Input         		| 32x32x1 grayscaled image 																| 
| Convolution 7x7     	| 1x1 stride, valid padding, 6 feature maps, outputs 26x26x6 							|
| RELU					|																						|
| Max pooling	      	| 2x2 patch, 2x2 stride, outputs 13x13x6												|
| Convolution 7x7	    | 1x1 stride, valid padding, 16 feature maps, outputs 7x7x16							|
| RELU					|																						|
| Max pooling	      	| 2x2 patch, 2x2 stride, outputs 3x3x16													|
| Flatten				| outputs 1058 (concatenates flattened 13x13x6=1014 and 3x3x16=144 max pooling layers)  |
| Fully connected		| outputs 100																			|
| RELU					|																						|
| Fully connected		| outputs 100																			|
| RELU					|																						|
| Softmax				| outputs 43																			|

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

At the start of every epoch, the training set was shuffled.

To train the model, I used an Adam optimizer, which is more robust to less-than-optimal hyperparameter choices than vanilla gradient descent. The Adam algorithm both adapts the learning rate and uses momentum, so it is less likely to get stuck in local optima. The learning rate was set at 0.001; larger learning rates tended to cause unstable loss and accuracy metrics, probably due to the effects of momentum. To allow momentum to stabilization, I used a rather long training time of 100 epochs.

I used a batch size of 128. I trained my model using a g2.2xlarge AWS GPU instance, which has 15 GB of memory. The total memory footprint of 128 x (32, 32) float32 values is about 0.5 MB, which is relatively tiny. However, I stuck with this small batch size, since larger batch sizes seemed to increase overfitting.

For weight initialization, small random values were chosen using the truncated normal with mean 0 and standard deviation 0.1. Bias terms were simply initialized at 0.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 94.51%
* test set accuracy of 93.04%

I used an iterative approach.

* What was the first architecture that was tried and why was it chosen?

My starting architecture was the LeNet-5 architecture, and I used the whole training set (i.e. no resampling) for training each epoch. Traffic signs are stereotyped images, just like numerical digits, so it seemed reasonable that this architecture could be adapted for use on traffic signs. I used a batch size of 128, a learning rate of 0.001, and 10 epochs, as in the [CarND-LeNet-Lab solution](https://github.com/udacity/CarND-LeNet-Lab/blob/master/LeNet-Lab-Solution.ipynb).

* What were some problems with the initial architecture?

This architecture was only able to attain validation accuracies of about 89%, while training loss was just above 0 and training accuracy was almost 99%. Increasing the number of epochs to 50 causes training loss to become negligible and training accuracy to rise to just under 100%. The best validation accuracy seen in those 50 epochs, depending on initialization conditions, might surpass 93%; typically, they achieved 91%. One concludes that the LeNet-5 architecture causes models to overfit the training data. 

* How was the architecture adjusted and why was it adjusted?

To reduce overfitting, I tried to reduce the number of parameters in the model.

I first tried reducing the number of convolution layers to 1 -- specifically, I removed the second convolutional layer. Training accuracies were still near-perfect, but validation accuracies got worse -- about 89% in 50 epochs. I next tried to reduce the number of layers in the classifier by eliminating the 84-unit layer. This also didn't seem to help -- training accuracy was still near-perfect but validation accuracies in 50 epochs, ended to be stay just below 93%. Reducing the number of feature maps and the size of the convolution layer patches seemed neither helpful nor particularly harmful.

Increasing the size of the convolution layer patches to 7x7 seemed to be improve validation accuracy slightly without impacting the training metrics; this makes sense -- handwritten digits are less complex than traffic signs, so combining larger chunks of the image into features could help with overfitting. However, even larger patch sizes caused validation accuracy to drop. I didn't make this change to the model, but I kept it in mind.

Running out of ideas, I tried changing ReLUs to sigmoids, which slowed down training substantially. I also tried adding a convolutional layer, and separately adding hue and saturation features -- both of these changes caused the model to have substantially lower accuracies in both training and validation; i.e. severe underfitting.

Having had little success on the problem, I took a look at the "Traffic Sign Recognition with Multi-Scale Convolutional Networks" paper, in which the authors were able to achieve nearly 99% accuracy! It suggests feeding the output of the both the first and second convolution layers to the classifier and changing the classifier layers to 100 units each. Implementing these change helped raise validation accuracies to consistently over 92% and sometimes over 94% in 50 epochs!

It seems that the model should be viewed as consisting of feature extraction layers (the convolutional layers) and a classifier that takes as input features not the raw image but on the latent features extracted by the feature extraction layers. I observed that not changing the classifier layer sizes caused validation accuracies not to rise. This is probably because the 84-unit layer was acting as a bottleneck on the increased number of features from the convolutional layers. Changing convolutional layer patch sizes to 7x7 bumps validation accuracies to over 94.5% somewhat often, by the end of 50 epochs.

Next I tried to equalize the class distributions in the training set. Neither upsampling, which resamples the smaller classes to be the same size as the largest class, or downsampling, which samples larger classes to be the same size as the smallest class, helped improve accuracies.

* Which parameters were tuned? How were they adjusted and why?

Finally, with the architecture set, I tried tuned the batch size and learning rate. Increasing either tended to make both training and validation losses and accuracies less stable, so I kept the batch size at 128 and the learning rate at 0.001. However, I increased the number up epochs, to ensure that model training had sufficient time to stabilize (in light of training with momentum).

* What are some of the important design choices and why were they chosen?

The largest design decision was to use the "multi-scale" idea from the paper -- feeding finer-grained shape information from the first convolutional layer and coarser-grained shape information from the second convolutional layer into the classifier layers. From handwriting recognition, we know that LeNet-5 is able to detect lines. Traffic signs have two types of shapes: the shape of the sign itself and the shapes of the things written on the sign; it makes sense to try and capture both types of information in the feature extractor.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web, along with the image after grayscaling and resizing:

![alt text][image5]

This image might be difficult to classify because it is set at an angle.

![alt text][image6]

This image might be difficult to classify because the text across the middle might be misinterpreted. (This risk is increased from the significant loss in resolution due to resizing the image so that it can be fed into the model -- the text just looks like a dark line.)

![alt text][image7]

This image might be difficult to classify due to the uneven shadows cast on the image.

![alt text][image8]

This image might be difficult to classify due to the extraneous sausage graffiti, which wasn't seen in the training data.

![alt text][image9]

This image might be difficult to classify for two reasons: the sign is not centered like in the training, and the overhead signs have arrows that might make the model confuse it for a traffic sign.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No entry      		| Children crossing								| 
| No vehicles     		| Children crossing								|
| Road work				| Wild animals crossing							|
| Roundabout mandatory  | Priority road					 				|
| 120 km/h				| Turn left ahead      							|


The model got none of the traffic signs right: an accuracy of 0%. This is abysmal, compared to performance on the test set.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image ("no entry"), the model is dead certain it is a "children crossing" sign, which is completely wrong -- it isn't even the right shape ("no entry" is round whereas "children crossing" is triangular"). The top five softmax probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99998689e-01		| Children crossing								| 
| 1.20743573e-06		| Bicycles crossing								|
| 8.33211686e-08		| Keep left										|
| 2.69659406e-19		| Ahead only					 				|
| 5.06019206e-24	    | Turn right ahead     							|

For the second image ("no vehicles"), the model is again dead certain it is a "children crossing" sign, which is still completely wrong and not the right shape ("no vehicles" is round). The top five softmax probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99999404e-01		| Children crossing								| 
| 6.39302186e-07		| No passing									|
| 1.08406461e-13		| No entry										|
| 6.09433497e-14		| Roundabout mandatory			 				|
| 2.14564538e-19	    | Go straight or right 							|

For the third image ("road work"), the model is somewhat sure (p = 0.62) that it is a "wild animals crossing" sign, which only differs from the actual prediction by the image on the sign. However, overall the predictions are still rather bad -- although the second prediction ("traffic signals"; p = 0.22) is also the right shape, the third prediction ("turn right ahead"; p = 0.12) is not, and the correct prediction does not appear in the top five predictions. The top five softmax probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 6.16209507e-01		| Wild animals crossing							| 
| 2.24655166e-01		| Traffic signals								|
| 1.23108983e-01		| Turn right ahead								|
| 1.71628781e-02		| Double curve					 				|
| 1.42895617e-02	    | No passing for vehicles over 3.5 metric tons	|

For the fourth image ("roundabout mandatory"), the model is dead certain it is is a "priority road" sign. This sign is the wrong shape ("roundabout mandatory" and "priority road" is a diamond), but the error might be understandable, since both signs have rotational symmetry. The top five softmax predictions were

| Probability         	|     Prediction	        							| 
|:---------------------:|:-----------------------------------------------------:| 
| 1.00000000e+00		| Priority road											| 
| 4.84715425e-23		| Roundabout mandatory									|
| 4.95785767e-25		| Speed limit (100km/h)									|
| 2.20021622e-29		| Right-of-way at the next intersection					|
| 4.25444971e-34	    | End of no passing by vehicles over 3.5 metric tons	|

For the fifth image ("speed limit (120km/h)"), the model is rather certain it is a "turn left ahead" sign, which isn't right, but doesn't match any of the overhead signs either. The top softmax probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 8.58690083e-01		| Turn left ahead								| 
| 1.40779093e-01		| Yield											|
| 3.41007195e-04		| Speed limit (60km/h)							|
| 9.74344584e-05		| Stop					 						|
| 7.50607287e-05	    | No entry										|

In conclusion: The model is very brittle.


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?