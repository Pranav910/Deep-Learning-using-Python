
# Hand written image classification from scratch


* In this small project we are going to classify handwritten images using simple ANN.

* We will be using image dataset from keras library often called as ***MNIST***  Dataset. 

* The images are of resolution 28x28 pixels.

* The dataset consists of 70k images where will be splitting them in two parts basically for training and testing.

* 60k images will be used for training and the left over 10k will be used for testing. Altough we will not be testing all the 10k images instead will be testin 2-3 images.

* Our ANN architecture will contain 3 layers : 
  * 1st layer will contain 784 nodes i.e, we will be flattening 28x28 image data into a single column.

  * 2nd layer will contain 128 neurons tuned with relu as its activation function.

  * 3rd layer will contain 10 neurons obviously because we have to classify 10 handwritten numbers and tuned with softmax as the activation function.

