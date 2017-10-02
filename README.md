[//]: # (Image References)

[image1]: ./images/mydog.png "My dog :)"


## Project Overview

Welcome to my Convolutional Neural Networks (CNN) project! I made a dog breed classifier using transfer learning of the ResNet 50 as part of my AI nanodegree at Udacity

![Sample Output][image1]


## Instructions

1. Clone the repository and navigate to the downloaded folder.
	
	```	
		git clone https://github.com/WilliamTakeshi/dog_recognizer.git
		cd dog-project
	```
The steps 2-4 are optional, only if you want to train your own CNN!

2. (Optional) Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/dogImages`. 
3. (Optional) Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/lfw`. 
4. (Optional) Donwload the [VGG-16 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz) for the dog dataset.  Place it in the repo, at location `path/to/dog-project/bottleneck_features`.

5. Obtain the necessary Python packages, and switch Keras backend to Tensorflow.  
	
	For __Mac/OSX__:
	```
		conda env create -f requirements/aind-dog-mac.yml
		source activate aind-dog
		KERAS_BACKEND=tensorflow python -c "from keras import backend"
	```

	For __Linux__:
	```
		conda env create -f requirements/aind-dog-linux.yml
		source activate aind-dog
		KERAS_BACKEND=tensorflow python -c "from keras import backend"
	```

	For __Windows__:
	```
		conda env create -f requirements/aind-dog-windows.yml
		activate aind-dog
		set KERAS_BACKEND=tensorflow
		python -c "from keras import backend"
	```
6. If you want to train your CNN, open the notebook and follow the instructions. 
	
	```
		jupyter notebook dog_app.ipynb
	```
__NOTE:__ Instead of training your model on a local CPU (or GPU), you could use Amazon Web Services to launch an EC2 GPU instance.
__NOTE:__ The notebook have all the answers, I really recommend you try for yourself before reading the jupyter notebook.

7. If you want to use the Flask Web App, just open the dog_app_flask.py

	```
		python dog_app_flask.py
	```
