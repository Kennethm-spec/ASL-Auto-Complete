# ASL-Auto-Complete
A CV implementation for real-time ASL detection, completion, and correction for mobile devices.

Authors:
- [Andrew X](https://github.com/Qulxis)
- [Kenneth M](https://github.com/Kennethm-spec)
- [Sam B](https://github.com/sdb2174)
- [Alban D](https://github.com/alban999)
# Setup
To install the version of python libraries used run:
```
$ pip install -r requirements.txt
```
We tested our system on Python Version 3.7-3.9

# Walkthrough
The overal system is composed of MediaPipe's segment detector, data preprocessing sequences, a CNN model, and a front end Flask Webapp that uses the user's webcam and CNN model to allow the user to type using only ASL signs.
The system is lightweight and is suitable for mobile devices: IOS and Android. 

## Data Collection and processing
We use a combination of 3 Kaggle datasets:
- [Synthetic ASL Alphabet](https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet)
- [Synthetic ASL Numbers](https://www.kaggle.com/datasets/lexset/synthetic-asl-numbers)
- [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

We use all the letters from the Synthetic ASL Letters set, numbers 1-4 of the Synthetic ASL Numbers set, and the "space" and "del" classes from the ASL Alphabet dataset. We split it into a training set for both train and validation during training with a total of 900 images per class. We additionally have a test set made of 100 images per class. In total, the datasets require 10 GB of storage which are thus not included in this repository. Instead, we use MediaPipe's hand landmarks extracted from each image and then store them in csv files found in [/src/Modeling/data/](https://github.com/Kennethm-spec/ASL-Auto-Complete/tree/main/src/Modeling/data) which are then saved in training and testing folders. This allows any user to use pre-extracted features to try different models for classification.


## Model Demonstration
1. Clone the repository using 
```
$ git clone https://github.com/Kennethm-spec/ASL-Auto-Complete.git
```
 or [download as zip](https://github.com/Kennethm-spec/ASL-Auto-Complete/archive/refs/heads/main.zip) and extract the files.

2. Navigate to /src/Modeling/train_model.ipynb.
3. Walk through the steps to use the pre-extracted MediaPipe landmarks from images as previous described.
4. Feel free to try different models. We include the one used in our project in the model_saves folder which achieves 98% accuracy on the test set.

## Webapp Demonstraction (Computer)

1. Navigate to /src/app/main_computer.py. At the very bottom on line 287, set the 'host' input to your own IPv4 Address. 
2. Then, run the file in the enviornment that requirments.txt was installed to. 
3. You can then access the webapp by going to "https://INSERT_Your_IPv4_Addres:5003" in your local browser
4. The app allows the user to type by signing letters, autocomplete by signing one of the number signs (1-3), delete characters using the "del" hand sign, and finally make spaces using the "space" hand sign. For reference on these, please refer to the images in the datasets.

## Webapp Demonstration (Mobile)
1. Navigate to /src/app/main_mobile.py and run the script on a host machine that's connected to a Wi-Fi network to deploy the Flask enviornment to the network. 
2. You can then navitage to the IPv4 address of the network+5003 (ex. https://10.100.100.100:5003) on your phone or mobile device and enjoy the app.


# Paper
Coming soon!
