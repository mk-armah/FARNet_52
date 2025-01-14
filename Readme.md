# Feature Aggregation and Refinement Network for 2D Anatomical Landmark Detection - FARNet

## Overview from the Original FARNet Research

Feature Aggregation and Refinement Network (FARNet) is a deep neural network proposed by Yueyuan Ao and Hong Wu for automatic detection of anatomical landmarks. The backbone of the network is a CNN pre-trained on natural images, this helps in alleviating the issue of limited training data in the medical domain and also eliminate the need of learning features 100% from scratch. The Architecture also includes a multi-scale feature aggregation module for multiscale feature fusion and a feature refinement module for high resolution heatmap regression.
Loss function for the network is a custom loss named Exponential Weighted Center loss which supresses landmark predictions far from the ground truth and focuses on losses from the pixels near landmarks.

<img src = ".\architecture.PNG" alt = "Model Architecture">

## FARNet_52 Project Briefing

Localization of anatomical landmarks is vital for clinical diagnosis, treatment planning, and research. In 
this project I built on a recent novel deep convolutional neural network architecture, FARNET, which 
localizes 19 landmarks of anatomical images for medical diagnosis. This work seeks to extend the model's 
architecture to localize 52 anatomical landmarks. Other modifications done include the construction of a new pytorch dataset class which takes in an Image directory and a json file Path containing the corresponding image details for training and testing the Network. All other configurations including backbone network,loss functions and optimizers remain unchanged as presented in the original paper.



PS: This configuration was tailored to suit the specific setup of a Client on a freelancing platform. The complete dataset as well as the model's learned weights and biases after training is not open sourced or included in this repository. Few samples of the dataset have been provided to aid in setting up
the dataloader class for your project as well.


## Directory Setup
setup for the new data class should strictly follow the that of the CustomDataset folder provided


├── CustomDataSet <br/>
├── └── imgs <br/>
├──        └──cephalo (281).jpg    #sample image <br/>
├──        └──cephalo (505).jpg    #second sample image <br/>
├──    └── json  <br/>
├──        └──data.json            #single json file to host info of all images <br/>
├── image                           
├── architecture.PNG               
├── config.py                      # ToDo: Config File - customized to suit your os set-up <br/>
├── data.py                        # ToDo: Dataset Class <br/>
├── main.py                        #main pipeline for training and testing <br/>
├── model.py                       # Store EDA results  <br/>
├── Readme.md <br/>
├── requirements.txt <br/>
├── test.py                        # model testing functions <br/>
├── train.py                       # model training functions <br/>
├── utils.py                       # helper functions <br/>


## USAGE 

### config.py
You should set the image path in config by yourself

### Run main.py
Run main.py to train the model and test its performance

### Reference
FARNet Repo - https://github.com/iskenderkahramanoglu/FARNet
