
import numpy as np
import os
import torch
import torchvision
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json

from config import Config 


class medical_dataset(torch.utils.data.Dataset):
    def __init__(self,imagePath,jsonfile_path,resize_height, resize_width, point_num, sigma, transform=None):
        """Pytorch Data set class to map ground truth image to ground truth label and enabling indexing and slicing
        To be used with dataloader
        
        Args:
            imagePath: file directory where the image data is stored
            jsonfile_path: file directory where the single json file is stored. Note however that the single jsonfile
            must contain all information for the images in the imagePath
            resize_height: desired height of images
            resize_width : desired width of images
            pint_num : number of points to localize
            sigma: a hyperparameter to control the spread of the peak, see original paper for more details 
        Return:
            image : the original input image
            heatmaps : predicted heatmaps
            heatmaps_refine :predicted refine heatmaps
            img_name:input image name 
            x_all: heatmap x-axis points 
            y_all:  heatmap y-axis points"""


        self.resize_height = resize_height
        self.resize_width = resize_width
        self.point_num = point_num
        self.sigma = sigma
        self.heatmap_height = int(self.resize_height)
        self.heatmap_width = int(self.resize_width)
        self.imagePath = imagePath
        self.images = os.listdir(self.imagePath) #list the image directories
        self.img_nums = len(self.images)
        self.jsonfile_path = jsonfile_path 
        self.transform = transform
        
        with open(self.jsonfile_path,"r") as f:
            self.loaded_json = json.load(f)

    def __len__(self):
        return len(self.images)
        
    def __getitem__(self,idx):
        index = idx % self.img_nums
        img_name = self.images[index]
        
        image = os.path.join(self.imagePath,self.loaded_json[idx]['imagePath']) #get the same image index from the image directory using the imagePath info from the json file
        
        image,scal_ratio_w, scal_ratio_h = self.img_preproccess(img_path = image)
        
        x_all = []
        y_all = []

        for i in self.loaded_json[idx]["points"]: 
            x_all.append(i["X"]),y_all.append(i["Y"])
            
        x_all = np.array(x_all)/scal_ratio_w
        y_all = np.array(y_all)/scal_ratio_h
      
        heatmaps = self.get_heatmaps(x_all, y_all, self.sigma)
        heatmaps_refine = self.get_refine_heatmaps(x_all / 2, y_all / 2, self.sigma)
        heatmaps = self.data_preproccess(heatmaps)
        heatmaps_refine = self.data_preproccess(heatmaps_refine)
        
        return image, heatmaps, heatmaps_refine, img_name, x_all, y_all
    
    def get_heatmaps(self, x_all, y_all, sigma):
        heatmaps = np.zeros((self.point_num, self.heatmap_height, self.heatmap_width))
        for i in range(self.point_num):
            heatmaps[i] = CenterLabelHeatMap(self.heatmap_width, self.heatmap_height, x_all[i], y_all[i], sigma)
        heatmaps = np.asarray(heatmaps, dtype="float32")
        return heatmaps

    def get_refine_heatmaps(self, x_all, y_all, sigma):
        heatmaps = np.zeros((self.point_num, int(self.heatmap_height / 2), int(self.heatmap_width / 2)))
        for i in range(self.point_num):
            heatmaps[i] = CenterLabelHeatMap(int(self.heatmap_width / 2), int(self.heatmap_height / 2), x_all[i],
                                             y_all[i], sigma)
        heatmaps = np.asarray(heatmaps, dtype="float32")
        return heatmaps

    def img_preproccess(self, img_path):
        img = cv2.imread(img_path)
        img_h, img_w, _ = img.shape
        img = cv2.resize(img, (self.resize_width, self.resize_height))
        img = np.transpose(img, (2, 0, 1))
        scal_ratio_w = img_w / self.resize_width
        scal_ratio_h = img_h / self.resize_height

        img = torch.from_numpy(img).float()

        transform1 = torchvision.transforms.Compose([
            # transforms.Normalize([121.78, 121.78, 121.78], [74.36, 74.36, 74.36])
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
        )
        img = transform1(img)

        return img, scal_ratio_w, scal_ratio_h

    def data_preproccess(self, data):
        data = torch.from_numpy(data).float()
        return data


def CenterLabelHeatMap(img_width, img_height, c_x, c_y, sigma):
    X1 = np.linspace(1, img_width, img_width)
    Y1 = np.linspace(1, img_height, img_height)
    [X, Y] = np.meshgrid(X1, Y1)
    X = X - c_x
    Y = Y - c_y
    D2 = X * X + Y * Y
    E2 = 2.0 * sigma * sigma
    Exponent = D2 / E2
    heatmap = np.exp(-Exponent)
    # heatmap[int(c_y)][int(c_x)] = 2
    return heatmap
