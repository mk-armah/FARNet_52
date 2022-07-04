import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import Config
from data import medical_dataset
from model import Farnet
from test import predict
from train import train_model

model = Farnet()
model.to(Config.device)

train_set = medical_dataset(Config.img_dir, Config.json_dir, Config.resize_h, Config.resize_w, Config.point_num,
                            Config.sigma)


####### please uncomment all test_set1, test_set2 amd test_loader in this file if you have the testing data available


# test_set1 = medical_dataset(Config.test_img_dir1, Config.test_json_dir1, Config.resize_h, Config.resize_w,
#                             Config.point_num, Config.sigma)
# test_set2 = medical_dataset(Config.test_img_dir2, Config.test_json_dir2, Config.resize_h, Config.resize_w,
#                             Config.point_num, Config.sigma)
train_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=True, num_workers=4) 
# test_loader = DataLoader(dataset=test_set1, batch_size=1, shuffle=False, num_workers=4)

criterion = nn.MSELoss(reduction='none')
criterion = criterion.to(Config.device)
optimizer_ft = optim.Adam(model.parameters(), lr=Config.lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_ft, [200], gamma=0.1, last_epoch=-1)
model_ft = train_model(model, criterion, optimizer_ft, scheduler, train_loader, Config.num_epochs)


#torch.save(model_ft, Config.save_model_path)     #provide a directory for saving the model in the config file before uncommenting this line of code



#please remember that pred takes just a single image to run on multiple images , iterate.
# pred = predict(model_ft, img_path = "/content/drive/MyDrive/Coding-Stuffs/Repository/FARNet/CustomDataSet/Training/imgs/cephalo (281).jpg")
# print("predicted points :",pred)




#get_errors(model, test_loader, Config.test_json_dir1, Config.save_results_path) #dont uncomment
