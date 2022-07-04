import torch
class Config():
    img_dir = '' #image folder directory
    json_dir = '' #json file itself not the folder directory
    test_img_dir2 = '' 
    test_json_dir2 = '' 
    test_img_dir1 = '' #test image directory
    test_json_dir1 = ''#test json directory
    GPU = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_number = 40
    scal_h = 3
    scal_w = 1935 / 640
    resize_h = 800
    resize_w = 640
    sigma = 10
    point_num = 52
    num_epochs = 1 # default was 300...please increase or change to 300 if you are training on a gpu device
    lr = 1e-4
    save_model_path = '' #provide a path to save the model
    save_results_path = '' #provide a path to save the results
