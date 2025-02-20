# 将数据转成单张图片输出


import torch
from torch.utils.data import DataLoader
from action_dataset import load_predict_dataset, ActionDataset
import cv2

def prepare_input_image(configfile, tag="train"):
    data, labels, filenames = load_predict_dataset(configfile, tag=tag)

    data_loader = DataLoader(ActionDataset(data, labels, filenames,
                                           crop_size=224, augment=False,
                                           output_image = True), batch_size=32, shuffle=False)
    for i, (inputs, labels, filenames) in enumerate(data_loader):
        for j, (input, label, filename) in enumerate(zip(inputs, labels, filenames)):
            outputname = filename.split('/')[-1].replace("gif", "jpg")
            cv2.imwrite(f"{dirname}/{outputname}", input.numpy())
        # print(f"Number of batches: {len(predict_loader)}, Number of samples: {len(data)}")

if __name__ == "__main__":
    configfile = 'config.yml'
    dirname = "data/single"
    prepare_input_image(configfile, tag="test")
