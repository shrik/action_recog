from torch.utils.data import Dataset
import yaml
import os
import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import imgaug.augmenters as iaa
import random
import torchvision

def subdirs(dirs):
    ret = []
    for dir in dirs:
        ret.extend([f.path for f in os.scandir(dir) if f.is_dir()])
    return ret

def load_config(configfile):
    with open(configfile, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def load_frames(itemdirs, num_frames=5):
    ret = []
    for itemdir in itemdirs:
        frames = []
        jpg_files = [f for f in os.listdir(itemdir) if f.endswith('.jpg')]
        jpg_files.sort()
        for jpg_file in jpg_files:
            frame_path = os.path.join(itemdir, jpg_file)
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # 转换为rgb, 因为rknn输入的数据是RGB格式
            frames.append(frame)
        if len(frames) == 11:
            num = (11-num_frames)//2
            assert len(frames[num:-num]) == num_frames
            ret.append(frames[num:-num])
        else:
            raise ValueError(f"Expected 11 frames in {itemdir}, but got {len(frames)}")
    return ret

def to_gif_filename(filename):
    bn = os.path.basename(filename)
    gifname = bn.replace("_", "_bounce_") + ".gif"
    return filename.replace(bn, gifname)

def load_data(config, num_frames=5):
    base_dir = config['base_dir']
    positive_dirs = config['positive_dirs']
    negative_dirs = config['negative_dirs']
    positive_dirs = [os.path.join(base_dir, dir) for dir in positive_dirs]
    negative_dirs = [os.path.join(base_dir, dir) for dir in negative_dirs]
    pos_subdirs = subdirs(positive_dirs)
    neg_subdirs = subdirs(negative_dirs)
    pos_ret = load_frames(pos_subdirs, num_frames)
    neg_ret = load_frames(neg_subdirs, num_frames)
    labels = [1] * len(pos_ret) + [0] * len(neg_ret)
    filenames = [to_gif_filename(dir) for dir in pos_subdirs] + [to_gif_filename(dir) for dir in neg_subdirs]
    data = pos_ret + neg_ret
    data = np.array(data)
    labels = np.array(labels)  
    # Ensure data shape is correct for the model
    print("data shape: ", data.shape)
    print("labels shape: ", labels.shape)
    return data, labels, filenames
    

def load_dataset(configfile, tag="train", train_size=0.8, num_frames=5):
    config = load_config(configfile)['dataset'][tag]
    data, labels, filenames = load_data(config, num_frames)
    data, labels = shuffle(data, labels, random_state=42)   
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=1-train_size, random_state=42)
    return train_data, train_labels, test_data, test_labels


def load_predict_dataset(configfile, tag="test"):
    config = load_config(configfile)['dataset'][tag]
    data, labels, filenames = load_data(config)
    return data, labels, filenames


def augment_frames(frames):
    # Convert frames to torch tensors
    frames = [torch.from_numpy(frame).permute(2, 0, 1) for frame in frames]
    frames = torch.stack(frames)

    # Create transforms pipeline
    transforms = torch.nn.Sequential(
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        torchvision.transforms.RandomApply([
            torchvision.transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
        ], p=0.2),
        
    )

    # Apply same transforms to all frames
    frames = transforms(frames)
    
    # Convert back to numpy arrays
    frames = frames.numpy()
    frames = np.split(frames, len(frames), axis=0)
    frames = [frame.squeeze(0).transpose(1, 2, 0) for frame in frames]
    return frames

# Custom dataset class for handling 5-frame sequences
class ActionDataset(Dataset):
    def __init__(self, data, labels, filenames=[], crop_size=None, augment=False, output_image=False):
        self.data = data
        self.labels = labels
        self.filenames = filenames
        self.crop_size = crop_size
        self.augment = augment
        self.output_image = output_image
    def __crop_frame(self, frame, size, random_shift=None):
        shift_x = 0
        shift_y = 0
        if random_shift:
            shift_x = random.randint(-random_shift, random_shift)
            shift_y = random.randint(-random_shift, random_shift)
        center_x = frame.shape[1] // 2 + shift_x
        center_y = frame.shape[0] // 2 + shift_y
        # 防止越界
        center_x = max(center_x, size//2)
        center_x = min(center_x, frame.shape[1] - size//2)
        center_y = max(center_y, size//2)
        center_y = min(center_y, frame.shape[0] - size//2) 

        start_x = center_x - size // 2
        start_y = center_y - size // 2
        return frame[start_y:start_y+size, start_x:start_x+size]

    def transform(self, data):
        ret = []
        for frame in data:
            if self.crop_size is not None:
                if self.augment:
                    ret.append(self.__crop_frame(frame, self.crop_size, random_shift=35))
                else:
                    ret.append(self.__crop_frame(frame, self.crop_size))
            else:
                ret.append(frame)
        if self.augment:
            ret = augment_frames(ret)
        ret = np.concatenate(ret, axis=0)
        if not self.output_image:
            ret = ret.transpose(2, 0, 1).astype(np.float32)
            ret = ret - 127.5
            ret = ret / 127.5
        return ret

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if len(self.filenames) > 0:
            return torch.tensor(self.transform(self.data[idx])), torch.tensor(self.labels[idx]), self.filenames[idx]
        else:
            return torch.tensor(self.transform(self.data[idx])), torch.tensor(self.labels[idx])
