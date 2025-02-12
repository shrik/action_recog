from action_model import ActionModel
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from action_dataset import load_predict_dataset, ActionDataset
import numpy as np
import onnxruntime


def predict_onnx(features_onnx, classifier_onnx):

    
    # batch_size = inputs.size(0)
    # inputs = inputs.cpu().numpy()  # Convert to numpy for ONNX Runtime
    # labels = labels.cpu().numpy()

    # all_features = []
    # for i in range(5):
    #     # Prepare input in correct format
    #     frame_input = inputs[:, i].astype(np.float32)  # Ensure float32
    #     # Run inference
    #     features = features_onnx.run(None, {'input': frame_input})[0]
    #     all_features.append(features)
        
    # # Concatenate features correctly
    # features = np.concatenate(all_features, axis=1)
    
    # Run classifier
    # import pdb; pdb.set_trace()
    features = np.array([1.0] * 1280*5*7*7).reshape(1, 1280*5, 7, 7).astype(np.float32)
    outputs = classifier_onnx.run(None, {'input': features})
    print("classifier outputs: ", outputs)
    
def predict():
    configfile = 'config.yml'
    features_onnx = onnxruntime.InferenceSession("features_224_224.onnx")
    classifier_onnx = onnxruntime.InferenceSession("classifier_224_224.onnx")
    predict_onnx(features_onnx, classifier_onnx)


import cv2
def predict_features(image_path):
    print("image_path: ", image_path)
    image = cv2.imread(image_path)
    image = image.astype(np.float32)
    image = image.transpose(2, 0, 1)
    image = image.reshape(1, 3, 224, 224)

    # image = np.array([255] * 3*224*224).reshape(1, 3, 224, 224).astype(np.float32)

    image = image - 127.5
    image = image / 127.5
    features_onnx = onnxruntime.InferenceSession("features_224_224.onnx")

    # features = np.array([1.0] * 1280*5*7*7).reshape(1, 1280*5, 7, 7).astype(np.float32)
    outputs = features_onnx.run(None, {'input': image})
    # for c in range(1280):
    #     for h in range(7):
    #         for w in range(7):
    #             print(outputs[0][0][c][h][w], end=" ")
    # print()
    # print(np.sum(outputs[0][0]))
    return outputs[0][0]

import os

def predict_frames_features(frames_path, image_names):
    features_list = []
    index = 0
    for image_name in image_names:
        image_path = os.path.join(frames_path, image_name)
        image = cv2.imread(image_path)
        croped = image[image.shape[0]//2-112:image.shape[0]//2+112, image.shape[1]//2-112:image.shape[1]//2+112, :]
        cv2.imwrite(f"frame_{index}.jpg", croped)
        image_path = f"frame_{index}.jpg"
        features = predict_features(image_path)
        features_list.append(features)
        index += 1
    return features_list

def classifier_step_predict(features_list):

    features = np.concatenate(features_list, axis=0)
    # import pdb;pdb.set_trace()
    features = features.reshape(1, 1280*5, 7, 7)
    features = features.astype(np.float32)

    # features = features_list[0].flatten().tolist() + features_list[1].flatten().tolist() + features_list[2].flatten().tolist() + features_list[3].flatten().tolist() + features_list[4].flatten().tolist()
    # features = np.array(features).reshape(1, 1280*5, 7, 7).astype(np.float32)

    # features = []
    # for i in range(5*1280*7*7):
    #     features.append(i * 0.00001)
    # features = np.array(features).reshape(1, 1280*5, 7, 7).astype(np.float32)

    classifier_onnx = onnxruntime.InferenceSession("classifier_224_224.onnx")
    outputs = classifier_onnx.run(None, {'input': features})
    return outputs[0]

# Example usage:
if __name__ == "__main__":
    # predict()
    frames_dir = "/home/zrgy/workspace/sports/datasets/tennis_land/bounce_event_frames/right_1732185687477"
    frames_dir = "/home/zrgy/workspace/sports/datasets/tennis_land/neg_bounces/right_1732185676412" # neg
    image_names = os.listdir(frames_dir)
    image_names.sort()
    image_names = image_names[3:8]
    print("image_names: ", image_names)
    # image_names = ["right_1732185687357.jpg", 
    #                "right_1732185687381.jpg", 
    #                "right_1732185687405.jpg", 
    #                "right_1732185687429.jpg",
    #                "right_1732185687453.jpg"]
    features_list = predict_frames_features(frames_dir, image_names)
    np.save("features_list.npy", features_list)
    outputs = classifier_step_predict(features_list)
    print(outputs)
    print(np.exp(outputs[0][0]) / np.sum(np.exp(outputs)))
    print(np.exp(outputs[0][1]) / np.sum(np.exp(outputs)))


