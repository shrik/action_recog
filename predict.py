from action_model import ActionModel
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from action_dataset import load_predict_dataset, ActionDataset


def predict_model(model, predict_loader, device):
    model.eval()
    correct = 0
    total = 0
    false_positive = 0
    false_negative = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, labels, filenames in predict_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Calculate false positives and negatives
            false_positive += ((predicted == 1) & (labels == 0)).sum().item()
            false_negative += ((predicted == 0) & (labels == 1)).sum().item()
            fp_filenames = [filenames[i] for i in range(len(filenames)) if predicted[i] == 1 and labels[i] == 0]
            fn_filenames = [filenames[i] for i in range(len(filenames)) if predicted[i] == 0 and labels[i] == 1]
            print(f"False Positives: {fp_filenames}")
            # print(f"False Negatives: {fn_filenames}")
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    
    accuracy = 100. * correct / total
    epoch_loss = running_loss / len(predict_loader)
    precision = 100. * correct / (correct + false_positive) if (correct + false_positive) > 0 else 0
    recall = 100. * correct / (correct + false_negative) if (correct + false_negative) > 0 else 0
    
    print(f'Test Loss: {epoch_loss:.4f}, '
          f'Accuracy: {accuracy:.2f}%, Precision: {precision:.2f}%, '
          f'Recall: {recall:.2f}%')
import onnxruntime
def predict_onnx(features_onnx, classifier_onnx, predict_loader, device):
    correct = 0
    total = 0
    false_positive = 0
    false_negative = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    features_onnx = onnxruntime.InferenceSession(features_onnx)
    classifier_onnx = onnxruntime.InferenceSession(classifier_onnx)
    
    for inputs, labels, filenames in predict_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        all_features = []
        for i in range(5):
            features = features_onnx.run(None, inputs[:, i])
            all_features.append(features)
        features = torch.cat(all_features, dim=1)
        outputs = classifier_onnx.run(None, features)
        _, predicted = outputs.max(1)

        
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Calculate false positives and negatives
        false_positive += ((predicted == 1) & (labels == 0)).sum().item()
        false_negative += ((predicted == 0) & (labels == 1)).sum().item()
        fp_filenames = [filenames[i] for i in range(len(filenames)) if predicted[i] == 1 and labels[i] == 0]
        fn_filenames = [filenames[i] for i in range(len(filenames)) if predicted[i] == 0 and labels[i] == 1]
        print(f"False Positives: {fp_filenames}")
    
    accuracy = 100. * correct / total
    epoch_loss = running_loss / len(predict_loader)
    precision = 100. * correct / (correct + false_positive) if (correct + false_positive) > 0 else 0
    recall = 100. * correct / (correct + false_negative) if (correct + false_negative) > 0 else 0
    
    print(f'Test Loss: {epoch_loss:.4f}, '
          f'Accuracy: {accuracy:.2f}%, Precision: {precision:.2f}%, '
          f'Recall: {recall:.2f}%')
    
    
import cv2
import numpy as np
def predict_images(model, images, device):
    model.eval()
    data = []
    for image in images:
        image = cv2.imread(image)
        image = image.astype(np.float32)
        image = image.transpose(2, 0, 1)
        image = image.reshape(1, 1, 3, 224, 224)
        image -= 127.5
        image = image / 127.5
        image = torch.tensor(image).to(device)
        data.append(image)
    data = torch.cat(data, dim=1)
    outputs = model(data)
    # _, predicted = outputs.max(1)
    print(outputs)

# Example usage:
if __name__ == "__main__":
    device = torch.device("cuda")
    # # model = ActionModel(num_frames=5).to(device)
    # model = torch.load(f'models/model_49.pth').to(device)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    
    # configfile = 'config.yml'
    # data, labels, filenames = load_predict_dataset(configfile)

    # predict_loader = DataLoader(ActionDataset(data, labels, filenames, crop_size=224, augment=False), batch_size=32, shuffle=False)
    # print(f"Number of batches: {len(predict_loader)}, Number of samples: {len(data)}")
    
    model = torch.load(f'models/model_49.pth').to(device)

    # predict_model(model, predict_loader, device=device)

    predict_images(model, ["frame_0.jpg", "frame_1.jpg", "frame_2.jpg", "frame_3.jpg", "frame_4.jpg"], device)
    # # predict_model(model, predict_loader, device=device)
    # # export this model to onnx
    # # model.eval()
    # # # import pdb; pdb.set_trace()
    # feature_onnx = torch.onnx.export(model.features, 
    #                   (torch.randn(1, 3, 224, 224).to(device).float()), 
    #                   "features_224_224.onnx", 
    #                   input_names=['input'],
    #                   output_names=['output'],
    #                   opset_version=11,
    #                   verbose=True)
    
    # classifier_onnx = torch.onnx.export(model.classifier, 
    #                 (torch.randn(1, 1280*5, 224, 224).to(device).float()), 
    #                 "classifier_224_224.onnx", 
    #                 input_names=['input'],
    #                 output_names=['output'],
    #                 opset_version=11,
    #                 verbose=True)


