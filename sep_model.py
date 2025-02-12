from action_model import ActionModel
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from action_dataset import load_predict_dataset, ActionDataset
import numpy as np
import onnxruntime


def predict_onnx(features_onnx, classifier_onnx, predict_loader):
    correct = 0
    total = 0
    false_positive = 0
    false_negative = 0
    running_loss = 0.0
    
    for inputs, labels, filenames in predict_loader:
        batch_size = inputs.size(0)
        inputs = inputs.cpu().numpy()  # Convert to numpy for ONNX Runtime
        labels = labels.cpu().numpy()

        all_features = []
        for i in range(5):
            # Prepare input in correct format
            frame_input = inputs[:, i].astype(np.float32)  # Ensure float32
            # Run inference
            features = features_onnx.run(None, {'input': frame_input})[0]
            all_features.append(features)
            
        # Concatenate features correctly
        features = np.concatenate(all_features, axis=1)
        
        # Run classifier
        # import pdb; pdb.set_trace()
        outputs = classifier_onnx.run(None, {'input': features})[0]
        predicted = np.argmax(outputs, axis=1)
        
        total += len(labels)
        correct += (predicted == labels).sum()
        
        # Calculate false positives and negatives
        false_positive += ((predicted == 1) & (labels == 0)).sum()
        false_negative += ((predicted == 0) & (labels == 1)).sum()
        fp_filenames = [filenames[i] for i in range(len(filenames)) if predicted[i] == 1 and labels[i] == 0]
        print(f"False Positives: {fp_filenames}")
    
    accuracy = 100. * correct / total
    precision = 100. * correct / (correct + false_positive) if (correct + false_positive) > 0 else 0
    recall = 100. * correct / (correct + false_negative) if (correct + false_negative) > 0 else 0
    
    print(f'Accuracy: {accuracy:.2f}%, Precision: {precision:.2f}%, '
          f'Recall: {recall:.2f}%')
    
    
def predict():
    configfile = 'config.yml'
    data, labels, filenames = load_predict_dataset(configfile)

    predict_loader = DataLoader(ActionDataset(data, labels, filenames, crop_size=224, augment=False), batch_size=1, shuffle=False)
    print(f"Number of batches: {len(predict_loader)}, Number of samples: {len(data)}")
    
    features_onnx = onnxruntime.InferenceSession("features_224_224.onnx")
    classifier_onnx = onnxruntime.InferenceSession("classifier_224_224.onnx")
    predict_onnx(features_onnx, classifier_onnx, predict_loader)


def export_onnx():
    device = torch.device("cuda")
    model = torch.load(f'models/model_49.pth').to(device)
    model.eval()
    
    # Export features
    dummy_input = torch.randn(1, 3, 224, 224).to(device).float()
    feature_onnx = torch.onnx.export(model.features, 
                      dummy_input,
                      "features_224_224.onnx", 
                      input_names=['input'],
                      output_names=['output'],
                      opset_version=11,
                      verbose=True)
    
    # Export classifier with correct input shape
    dummy_classifier_input = torch.randn(1, 1280*5, 7, 7).to(device).float()  # Corrected shape
    classifier_onnx = torch.onnx.export(model.classifier, 
                    dummy_classifier_input,
                    "classifier_224_224.onnx", 
                    input_names=['input'],
                    output_names=['output'],
                    opset_version=11,
                    verbose=True)
from rknn.api import RKNN
# Example usage:
if __name__ == "__main__":
    # predict()
    RKNN().init_runtime()
    export_onnx()

