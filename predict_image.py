import torch
import cv2
import numpy as np
import sys

device = torch.device("cuda")
model = torch.load(f'models/mbv3_total/model_49.pth').to(device)
model.eval()
image_path = sys.argv[1]
image = cv2.imread(image_path)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
inputdata = np.array(image).astype(np.float32)
inputdata = inputdata.transpose(2, 0, 1)
inputdata = inputdata - 127.5
inputdata = inputdata / 127.5
print(inputdata.shape)
inputdata = inputdata.reshape(1, 3, 1120, 224)
inputdata = torch.tensor(inputdata).to(device)
result = model(inputdata)
print(result)

import onnxruntime
classifier_onnx = onnxruntime.InferenceSession("models/mbv3_1120_224.onnx")
inputdata = inputdata.cpu().numpy()
inputdata = inputdata.reshape(1, 3, 1120, 224)
outputs = classifier_onnx.run(None, {'input': inputdata})
print(outputs)