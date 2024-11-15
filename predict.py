import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from IPython.terminal.shortcuts.auto_suggest import accept
# from PyQt5.QtCore.QUrl import toAce
# from setuptools.sandbox import save_path
from tensorflow.python.keras.backend import learning_phase
from tensorflow.python.keras.models import save_model
from tensorflow.python.ops.metrics_impl import accuracy
from torch.cuda import device
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from tqdm import tqdm
import os

# from .resnet import ResNet,resnet18
from myresnet.resnet import ResNet,resnet18,resnet50

predict_path = r"data/val"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

predict_transformer = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

predictset = datasets.ImageFolder(root=predict_path,transform=predict_transformer)
predict_loader = DataLoader(predictset,batch_size=32,num_workers=0,shuffle=False)

def predict(model,predict_loader,class_indict):
    model.eval()
    predictlist = []
    with torch.no_grad():
        for inputs,labels in predict_loader:
            inputs,labels = inputs.to(device),labels.to(device)
            outputs = model(inputs)
            _,predicted = torch.max(outputs,1)
            predicted = predicted.tolist()
            templist = [class_indict[str(i)] for i in predicted]
            predictlist.append([templist])



    return predictlist

if __name__ =='__main__':
    json_file = open("./class_indices.json", "r")
    class_indict = json.load(json_file)
    print(len(class_indict))
    num_class = len(class_indict)
    model = resnet50(num_class).to(device)
    model_weight_path = r"models/best2.pth"
    model.load_state_dict(torch.load(model_weight_path))

    # print(class_indict)

    predictlist = predict(model, predict_loader,class_indict)
    # print(len(predictlist))
    for i in predictlist:
        print(i)

