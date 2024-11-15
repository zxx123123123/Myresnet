import json
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#图像变换
train_transformer = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

test_transformer = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

trainset = datasets.ImageFolder(root=r'data/train',transform=train_transformer)
testset = datasets.ImageFolder(root=r'data/test',transform=test_transformer)

train_loader = DataLoader(trainset,batch_size=32,num_workers=0,shuffle=True)
test_loader = DataLoader(testset,batch_size=32,num_workers=0,shuffle=False)

def train(model,train_loader,criterion,optimizer,num_epochs,save_path):
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs,labels in tqdm(train_loader,desc=f"epoch:{epoch + 1}/{num_epochs}",unit = 'batch'):
            inputs,labels = inputs.to(device),labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss/len(train_loader.dataset)
        print(f"epoch[{epoch + 1}/{num_epochs},train_loss:{epoch_loss:.4f}]")
        accuracy = evaluate(model,test_loader,criterion)
        if accuracy > best_acc :
            best_acc = accuracy
            save_model(model,save_path)
            print("model saved with best acc",best_acc)

def evaluate(model,test_loader,criterion):
    model.eval()
    test_loss = 0.0
    correct = 0.0
    total = 0
    with torch.no_grad():
        for inputs,labels in test_loader:
            inputs,labels = inputs.to(device),labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _,predicted = torch.max(outputs,1)
            total = total + labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = test_loss / len(test_loader.dataset)
    accuracy = 100.0 * correct / total
    print(f'Test_loss:{avg_loss:.4f},Accuracy:{accuracy:.4f}%')
    return accuracy

def save_model(model,save_path):
    torch.save(model.state_dict(),save_path)

if __name__ =='__main__':
    num_epochs = 10
    learning_rate = 0.001
    num_class = 2
    save_path = r"models/best3.pth"
    model = resnet18(num_class).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr = learning_rate)

    flower_list = trainset.class_to_idx
    #   采用遍历方法，将分类名称的key与value反过来
    cla_dict = dict((val, key) for key, val in flower_list.items())
    #   将字典cla_dict编码为json格式
    json_str = json.dumps(cla_dict, indent=4)
    with open("class_indices.json", "w") as json_file:
        json_file.write(json_str)

    print('Train start')
    train(model,train_loader, criterion, optimizer, num_epochs,save_path)
    print('Train finish')
    evaluate(model, test_loader, criterion)






