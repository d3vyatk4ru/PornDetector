
from torchvision import models
import matplotlib.pyplot as plt
import torch
import os
from config import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = models.resnet34(pretrained=True).eval()

model = models.resnet34(pretrained=True, progress=False).to(device)

for param in model.parameters():
    param.requires_grad = False
    
model.fc = torch.nn.Linear(512, 2).to(device)

params_to_update = model.parameters()

print("Params to learn:")
params_to_update = []
for name,param in model.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)

optimizer = torch.optim.Adam(params_to_update, lr=0.001)
criterion = torch.nn.CrossEntropyLoss()
  
model_, val_acc_history = train_model(model, dataloaders,
                                      criterion, optimizer,
                                      device, num_epochs=10)

torch.save(model_, '/content/drive/MyDrive/model/resnet34_porn_model.pth')
