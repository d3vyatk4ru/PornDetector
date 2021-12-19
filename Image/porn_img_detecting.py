from torchvision import transforms
from torchvision import models
import matplotlib.pyplot as plt
import torch
import os
from torchvision import datasets
from torch.utils.data import DataLoader
import time
import copy
from tqdm import tqdm

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ]
)

datasets = {
    'train': datasets.ImageFolder(
        root='/content/drive/MyDrive/data/train/',
        transform=transform
    ),
    'valid': datasets.ImageFolder(
        root='/content/drive/MyDrive/data/test/',
        transform=transform
    ),
}

batch_size = 64
num_workers = 8
dataloaders = {
    'train': DataLoader(
        datasets['train'], 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    ),
    'valid': DataLoader(
        datasets['valid'], 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
}

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

def train_model(model, dataloaders, criterion, optimizer, num_epochs=30):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
            if phase == 'valid':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history
  
model_, val_acc_history = train_model(model, dataloaders, criterion, optimizer, num_epochs=10)

torch.save(model_, '/content/drive/MyDrive/model/resnet34_porn_model.pth')
