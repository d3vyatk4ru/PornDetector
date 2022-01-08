from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class PornTextDataset(Dataset):

    def __init__(self, vectorize_text, labels=None, transforms=None):
        self.X = vectorize_text
        self.y = labels
        self.transforms = transforms
         
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
            
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]
          

class TextClassificationModel(nn.Module):

    def __init__(self):
        super(TextClassificationModel, self).__init__()
        self.fc1 = nn.Linear(100, 250)
        self.fc2 = nn.Linear(250, 50)
        self.fc3 = nn.Linear(50, 2)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()
        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()
        self.fc3.weight.data.uniform_(-initrange, initrange)
        self.fc3.bias.data.zero_()

    def forward(self, vec_text):
        x = F.relu(self.fc1(vec_text))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x))