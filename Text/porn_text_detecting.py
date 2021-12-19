import gensim
import pandas as pd
import numpy as np
import re
import nltk
nltk.download('stopwords')
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader

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
      
def test(net, testloader):
    correct = 0
    total = 0
    pred = []
    orig = []
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on test data: %0.3f %%' % (
        100 * correct / total))
    

def train(net, trainloader):
    for epoch in range(500): # no. of epochs
        running_loss = 0
        correct = 0
        total = 0
        loss_ = []
        for data in trainloader:
            # data pixels and labels to GPU if available
            inputs, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            # set the parameter gradients to zero
            optimizer.zero_grad()
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            # propagate the loss backward
            loss.backward()
            # update the gradients
            optimizer.step()
 
            running_loss += loss.item()
        print('[Epoch %d] loss: %.3f' %
                      (epoch + 1, running_loss/len(trainloader)), end=' ')
        print('Accuracy of the network on train data: %0.3f %%' % (
        100 * correct / total))
        loss_.append(running_loss/len(trainloader))
 
    print('Done Training')
    return loss_

if __name__ == '__main__':
    word2vec_porn = gensim.models.Word2Vec.load('../word2vec/porn_word2vec.model')

    df_porn_all = pd.read_csv('../data_text/normal_data_for_porn_text.csv')

    y = df_porn_all.target.astype(int).values
    X = df_porn_all.title.apply(lambda x: x.split(' ')).values

    stopword_set = set(nltk.corpus.stopwords.words('russian'))

    dataset = []
    app = ['вк','xxx', 'м', 'н', 'г', 'з', 'вт', 'р', 'й', 'ьк', 'т', 'шк', 'б', 'дв', 'мя', 'п', 'ь']
    for list_ in X:
            dataset.append([token for token in list_ if token not in list(stopword_set) + app])

    corpus = []
    for words in dataset:
        corpus.append(' '.join(words))

    sent2vec = np.empty((len(dataset), 100))
    for idx, sentenses in enumerate(dataset):
        res = 0
        for word in sentenses:
            res += word2vec_porn.wv[word]

        sent2vec[idx] = np.array(res)

    sent2vec = torch.from_numpy(np.array(sent2vec).astype(np.float32))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    X_train, X_test, y_train, y_test = train_test_split(sent2vec, y, test_size=0.1, random_state=42)

    train_data = PornTextDataset(X_train, y_train)
    test_data = PornTextDataset(X_test, y_test)

    trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = DataLoader(test_data, batch_size=64, shuffle=True)

    net = TextClassificationModel().to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')

    train(net, trainloader)
    test(net, testloader)

    torch.save(net, 'PornTextDetector_2.pth')
