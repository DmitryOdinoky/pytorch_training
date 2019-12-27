
import torch
import torchvision
from torchvision import transforms, datasets

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

## setting un the dataset

train = datasets.MNIST('', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor()]))
test = datasets.MNIST('', train=False, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)

#%%

## initializing the network


class Net(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28,64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,64)
        self.fc4 = nn.Linear(64,10)
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)   
        
        return  F.log_softmax(x, dim=1)
        
        
net = Net()
print(net)

X = torch.rand((28,28))
X = X.view(-1,28*28)

output = net(X)

#%%

## backprop in 3 epochs

optimizer = optim.Adam(net.parameters(), lr=0.001)

EPOCHS = 3
loss_history = []

for epoch in range(EPOCHS):
    for data in trainset:
        # data is batch of feature sets and labels
        X, y = data
        net.zero_grad()
        output = net(X.view(-1, 28*28))
        loss = F.nll_loss(output, y)
        loss.backward()
        optimizer.step()
        
    loss_history.append(loss.item())    
    print(loss)
    
    
    
#%%

## accuracy revision
        
correct = 0
total = 0

with torch.no_grad():
    for data in trainset:
        X, y = data
        output = net(X.view(-1, 784))
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1
            
print("Accuracy: ", round(correct/total, 3))

#%%

## printing the images and verifying their labeling

import matplotlib as plt

targetDigit = 4

plt.pyplot.imshow(X[targetDigit].view(28,28))

print(torch.argmax(net(X[targetDigit].view(-1,784))[0]))

#%%

import matplotlib.pyplot as plt
import numpy as np

epochs = np.linspace(1, EPOCHS,num=EPOCHS)

fig, ax = plt.subplots(1)



ax.plot(epochs, loss_history[::-1])

plt.show()


