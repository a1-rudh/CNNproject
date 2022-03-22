
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rn
import time

import torch
import torchvision.transforms as transform
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torch.nn as nn
import torch.optim as optim


device= torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using.. ',device)

def accuracy(loader, model):
    """Calculate the accuracy of a given model on a given loader"""
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # calculate outputs by running images through the network
            outs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnv_layer = nn.Sequential(nn.Conv2d(3, 32, 5),
                                       nn.ReLU(),
                                       nn.MaxPool2d(2, 2),
                                       nn.Dropout(p=0.25),
                                       nn.Conv2d(32, 64, 5),
                                       nn.ReLU(),
                                       nn.MaxPool2d(2, 2),
                                       nn.Dropout(p=0.25),
                                       nn.Conv2d(64,128,5),
                                       nn.ReLU(),
                                       nn.MaxPool2d(2, 2),
                                       nn.Dropout(p=0.25),
                                       nn.Conv2d(128,128,5),
                                       nn.ReLU(),
                                       nn.MaxPool2d(2, 2),
                                       nn.Dropout(p=0.25))

        self.lin_layer = nn.Sequential(nn.Linear(4*4*128, 120),
                                          nn.ReLU(),
                                          nn.Dropout(),
                                          nn.Linear(120, 5))

    def forward(self, x):
        x = self.cnv_layer(x)
        x = torch.flatten(x, 1)
        x = self.lin_layer(x)
        return x


path = 'archive/flowers/'
batch_size = 120
seed = 42

inverse = transform.Compose([transform.Normalize(mean = [ 0., 0., 0. ],
                                                 std = [ 1/0.3268945515155792, 1/0.29282665252685547, 1/0.29053378105163574 ]),
                             transform.Normalize(mean = [ -0.4124234616756439, -0.3674212694168091, -0.2578217089176178 ],
                                                 std = [ 1., 1., 1. ])])

aug_1 = transform.Compose([transform.Resize((128, 128)),
                          transform.ToTensor(),
                          transform.Normalize((0.4124234616756439, 0.3674212694168091, 0.2578217089176178),
                                              (0.3268945515155792, 0.29282665252685547, 0.29053378105163574)),
                          transform.RandomHorizontalFlip(p=0.5),
                          transform.RandomRotation(10),
                            transform.RandomErasing(inplace=True, scale=(0.01, 0.23)),
                          transform.RandomAffine(translate=(0.05,0.05), degrees=0)])

aug_2 = transform.Compose([transform.Resize((128, 128)),
                          transform.ToTensor(),
                          transform.Normalize((0.4124234616756439, 0.3674212694168091, 0.2578217089176178),
                                              (0.3268945515155792, 0.29282665252685547, 0.29053378105163574)),
                           transform.RandomVerticalFlip(p=0.5),
                           transform.RandomRotation(5),
                          transform.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                          transform.RandomAffine(degrees=11, translate=(0.1,0.1), scale=(0.8,0.8))])

aug_3 = transform.Compose([transform.Resize((128, 128)),
                          transform.ToTensor(),
                          transform.Normalize((0.4124234616756439, 0.3674212694168091, 0.2578217089176178),
                                              (0.3268945515155792, 0.29282665252685547, 0.29053378105163574)),
                          transform.RandomHorizontalFlip(p=0.5),
                          transform.RandomRotation(15),
                          transform.RandomAffine(translate=(0.08,0.1), degrees=15)])


train_data = ImageFolder(path+'train', transform=transform.Compose([transform.Resize((128, 128)),
                                                                 transform.ToTensor(),
                                                                 transform.Normalize((0.4124234616756439, 0.3674212694168091, 0.2578217089176178),
                                                                                     (0.3268945515155792, 0.29282665252685547, 0.29053378105163574))]))
test_data = ImageFolder(path+'val', transform=transform.Compose([transform.Resize((128, 128)),
                                                              transform.ToTensor(),
                                                              transform.Normalize((0.4124234616756439, 0.3674212694168091, 0.2578217089176178),
                                                                                  (0.3268945515155792, 0.29282665252685547, 0.29053378105163574))]))

aug1_data = ImageFolder(path+'train', transform=aug_1)
aug2_data = ImageFolder(path+'train', transform=aug_2)
aug3_data = ImageFolder(path+'train', transform=aug_3)

train = ConcatDataset([train_data, aug1_data, aug2_data, aug3_data])


trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

classes = ('Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip')



fig,ax=plt.subplots(3,4)
fig.set_size_inches(16,12)
img,y = iter(trainloader).next()
img = inverse(img)
for i in range(3):
    for j in range (4):
        idx=rn.randint(0,batch_size-1)
        # label = list(train_dataset.class_indices.keys())[np.argmax(y[l])]
        npimg = img[idx].numpy()
        ax[i,j].imshow(np.transpose(npimg, (1, 2, 0)))
        ax[i,j].set_title(classes[y[idx]])

plt.tight_layout()


net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay= 1e-4)
if torch.cuda.is_available():
    criterion.to(device)
    net.to(device)
print(net)

train_acc = []
test_acc = []
epochs = 1

start = time.time()
for epoch in range(epochs):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print(i)

    print(f'[Epoch: {epoch + 1}, Steps: {i + 1:5d}] loss after epoch: {running_loss / (i+1):.3f}')
    # train_acc.append(accuracy(trainloader, net))
    test_acc.append(accuracy(testloader, net))
end = time.time()
print('Finished Training')

print('Time taken: ', end - start)

PATH = './model16.pth'
torch.save(net.state_dict(), PATH)

plt.figure(figsize=(5, 8))
plt.plot(range(1,epochs+1), train_acc, label='Training Accuracy')
plt.plot(range(1,epochs+1), test_acc, label='Test Accuracy')
plt.legend()
plt.ylim(0.4,1.0)
plt.xlim(1,epochs+1)
plt.title('Training and Testing Accuracy')
plt.show()

fig,ax=plt.subplots(3,4)
fig.set_size_inches(16,12)
img,y = iter(testloader).next()
img, y = img.to(device), y.to(device)
outputs = net(img)
_, pred= torch.max(outputs, 1)
img = inverse(img)
img = img.to('cpu')
for i in range(3):
    for j in range (4):
        idx=rn.randint(0,batch_size-1)
        # label = list(train_dataset.class_indices.keys())[np.argmax(y[l])]
        npimg = img[idx].numpy()
        ax[i,j].imshow(np.transpose(npimg, (1, 2, 0)))
        if y[idx] == pred[idx]:
            ax[i,j].set_title('Actual: '+classes[y[idx]]+'  Predicted: '+classes[pred[idx]], color = 'green')
        else:
            ax[i,j].set_title('Actual: '+classes[y[idx]]+'  Predicted: '+classes[pred[idx]], color = 'red')

plt.tight_layout()
# print images


print(test_acc)


