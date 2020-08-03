import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

batch_size_train = 50
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5

# initialize the MNIST training set:
mnist_train_dataset = datasets.MNIST(root='./data',
                                       train=True,
                                       download=True,
                                       transform=torchvision.transforms.ToTensor())

mnist_train = torch.utils.data.DataLoader(
                        mnist_train_dataset,
                        batch_size=batch_size_train,
                        shuffle=True)

# initialize the MNIST test set:
mnist_test_dataset = datasets.MNIST(root='./data',
                                       train=False,
                                       download=True,
                                       transform=torchvision.transforms.ToTensor())
mnist_test = torch.utils.data.DataLoader(
                        mnist_test_dataset,
                        batch_size=batch_size_test,
                        shuffle=True)

cifar_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=torchvision.transforms.ToTensor())
cifar_train = torch.utils.data.DataLoader(
                        cifar_train_dataset,
                        batch_size=batch_size_train,
                        shuffle=True)

cifar_test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=torchvision.transforms.ToTensor())
cifar_test = torch.utils.data.DataLoader(
                        cifar_test_dataset,
                        batch_size=batch_size_test,
                        shuffle=True)

# MEMO CIFAR-10:
# 0=avion, 1=voiture, 2=oiseau, 3=chat, 4=cerf, 5=chien, 6=grenouille, 7=cheval, 8=bateau, 9=camion

# displays images from datasets and outputs for a neural network with several outputs
def plotdata(dataset, indexes, model = None):
    l = []
    for elt in indexes:
        t, _ = dataset[elt]
        l.append(t)
    cte = 30.0/9.0
    k = len(l)
    n = int(np.sqrt(cte * k))
    m = int(k/(n * 1.0))
    if (n*m<k):
        m = m+1
    width=20
    mult = 2
    if model is None:
        mult = 1
    f, ax = plt.subplots(mult*m,n,squeeze=False, figsize=(width,int(width*m/(n*2.0))))
    for i in range(m):
        if model is not None:
            for j in range(n):
                if(j+n*i < k):
                    ax[mult*i+1,j].tick_params(axis=u'both', which=u'both',length=0)
                    ax[mult*i+1,j].set_ylim([-0.5,10.5])
                    ax[mult*i+1,j].set_xlim([-1.5,12.5])
                    ax[mult*i+1,j].set_xticks([])
                    ax[mult*i+1,j].set_xticks(np.arange(0.5,9.5,1), minor=True)
                    ax[mult*i+1,j].set_yticks([])
                    ax[mult*i+1,j].grid(False)
                    ax[mult*i+1,j].set_aspect('equal')
                    cm = plt.cm.get_cmap('RdYlBu_r')
                    L = F.softmax(model(l[j+n*i]), dim=1).cpu().data.numpy().flatten()
                    C = [cm(x) for x in L]
                    ax[mult*i+1,j].barh(range(0,10), [z * 10.0 for z in reversed(L)], color=C)
                    for idx in range(len(L)):
                        if L[idx]>0.02:
                            ax[mult*i+1,j].text(10.0*L[idx]+0.15,len(L)-1-idx+0.1,idx)
                else:
                    ax[mult*i+1,j].axis('off')
        # ------------------
        for j in range(n):
            if(j+n*i < k):
                ax[mult*i+0,j].tick_params(axis=u'both', which=u'both',length=0)
                ax[mult*i+0,j].set_xticks([])
                ax[mult*i+0,j].set_yticks([])
                if k==1:
                    ax[mult*i+0,j].set_xticks(np.arange(0.5,28.5,1), minor=True)
                    ax[mult*i+0,j].set_yticks(np.arange(0.5,28.5,1), minor=True)
                    ax[mult*i+0,j].grid(which='minor')
                ax[mult*i+0,j].grid(False)
                ax[mult*i+0,j].set_xticklabels([])
                ax[mult*i+0,j].set_yticklabels([])
                if l[j + n * i].shape[0] == 3:
                    N = l[j + n * i][:, :, :]
                    ax[mult * i + 0, j].imshow(N.permute(1, 2, 0), )
                else:
                    N = l[j + n * i][0, :, :]
                    ax[mult*i+0,j].matshow(N, cmap='Greys', )
            else:
                ax[mult*i+0,j].axis('off')
    plt.show()

# evaluation on a batch of test data:
def evaluate(model, test_loader):
    batch_enum = enumerate(test_loader)
    batch_idx, (testdata, testtargets) = next(batch_enum)
    model = model.eval()
    oupt = torch.argmax(model(testdata), dim=1)
    result = torch.sum(oupt == testtargets) * 100.0 / len(testtargets)
    model = model.train()
    return result.item()


# iteratively train on batches:
def train(model, optimizer, train_loader, iterations):
    batch_enum = enumerate(train_loader)
    i_count = 0
    for batch_idx, (data, targets) in batch_enum:
        i_count = i_count+1
        outputs = model(data)
        # loss = F.nll_loss(outputs, targets, reduction='mean')
        loss = F.cross_entropy(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i_count == iterations:
            break

# a fully connected neural network with no hidden layer:
class Reseau1_mnist(nn.Module):
    def __init__(self):
        super(Reseau1_mnist, self).__init__()
        self.layer = nn.Linear(784, 10)

    def forward(self, x):
        # x = F.softmax(self.layer(x.view(-1,784)), dim=1)
        x = self.layer(x.view(-1,784))
        return x

reseau1_mnist = Reseau1_mnist().to(device)
optimizer_reseau1_mnist = torch.optim.SGD(reseau1_mnist.parameters(), lr=learning_rate, momentum=momentum)

# a fully connected neural network with no hidden layer:
class Reseau1_cifar(nn.Module):
    def __init__(self):
        super(Reseau1_cifar, self).__init__()
        self.layer = nn.Linear(3072, 10)

    def forward(self, x):
        # x = F.softmax(self.layer(x.view(-1,784)), dim=1)
        x = self.layer(x.view(-1,3072))
        return x

reseau1_cifar = Reseau1_cifar().to(device)
optimizer_reseau1_cifar = torch.optim.SGD(reseau1_cifar.parameters(), lr=learning_rate, momentum=momentum)

# print(evaluate(reseau1_cifar, cifar_test))
# train(reseau1_cifar, optimizer_reseau1_cifar, cifar_train, 1000)
# print(evaluate(reseau1_cifar, cifar_test))
#
# plotdata(cifar_test_dataset, [79, 128, 201, 344, 578, 648, 776, 1201,
#                               1257, 1511, 3291, 5072, 7144, 7458, 9999],
#                              reseau1_cifar)

# a more complex neural network:
# class Reseau2_mnist(nn.Module):
#     def __init__(self):
#         super(Reseau2_mnist, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, 3, 1)
#         self.conv2 = nn.Conv2d(32, 64, 3, 1)
#         self.dropout1 = nn.Dropout2d(0.25)
#         self.dropout2 = nn.Dropout2d(0.5)
#         self.fc1 = nn.Linear(9216, 128)
#         self.fc2 = nn.Linear(128, 10)
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x.view(-1, 1, 28, 28)))
#         x = self.conv2(x)
#         x = self.dropout1(F.max_pool2d(x, 2))
#         x = torch.flatten(x, 1)
#         x = self.dropout2(F.relu(self.fc1(x)))
#         x = self.fc2(x)
#         return x
#
# reseau2_mnist = Reseau2_mnist().to(device)
# optimizer_reseau2_mnist = torch.optim.Adam(reseau2_mnist.parameters(), lr=learning_rate)

# print(evaluate(reseau2_mnist, mnist_test))
# train(reseau2_mnist, optimizer_reseau2_mnist, mnist_train, 1000)
# print(evaluate(reseau2_mnist, mnist_test))

# plotdata(mnist_test_dataset, [79, 128, 201, 344, 578, 648, 776, 1201,
#                               1257, 1511, 3291, 5072, 7144, 7458, 9999],
#                              reseau2)

class Reseau2_cifar(nn.Module):
    def __init__(self):
        super(Reseau2_cifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(12544, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x.view(-1, 3, 32, 32)))
        x = self.conv2(x)
        x = self.dropout1(F.max_pool2d(x, 2))
        x = torch.flatten(x, 1)
        x = self.dropout2(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

reseau2_cifar = Reseau2_cifar().to(device)
optimizer_reseau2_cifar = torch.optim.SGD(reseau2_cifar.parameters(), lr=learning_rate, momentum=momentum)

reseau2_cifar.load_state_dict(torch.load('./data/model.pt'))

print(evaluate(reseau2_cifar, cifar_test))
train(reseau2_cifar, optimizer_reseau2_cifar, cifar_train, 1000)
train(reseau2_cifar, optimizer_reseau2_cifar, cifar_train, 1000)
print(evaluate(reseau2_cifar, cifar_test))

# torch.save(reseau2_cifar.state_dict(), './data/model.pt')