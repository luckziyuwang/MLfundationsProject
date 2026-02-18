import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

data = np.loadtxt('Gaussian 2D Wide.csv', delimiter=',')
[rows,cols] = np.shape(data)
# seperate class 0 and class 1 columns
half_col_len = len(cols)//2    
X_class0 = data[:,0:half_col_len]
X_class1 = data[:,half_col_len:len(cols)]

# TODO
# after make class 0 and 1 data sepearte, how to combine together to form new data set?
# x, x_test, y, y_test = train_test_split(xtrain,labels,test_size=0.2, stratify=labels) 

# 70% for training, 15% for validation, 15% for testing
num_train = int(0.7*rows)
num_test = int(0.15*rows)
num_validation = int(0.15*rows)
print(f"{X_class0} for num_validation")

# TODO data split:
sample_train = data[0:num_train,0:-1]
label_train = data[0:num_train,-1]

sample_test = data[rows-num_test:, -1]
label_test = data[rows-num_test:, -1]

sample_validation = data[]
print(f"{X_class0} for sample_validation")

# define neural network and intialize the network layers
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            # TODO made numbers could be change?
            # input layer
            nn.Linear(2, 64),
            nn.ReLU(),
            # hidden layer
            nn.Linear(64, 32),
            nn.ReLU(),
            # output layer
            nn.Linear(32, 2),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# goal of the network is classification
# initialize the model
net = NeuralNetwork()
# train_tensor = torch.tensor(train.values)

# run iteration to find the boundary
num_epochs = 50
# defined batch size
batch_size = 50
# defined learning rate
learning_rate = 0.01
batch_no = len(sample_train) // batch_size

train_dataloader = DataLoader(sample_train, batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(sample_test, batch_size=batch_size, shuffle=False)


m_values = [1,20,50,100,200]
# defined a Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

er_test = []
# for different epoch
for epoch in m_values:
    for i, data in enumerate(train_dataloader, 0):
        # TODO
        # do we need loop batch size data?
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        # update the gradients based on the training loss for the batch
        loss.backward()
        optimizer.step()


# https://pedromarquez.dev/blog/2022/10/pytorch-classification