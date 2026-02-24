import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def load_dataset(filepath, n_dims):
    df=pd.read_csv(filepath, header=None)
    numpy_array=df.to_numpy()
    [rows,cols]=np.shape(df)
    half_col_len=cols//2
    # Class 0: first n_dims columns
    class0=numpy_array[:,0:half_col_len]
    # Class 1: next n_dims columns
    class1=numpy_array[:,half_col_len:cols]

    # Stack them together and create labels
    whole_data = np.vstack([class0, class1])

    # label to class 0 and 1
    y_class0=np.zeros(class0.shape[0])
    y_class1=np.ones(class1.shape[0])
    y=np.concatenate([y_class0,y_class1])

    return whole_data, y

whole_data, y = load_dataset("Gaussian 2D Wide.csv", n_dims=2)

print("Data shape:", whole_data.shape)
print("Labels shape:", y.shape)
print("First 3 data points:", whole_data[:3])
print("First 3 labels:", y[:3])

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super(NeuralNetwork, self).__init__()
        self.layer1=nn.Linear(input_dim, hidden_size)   # input → hidden layer
        self.relu=nn.ReLU()                             # activation function
        self.layer2=nn.Linear(hidden_size, 1)           # hidden layer → output
        self.Sigmoid=nn.Sigmoid()                       # squish output to 0-1

    def forward(self, x):
        x=self.layer1(x)
        x=self.relu(x)
        x=self.layer2(x)
        x=self.Sigmoid(x)
        return x

def train_model(X, y, input_dim, hidden_size=4, lr=0.01, epochs=500):

    # split X and y into 85/15 and then split the 85 into 70/15 to get 70/15/15 for train/validate/test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42) # temp is the temporary 85 split of data
    X_train, X_validate, y_train, y_validate = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42)

    # standardize features using standard scaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train) # training data is used to calculate mean and std deviation
    # val and test are transformed using the same mean and std deviation calculated from training data
    X_validate = scaler.transform(X_validate)
    X_test = scaler.transform(X_test)

    # convert numpy arrays to PyTorch floating point tensors
    X_train = torch.FloatTensor(X_train)
    X_validate = torch.FloatTensor(X_validate)
    X_test = torch.FloatTensor(X_test)
    # unsqueeze(1) changes shape from (n,) to (N,1) so that it can be compared to model output
    y_train = torch.FloatTensor(y_train).unsqueeze(1)
    y_validate = torch.FloatTensor(y_validate).unsqueeze(1)
    y_test = torch.FloatTensor(y_test).unsqueeze(1)

    # create ANN with the given input dimensions and hidden layer size
    model = NeuralNetwork(input_dim, hidden_size)
    
    criterion = nn.BCELoss() # binary cross entropy between target and input
    optimizer = optim.Adam(model.parameters(), lr=lr) # updates weights using gradients
    # training loop
    for epoch in range(epochs):
        model.train()
        predictions = model(X_train) # probabilities for all training points
        loss = criterion(predictions, y_train) # computes scalar loss
        loss.backward() # calculates gradients
        optimizer.step() # updates parameters using gradients
        optimizer.zero_grad() # clear gradients at end of loop

    # evaluate model on training, validation, and test sets
    model.eval() # switch model from training to evaluation mode
    with torch.no_grad():
        # classifies probabilities as 0 if < 0.5 and 1 if >= 0.5
        train_predictions = (model(X_train) >= 0.5).float()
        validate_predictions = (model(X_validate) >= 0.5).float()
        test_predictions = (model(X_test) >= 0.5).float()
        # .float().mean().item() * 100 converts t/f value into 1.0/0.0, averages over all t/f values, and converts from decimal to percentage
        train_accuracy = (train_predictions == y_train).float().mean().item() * 100
        validate_accuracy = (validate_predictions == y_validate).float().mean().item() * 100
        test_accuracy = (test_predictions == y_test).float().mean().item() * 100

    return train_accuracy, validate_accuracy, test_accuracy

def run_multiple(filename, dims, name, runs=10):
    train_accuracis, val_accuracis, test_accuracis = [], [], []

    for i in range(runs):
        # load each dataset split to class 0 and class 1 with label
        X, y = load_dataset(filename, n_dims=dims)
        # use train model to train data
        train_accuracy, val_accuracy, test_accuracy = train_model(X, y, input_dim=dims)
        # append each trained data into seperate array
        train_accuracis.append(train_accuracy)
        val_accuracis.append(val_accuracy)
        test_accuracis.append(test_accuracy)
    
    # run 10 times and take average result to show in the table
    avg_train = np.mean(train_accuracis)
    avg_val   = np.mean(val_accuracis)
    avg_test  = np.mean(test_accuracis)

    print(f"{name:<25} {avg_train:>7.1f}% {avg_val:>7.1f}% {avg_test:>7.1f}%")

datasets = [
    ("Gaussian 2D Wide.csv",    2, "Gaussian 2D Wide"),
    ("Gaussian 2D Narrow.csv",  2, "Gaussian 2D Narrow"),
    ("Gaussian 2D Overlap.csv", 2, "Gaussian 2D Overlap"),
    ("Gaussian 3D Wide.csv",    3, "Gaussian 3D Wide"),
    ("Gaussian 3D Narrow.csv",  3, "Gaussian 3D Narrow"),
    ("Gaussian 3D Overlap.csv", 3, "Gaussian 3D Overlap"),
    ("Moons 2D Wide.csv",       2, "Moons 2D Wide"),
    ("Moons 2D Narrow.csv",     2, "Moons 2D Narrow"),
    ("Moons 2D Overlap.csv",    2, "Moons 2D Overlap"),
]

print(f"{'Dataset':<25} {'Train':>8} {'Val':>8} {'Test':>8}")
print("-" * 55)

for filename, dims, name in datasets:
    run_multiple(filename, dims, name, runs=10)