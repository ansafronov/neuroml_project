# Generic ML Libraries
import sklearn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight

# General Libraries
import numpy as np
from scipy.io import loadmat, savemat
from scipy.fft import fft, fftfreq, ifft
import h5py
import os
import pickle

# Figure Libraries
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import LinearSegmentedColormap

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from collections import defaultdict


## IMPORT DATA

datadir = "Spectral_Explainability"
filename = "segmented_sc_data.pkl"

with open(os.path.join(datadir,filename), 'rb') as f:
    mat_file = pickle.load(f)
X = np.float32(mat_file['data']); # data
Y = np.float32(mat_file['label']); # labels
S = np.float32(mat_file['subject']); # subject number

print(np.shape(X))
print(np.shape(Y))
print(np.shape(S))

# Remove Marked Samples
X = X[np.squeeze(Y)!=8*np.ones_like(np.squeeze(Y)),...]
S = S[np.squeeze(Y)!=8*np.ones_like(np.squeeze(Y)),...]
Y = Y[np.squeeze(Y)!=8*np.ones_like(np.squeeze(Y)),...]
print(np.shape(X))
print(np.shape(Y))
print(np.shape(S))

## ChannelDropout Code
class ChannelDropout(nn.Module):
    def __init__(self, rate, noise_shape=None, seed=None):
        super().__init__()
        self.rate = min(1., max(0., rate))
        self.noise_shape = noise_shape
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)

    def forward(self, inputs):
        if self.training:
            noise_shape = [1, 1, inputs.shape[2]]
            mask = torch.bernoulli(torch.full(noise_shape, 1 - self.rate))
            mask = mask.expand_as(inputs)
            return inputs * mask / (1 - self.rate)
        else:
            return inputs

    def extra_repr(self):
        return f'rate={self.rate}, noise_shape={self.noise_shape}, seed={self.seed}'
    

class ModelMDD(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.n_features = 19

        self.channel_dropout = ChannelDropout(rate=0.25)
        
        # Conv1D layers
        self.conv1 = nn.Conv1d(self.n_features, 5, kernel_size=10, stride=1, padding=0)
        self.conv2 = nn.Conv1d(5, 10, kernel_size=10, stride=1, padding=0)
        self.conv3 = nn.Conv1d(10, 10, kernel_size=10, stride=1, padding=0)
        self.conv4 = nn.Conv1d(10, 15, kernel_size=5, stride=1, padding=0)
        
        # BatchNorm layers
        self.bn1 = nn.BatchNorm1d(5)
        self.bn2 = nn.BatchNorm1d(10)
        self.bn3 = nn.BatchNorm1d(10)
        self.bn4 = nn.BatchNorm1d(15)
        
        # Dense layers
        self.fc1 = nn.Linear(15 * 181, 64)  # 184 is calculated based on the input size and conv/pool operations
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 5)
        
        self.dropout = nn.AlphaDropout(dropout)
        
    def forward(self, x):
        # channel dropout
        x = self.channel_dropout(x)

        # Conv layers
        x = F.elu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        x = self.bn1(x)
        
        x = F.elu(self.conv2(x))
        x = F.max_pool1d(x, 2)
        x = self.bn2(x)
        
        x = F.elu(self.conv3(x))
        x = F.max_pool1d(x, 2)
        x = self.bn3(x)
        
        x = F.elu(self.conv4(x))
        x = F.max_pool1d(x, 2)
        x = self.bn4(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Dense layers
        x = self.dropout(x)
        x = F.elu(self.fc1(x))
        x = self.dropout(x)
        x = F.elu(self.fc2(x))
        x = self.dropout(x)
        x = F.softmax(self.fc3(x), dim=1)
        
        return x
    
def get_model(dropout=0.5):
    model = ModelMDD(dropout)
    return model

model = get_model()
model.train()
print(model)
model = []

def evaluate_model(X_train, X_val, Y_train, Y_val, checkpoint_path):
    # Convert numpy arrays to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    X_val = torch.FloatTensor(X_val)
    Y_train = torch.LongTensor(Y_train)
    Y_val = torch.LongTensor(Y_val)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128)

    # Get model
    model = get_model()

    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(Y_train.numpy()), 
                                         y=Y_train.numpy().squeeze())
    class_weights = torch.FloatTensor(class_weights)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.00075)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, verbose=True)

    # Training loop
    num_epochs = 200
    best_val_acc = 0
    patience = 20
    counter = 0

    history = defaultdict(list)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            print(inputs.shape)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = val_correct / val_total

        # Print epoch results
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        history['acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['loss'].append(history)
        history['val_loss'].append(history)

        # Learning rate scheduler step
        scheduler.step(val_acc)

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping")
                break

    # Load best model
    model.load_state_dict(torch.load(checkpoint_path))

    return model, history  # Return None instead of history, as PyTorch doesn't have a built-in history object


# Run Classifier for 10 Folds
n_folds = 10
Y_pred = []; Y_test_all = []; Y_pred_val = []; Y_val_all = [];
val_loss = []; train_loss = [];
val_acc = []; train_acc = [];
Sample_Idx = np.expand_dims(np.arange(np.shape(Y)[0]),axis=1) 

count = 0
# split data into Train/Val and Test Groups
cv = GroupShuffleSplit(n_splits=10,test_size=0.1,train_size=0.9,random_state=0)
for train_val_idx, test_idx in cv.split(X,Y,S):
    X_train_val = X[train_val_idx,...]
    Y_train_val = Y[train_val_idx,...]
    S_train_val = S[train_val_idx,...]
    X_test = X[test_idx,...]
    Y_test = Y[test_idx,...]
    S_test = S[test_idx,...]
    Sample_Idx_Test = Sample_Idx[test_idx,...]
    
    # Split Train/Val Data into Training and Validation Groups
    cv2 = GroupShuffleSplit(n_splits=1,test_size=0.10,train_size=0.90,random_state=0)
    for train_idx, val_idx in cv2.split(X_train_val,Y_train_val,S_train_val):
        X_train = X_train_val[train_idx,...]
        Y_train = Y_train_val[train_idx,...]
        S_train = S_train_val[train_idx,...]
        X_val = X_train_val[val_idx,...]
        Y_val = Y_train_val[val_idx,...]
        S_val = S_train_val[val_idx,...]
    X_train_val = []; Y_train_val = []; S_train_val = []
    
    # Define Model Checkpoints
    file_path = "sleep_pretrain_ckpt/sleep_model_Fold"+str(count)+".pt"

    # Evaluate model
    model, history = evaluate_model(X_train, X_val, Y_train, Y_val, checkpoint_path=file_path)
    
    print('Train Acc')
    print(np.max(history['acc']))
    print('Val Acc')
    print(np.max(history['val_acc']))
    
    val_loss.append(history['val_loss']); train_loss.append(history['loss'])
    val_acc.append(history['val_acc']); train_acc.append(history['acc'])

    # Load Weights of Best Model for Fold
    model.load_state_dict(torch.load(file_path))

    Y_pred.append(np.argmax(model.predict(np.repeat(X_test,repeats=19,axis=2)),axis=1))
    Y_test_all.append(Y_test)
    
    Y_pred_val.append(np.argmax(model.predict(np.repeat(X_val,repeats=19,axis=2)),axis=1))
    Y_val_all.append(Y_val)

    # Save Test Data for Fold
    # dc = {'X':X_test, 'Y':Y_test, 'subject': S_test, 'Sample_Idx':Sample_Idx_Test}
    # savemat("/home/users/cellis42/Spectral_Explainability/PreTraining/test_data" + str(count) + ".mat",dc)
    
    print(count)
    count += 1
  

# Output In Depth Validation Results
n_folds = 10
conf_mat = np.zeros((5,5))
precision = np.zeros((5,n_folds))
recall = np.zeros((5,n_folds))
f1 = np.zeros((5,n_folds))
f1_ind = []

for i in range(n_folds):
    conf_mat += confusion_matrix(Y_val_all[i],Y_pred_val[i])
    metrics = np.array(sklearn.metrics.precision_recall_fscore_support(Y_val_all[i], Y_pred_val[i], beta=1.0, average=None))
    precision[:,i] = np.array(metrics)[0,:]
    recall[:,i] = np.array(metrics)[1,:]
    f1[:,i] = np.array(metrics)[2,:]
    f1_ind.append(sklearn.metrics.f1_score(Y_val_all[i], Y_pred_val[i],average = 'weighted')) # Compute Weighted F1 Score

print('Validation Results')
print(np.int64(conf_mat))
print('Precision Mean')
print(np.mean(precision,axis=1))
print('Precision SD')
print(np.std(precision,axis=1))
print('Recall Mean')
print(np.mean(recall,axis=1))
print('Recall SD')
print(np.std(recall,axis=1))
print('F1 Mean')
print(np.mean(f1,axis=1))
print('F1 SD')
print(np.std(f1,axis=1))

# Output In Depth Test Results
n_folds = 10
conf_mat = np.zeros((5,5))
precision = np.zeros((5,n_folds))
recall = np.zeros((5,n_folds))
f1 = np.zeros((5,n_folds))
f1_ind = []

for i in range(n_folds):
    conf_mat += confusion_matrix(Y_test_all[i],Y_pred[i])
    metrics = np.array(sklearn.metrics.precision_recall_fscore_support(Y_test_all[i], Y_pred[i], beta=1.0, average=None))
    precision[:,i] = np.array(metrics)[0,:]
    recall[:,i] = np.array(metrics)[1,:]
    f1[:,i] = np.array(metrics)[2,:]
    f1_ind.append(sklearn.metrics.f1_score(Y_test_all[i], Y_pred[i],average = 'weighted')) # Compute Weighted F1 Score

print('Test Results')
print(np.int64(conf_mat))
print('Precision Mean')
print(np.mean(precision,axis=1))
print('Precision SD')
print(np.std(precision,axis=1))
print('Recall Mean')
print(np.mean(recall,axis=1))
print('Recall SD')
print(np.std(recall,axis=1))
print('F1 Mean')
print(np.mean(f1,axis=1))
print('F1 SD')
print(np.std(f1,axis=1))
