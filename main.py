import torch
import torch.nn as nn
import torch.optim as optim
from vogn import VOGN
from models import SimpleConvNet
from datasets import Dataset
from utils import train_model
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Test wether GPUs are available
use_cuda =  torch.cuda.is_available()
print("Using Cuda: %s" % use_cuda)

# Set Random Seed
torch.manual_seed(42)

# Load the dataset, 'mnist' or 'cifar10'
dataset = 'mnist'
data = Dataset(dataset)
trainloader = data.get_train_loader(batch_size=128)
testloader = data.get_train_loader(batch_size=128)
N = data.get_train_size() # Save the size of the train set
input_channels = 1
dims = 28

# Initilaize the model, criterion and optimizer
model = SimpleConvNet(input_channels=input_channels, dims=dims)
if use_cuda:
    model = model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = VOGN(model, train_set_size=N, initial_prec=1e2, num_samples=10)
#optimizer = optim.Adam(model.parameters())
model, train_loss, train_accuracy, test_loss, test_accuracy = train_model(model, [trainloader, testloader], criterion,
                                                                          optimizer, num_epochs=2)

# Plot the results
fig, ax = plt.subplots()
ax.plot(train_loss, 'b')
ax.plot(test_loss, 'g')
ax.legend(["Train Loss", "Test Loss"])
plt.ylabel("Log Loss")
plt.xlabel("Epochs")
plt.show()

fig, ax = plt.subplots()
ax.plot(train_accuracy, 'b')
ax.plot(test_accuracy, 'g')
ax.legend(["Train Accuracy", "Test Accuracy"])
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.show()