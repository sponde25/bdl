# Natural-Gradient VI for Bayesian Neural Nets
Minimal code to run Variational Online Gauss-Newton(VOGN) algorithm on MNIST and CIFAR10 datasets using Convolutional Neural Networks. The optimizer is compatible with any PyTorch model containing fully connected or convolutional layers. Layers without trainable parameters such as pooling layers or batch normalization layers with `affine = False` can also be used. 

## Requirements
The project has following dependencies:
- Python >= 3.5
- PyTorch == 1.0

## Scripts Explanation
There are 5 Python scripts:
- `main.py`: This is the main script to perform training and visualize the results
- `vogn.py`: This script contains the VOGN optimizer
- `datasets.py`: This script downloads and create PyTorch data loaders for MNIST and CIFAR10. The default data folder is `./data/`
- `models.py`: This script contains PyTorch implementations of 3 Layer Convolutional Network and LeNet5. The details of model architectures are in Table 1 of attached note.
- `utils.py`: This script contains the function to perform model training and return metric history on test and train datasets.
  
The simplest way to perform training is to run `python main.py` within a suitable anaconda environment.

## How to use VOGN optimizer
VOGN is intended as a drop-in replacement for Adam optimizer. However, VOGN has two important distinctions from Adam. When the opitmizer is initialized, `model.parameters()` in Adam must be replaced with `model`. Within the training loop, forward pass and evaluating loss must be performed within a `closure()`. It is important to **not** perform `loss.backward()` as that is performed within `opitimizer.step()`. The following is an example of training loop for VOGN:
```python
for data in dataloader:
    inputs, labels = data
    def closure():
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
    loss = optimizer.step()
```