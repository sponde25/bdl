import torch
import torch.nn.functional as F
from vogn import VOGN
use_cuda = torch.cuda.is_available()

def softmax_predictive_accuracy(logits_list, y, ret_loss = False):
    probs_list = [F.log_softmax(logits, dim=1) for logits in logits_list]
    probs_tensor = torch.stack(probs_list, dim = 2)
    probs = torch.mean(probs_tensor, dim=2)
    if ret_loss:
        loss = F.nll_loss(probs, y, reduction='sum').item()
    _, pred_class = torch.max(probs, 1)
    correct = pred_class.eq(y.view_as(pred_class)).sum().item()
    if ret_loss:
        return correct, loss
    return correct

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, test_samples=100):
    """
    Performs Training and Validation on test set on the given model using the specified optimizer
    :param model: (nn.Module) Model to be trained
    :param dataloaders: (list) train and test dataloaders
    :param criterion: Loss Function
    :param optimizer: Optimizer to be used for training
    :param num_epochs: Number of epochs to train the model
    :return: trained model, test and train metric history
    """
    train_loss_history = []
    train_accuracy_history = []
    test_accuracy_history = []
    test_loss_history = []
    trainloader, testloader = dataloaders

    for epoch in range(num_epochs):
        model.train(True)
        print('Epoch[%d]:' % epoch)
        running_train_loss = 0.
        running_train_correct = 0.
        for i, data in enumerate(trainloader):
            inputs, labels = data
            if use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            if isinstance(optimizer, VOGN):
                def closure():
                    optimizer.zero_grad()
                    logits = model.forward(inputs)
                    loss = criterion(logits, labels)
                    return loss, logits
            else:
                def closure():
                    optimizer.zero_grad()
                    logits = model.forward(inputs)
                    loss = criterion(logits, labels)
                    loss.backward()
                    return loss, logits
            loss, logits = optimizer.step(closure)
            running_train_loss += loss.detach().item() * inputs.shape[0]
            if isinstance(optimizer, VOGN):
                running_train_correct += softmax_predictive_accuracy(logits, labels)

            else:
                _pred = logits.argmax(dim=1, keepdim=True)
                running_train_correct += _pred.eq(labels.view_as(_pred)).sum().item()
            # Print Training Progress
            if i%200 == 199:
                train_accuracy = running_train_correct / (i*inputs.shape[0])
                print('Iteration[%d]: Train Loss: %f   Train Accuracy: %f ' % (i+1, running_train_loss/(i*inputs.shape[0]), train_accuracy))

        train_accuracy = 100 * running_train_correct / len(trainloader.dataset)
        train_loss = running_train_loss / len(trainloader.dataset)

        model.eval()
        test_loss = 0
        total_correct = 0
        device = 'cuda' if use_cuda else 'cpu'
        with torch.no_grad():
            for data, target in testloader:
                data, target = data.to(device), target.to(device)

                if optimizer is not None and isinstance(optimizer, VOGN):
                    raw_noises = []
                    for mc_sample in range(test_samples):
                        raw_noises.append(torch.normal(mean=torch.zeros_like(optimizer.state['mu']), std=1.0))
                    outputs = optimizer.get_mc_predictions(model, data,
                                                          raw_noises=raw_noises)
                    correct, loss = softmax_predictive_accuracy(outputs, target, ret_loss=True)
                    total_correct += correct
                    test_loss += loss

                else:
                    output = model(data)
                    test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    total_correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(testloader.dataset)
        test_accuracy = 100. * total_correct / len(testloader.dataset)

        train_accuracy_history.append(train_accuracy)
        train_loss_history.append(train_loss)
        test_accuracy_history.append(test_accuracy)
        test_loss_history.append(test_loss)
        print('## Epoch[%d], Train Loss: %f   &   Train Accuracy: %f' % (epoch, train_loss, train_accuracy))
        print('## Epoch[%d], Test Loss: %f   &   Test Accuracy: %f' % (epoch, test_loss, test_accuracy))
    return model, train_loss_history, train_accuracy_history, test_loss_history, test_accuracy_history

