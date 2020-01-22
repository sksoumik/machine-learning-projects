from model import create_model
from data_preprocessing import transform_data
import torch
from torch.optim import lr_scheduler
from torch.autograd import Variable


def train(epoch):
    train_loss = []
    train_accuracy = []
    model, optimizer, criterion, exp_lr_scheduler = create_model()
    train_loader, test_loader = transform_data()
    model.train()
    exp_lr_scheduler.step()
    tr_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        # Clearing the Gradients of the model parameters
        optimizer.zero_grad()
        output = model(data)
        pred = torch.max(output.data, 1)[1]
        correct += (pred == target).sum()
        total += len(data)

        # Computing the loss
        loss = criterion(output, target)

        # Computing the updated weights of all the model parameters
        loss.backward()
        optimizer.step()
        tr_loss = loss.item()
        if (batch_idx + 1) % 100 == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \t Accuracy: {} %'
                .format(epoch, (batch_idx + 1) * len(data),
                        len(train_loader.dataset),
                        100. * (batch_idx + 1) / len(train_loader),
                        loss.item(), 100 * correct / total))
            torch.save(model.state_dict(), '../save/model.pth')
            torch.save(model.state_dict(), '../save/optimizer.pth')
    train_loss.append(tr_loss / len(train_loader))
    train_accuracy.append(100 * correct / total)
    return train_loss, train_accuracy
