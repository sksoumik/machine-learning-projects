from model import create_model
from train import train
from data_preprocessing import transform_data
from torch.autograd import Variable
import torch
import torch.nn.functional as F


def evaluate(data_loader):
    model, optimizer, criterion, exp_lr_scheduler = create_model()
    valid_loss = []
    valid_accuracy = []
    model.eval()
    loss = 0
    correct = 0
    total = 0
    for data, target in data_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        output = model(data)
        loss += F.cross_entropy(output, target, size_average=False).item()
        pred = torch.max(output.data, 1)[1]
        total += len(data)
        correct += (pred == target).sum()
    loss /= len(data_loader.dataset)
    valid_loss.append(loss)
    valid_accuracy.append(100 * correct / total)
    print('\nAverage Validation loss: {:.5f}\tAccuracy: {} %'.format(
        loss, 100 * correct / total))
    return valid_loss, valid_accuracy


if __name__ == "__main__":
    train_loader, test_loader = transform_data()
    number_epochs = 50
    for epoch in range(number_epochs):
        train(epoch)
        evaluate(test_loader)
