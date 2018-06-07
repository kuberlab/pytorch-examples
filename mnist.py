from __future__ import print_function
import argparse
import gzip
import logging
import os
from os import path

from mlboardclient.api import client
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torchvision import transforms
from torchvision.datasets import mnist


logging.basicConfig(
    format='%(asctime)s %(levelname)-10s %(name)-25s [-] %(message)s',
    level='INFO'
)
logging.root.setLevel(logging.INFO)
LOG = logging.getLogger('main')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            train_loss = loss.item()
            LOG.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), train_loss))
        if batch_idx % (args.log_interval * 10) == 0:
            if use_mlboard:
                mlboard.update_task_info({'train_loss': train_loss, 'train_epoch': epoch})


def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    LOG.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy)
    )
    if use_mlboard:
        mlboard.update_task_info({'test_loss': test_loss, 'accuracy': accuracy / 100.})


def save_checkpoint(state, filename):
    torch.save(state, filename)


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument(
        '--data-dir',
        default=os.environ.get('DATA_DIR'),
        required=True,
    )
    parser.add_argument(
        '--training_dir',
        default=os.environ.get('TRAINING_DIR'),
        required=True,
    )
    parser.add_argument('--out-dir', required=True)
    parser.add_argument(
        '--skip-mlboard',
        action='store_true',
        default=False,
    )
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mlboard = not args.skip_mlboard
    mlboard = client.Client()

    LOG.info('-' * 53)
    if use_cuda:
        LOG.info('Use CUDA for processing.')
    else:
        LOG.info('Do not use CUDA.')
    LOG.info('-' * 53)

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    data_dir = args.data_dir
    training_dir = args.training_dir

    if not path.exists(training_dir):
        os.makedirs(training_dir)

    # If already processed
    if (not path.exists(path.join(training_dir, mnist.MNIST.processed_folder, mnist.MNIST.training_file)) or
            not path.join(training_dir, mnist.MNIST.processed_folder, mnist.MNIST.test_file)):
        # process and save as torch files
        LOG.info('Processing dataset...')

        files = os.listdir(data_dir)
        for file in files:
            full_path = path.join(data_dir, file)
            save_path = path.join(training_dir, file.replace('.gz', ''))
            with open(save_path, 'wb') as out_f, gzip.GzipFile(full_path) as zip_f:
                out_f.write(zip_f.read())

        training_set = (
            mnist.read_image_file(path.join(training_dir, 'train-images-idx3-ubyte')),
            mnist.read_label_file(path.join(training_dir, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            mnist.read_image_file(path.join(training_dir, 't10k-images-idx3-ubyte')),
            mnist.read_label_file(path.join(training_dir, 't10k-labels-idx1-ubyte'))
        )
        os.makedirs(path.join(training_dir, mnist.MNIST.processed_folder))
        with open(path.join(training_dir, mnist.MNIST.processed_folder, mnist.MNIST.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(path.join(training_dir, mnist.MNIST.processed_folder, mnist.MNIST.test_file), 'wb') as f:
            torch.save(test_set, f)

        LOG.info('Dataset processing done!')

    # Training dataset
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root=training_dir, train=True, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])), batch_size=args.batch_size, shuffle=True, num_workers=4)
    # Test dataset
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root=training_dir, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])), batch_size=args.batch_size, shuffle=True, num_workers=4)

    model = Net().to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test()

    model_path = path.join(args.out_dir, 'model.pth')
    LOG.info('Saving model to %s...' % model_path)

    if not path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    save_checkpoint(model, model_path)

    if use_mlboard:
        mlboard.update_task_info({'checkpoint_path': model_path})

    LOG.info('Saved.')
