import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description="VGGNet parameters")
parser.add_argument("--num_layers", type=str, default="13",
                    help="available options: 13, 16, 19")
parser.add_argument("--data_root", type=str, default="./data/")
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--epoch", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.0002)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# VGGNet model definition
class VGGNet(nn.Module):
    def __init__(self,
                 num_layers,
                 num_classes=10):
        super(VGGNet, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers

        if self.num_layers == '13':
            self.layers = self._get_vgg13()
        elif self.num_layers == '16':
            self.layers = self._get_vgg16()
        elif self.num_layers == '19':
            self.layers = self._get_vgg19()
        else:
            ValueError(f"No attribute \'{self.num_layers}\'.")
        
        self.fc_layers= nn.Sequential(
            nn.Linear(512*1*1, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 1000),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1000, self.num_classes)
        )

    def _get_vgg13(self) -> nn.Sequential:
        layers = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        return layers

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


def fn_accuracy(pred, label):
    return (pred.max(dim=1)[1] == label).type(torch.float).mean()


if __name__ == "__main__":
    args = parser.parse_args()
    writer = SummaryWriter()
    
    CLASSES = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),]
    )
    dataset_train = datasets.CIFAR10(download=True,
                                     root=os.path.join(args.data_root, "cifar10"),
                                     train=True,
                                     transform=transform)
    dataset_test = datasets.CIFAR10(download=True,
                                    root=os.path.join(args.data_root, "cifar10"),
                                    train=False,
                                    transform=transform)
    
    # dataloader
    loader_train = DataLoader(dataset=dataset_train,
                              batch_size=args.batch_size,
                              shuffle=True)
    loader_test = DataLoader(dataset=dataset_test,
                              batch_size=128,
                              shuffle=False)
    num_batch = len(loader_train.dataset) // args.batch_size

    # model
    model = VGGNet(args.num_layers).to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss().to(device)

    best_acc = 0
    step = 0
    for epoch in range(1, args.epoch+1):
        model.train()

        loss_arr = []
        acc_arr = []

        for batch, (input, label) in enumerate(loader_train):
            input = input.to(device)
            label = label.to(device)

            output = model(input)
            pred = F.softmax(output, dim=1)
            acc = fn_accuracy(pred, label)
            
            optimizer.zero_grad()

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            loss_arr += [loss.item()]
            acc_arr += [acc.item()]

            if batch % 20 == 0:
                print('TRAIN: EPOCH %d/%d | BATCH: %d/%d | LOSS: %.4f | ACC: %.4f'
                      % (epoch, args.epoch, batch, num_batch, np.mean(loss_arr), np.mean(acc_arr)))
                step += batch
                writer.add_scalar("Loss/train", np.mean(loss_arr), step)
                writer.add_scalar("Accuracy/train", np.mean(acc_arr), step)

            
        with torch.no_grad():
            model.eval()

            loss_arr = []
            acc_arr = []

            for batch, (input, label) in enumerate(loader_test):
                input = input.to(device)
                label = label.to(device)

                output = model(input)
                pred = F.softmax(output, dim=1)

                acc = fn_accuracy(pred, label)
                loss = criterion(output, label)

                loss_arr += [loss.item()]
                acc_arr += [acc.item()]

            print('TEST: EPOCH %d/%d | LOSS: %.4f | ACC: %.4f'
                    % (epoch, args.epoch, np.mean(loss_arr), np.mean(acc_arr)))
            writer.add_scalar("Loss/test", np.mean(loss_arr), epoch)
            writer.add_scalar("Accuracy/test", np.mean(acc_arr), epoch)
        
        # save model
        save_dir = './VGG/weights'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        best_model_path = os.path.join(save_dir, "best_epoch.pth")

        # overridden best model
        if best_acc < np.mean(acc_arr):
            best_acc = np.mean(acc_arr)
            if os.path.exists(best_model_path):
                os.remove(best_model_path)
            print(f"Best Epoch {epoch} Overridden.")

        torch.save(model.state_dict(), os.path.join(save_dir, 'best_epoch.pth'))
