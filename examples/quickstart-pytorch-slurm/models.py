
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision. models import resnet18

class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self, in_channels: int, conv1_channels: int) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, conv1_channels, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(conv1_channels, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ResNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super(ResNet, self).__init__()

        self.model = resnet18(num_classes=num_classes)

        # make adjustements so the model is more suitable for 32x32 or 28x28 inputs
        self.model.conv1 = nn.Conv2d(in_channels, self.model.conv1.out_channels, kernel_size=3, stride=1)
        self.model.maxpool = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
