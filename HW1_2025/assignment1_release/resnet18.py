'''ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt

class BasicBlock(nn.Module):
    
    def __init__(self, in_planes, planes, stride=1):
        """
            :param in_planes: input channels
            :param planes: output channels
            :param stride: The stride of first conv
        """
        super(BasicBlock, self).__init__()
        # Uncomment the following lines, replace the ? with correct values.
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        # 1. Go through conv1, bn1, relu
        # 2. Go through conv2, bn
        # 3. Combine with shortcut output, and go through relu
        output = F.relu(self.bn1(self.conv1(x)))
        output2 = self.bn2(self.conv2(output))
        return F.relu(output2+self.shortcut(x))


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        # Uncomment the following lines and replace the ? with correct values
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, in_planes, planes, stride):
        layers = []
        layers.append(BasicBlock(in_planes, planes, stride))
        layers.append(BasicBlock(planes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, images):
        """ input images and output logits """
        relu_output = F.relu(self.bn1(self.conv1(images)))
        layers_output = self.layer4(self.layer3(self.layer2(self.layer1(relu_output))))
        pooling_output = F.avg_pool2d(layers_output, 4) # pooling
        flattened_ouptut = torch.flatten(pooling_output, 1)
        output_logits = self.linear(flattened_ouptut)
        return output_logits

    def visualize(self, logdir):
        """ Visualize the kernel in the desired directory """
        if not os.path.exists(logdir):
            os.makedirs(logdir)

        # Visualize filters from the first convolutional layer (conv1)
        self._visualize_filters(self.conv1, "conv1_filters", logdir)

        # Visualize filters from each BasicBlock (layer1, layer2, layer3, layer4)
        self._visualize_filters(self.layer1[0].conv1, "layer1_conv1_filters", logdir)
        self._visualize_filters(self.layer2[0].conv1, "layer2_conv1_filters", logdir)
        self._visualize_filters(self.layer3[0].conv1, "layer3_conv1_filters", logdir)
        self._visualize_filters(self.layer4[0].conv1, "layer4_conv1_filters", logdir)

    def _visualize_filters(self, layer, name, logdir):
        """ Helper function to visualize filters of a specific layer """
        filters = layer.weight.data.clone()  # Extract filters from the layer
        filters = filters - filters.min()  # Normalize to [0, 1]
        filters = filters / filters.max()

        # Create a grid of filters (e.g., 8x8 grid for visualization)
        num_filters = filters.size(0)  # Number of filters in the layer
        filter_size = filters.size(2)  # Size of each filter (3x3 for the first layer)
        grid_size = int(np.ceil(np.sqrt(num_filters)))  # Define grid size based on the number of filters

        # Create an empty grid for the filters
        grid = np.zeros((grid_size * filter_size, grid_size * filter_size))

        # Place filters into the grid
        for i in range(grid_size):
            for j in range(grid_size):
                filter_idx = i * grid_size + j
                if filter_idx < num_filters:
                    # Get the filter and convert to numpy array
                    kernel = filters[filter_idx].cpu().numpy().transpose(1, 2, 0)  # Convert from CxHxW to HxWxC
                    kernel = np.mean(kernel, axis=-1)  # Convert to grayscale (average over channels)
                    kernel_resized = np.resize(kernel, (filter_size, filter_size))  # Resize to match grid size
                    grid[i * filter_size:(i + 1) * filter_size, j * filter_size:(j + 1) * filter_size] = kernel_resized

        # Plot and save the grid of filters as an image
        plt.imshow(grid, cmap='gray')
        plt.axis('off')
        plt.savefig(os.path.join(logdir, f'{name}.png'))  # Save the image in the logdir
        plt.close()

        print(f"Saved filter visualization at {logdir}/{name}.png")
