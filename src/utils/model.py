import torch.nn as nn


class CNN1(nn.Module):
    def __init__(self, output_dim=2, activation='elu'):
        super(CNN1, self).__init__()

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        else:
            raise NotImplementedError(f"invalid activation function {activation}")

        self.bottom_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            self.activation,
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            self.activation,
            nn.MaxPool2d(2, 2),
            nn.Flatten(start_dim=1))
        self.penultimate_layer = nn.Linear(2304, 128)
        self.top_layer = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.bottom_layers(x)
        x = self.activation(self.penultimate_layer(x))
        x = self.top_layer(x)
        return x


class CNN2(nn.Module):
    def __init__(self, output_dim=2, activation='relu'):
        super(CNN2, self).__init__()

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        else:
            raise NotImplementedError(f"invalid activation function {activation}")
        self.bottom_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2, padding=1),
            self.activation,
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=1),
            self.activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=1),
            self.activation,
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2),
            nn.Flatten(start_dim=1))
        self.top_layer = nn.Linear(256, output_dim)

    def forward(self, x):
        x = self.bottom_layers(x)
        if hasattr(self, 'penultimate_layer'):  # some models e.g. logistic regression only has a top-layer
            x = self.activation(self.penultimate_layer(x))
        else:
            x = self.top_layer(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, activation='relu'):
        super(BasicBlock, self).__init__()

        if activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        else:
            self.activation = nn.ReLU()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential(
             nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
             nn.BatchNorm2d(self.expansion * planes)
        )

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class ResNet(nn.Module):
    # taken from https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
    # incorporated ELU setting to combat vanishing gradients https://arxiv.org/pdf/1604.04112.pdf
    def __init__(self, block, num_blocks, output_dim=10, activation='relu'):
        super(ResNet, self).__init__()
        self.in_planes = 16

        if activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        else:
            self.activation = nn.ReLU()

        self.bottom_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            self.activation,
            self._make_layer(block, 16, num_blocks[0], stride=1, activation=activation),
            self._make_layer(block, 32, num_blocks[1], stride=2, activation=activation),
            self._make_layer(block, 64, num_blocks[2], stride=2, activation=activation),
            nn.AvgPool2d((2, 2)),
            nn.Flatten(start_dim=1))
        self.top_layer = nn.Linear(1024, output_dim)

    def _make_layer(self, block, planes, num_blocks, stride, activation):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, activation=activation))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bottom_layers(x)
        x = self.top_layer(x)
        return x


def build_model(output_dim, model_class, activation='elu', device='cpu'):
    """
    Builds the model and initializes its weights.

    Parameters
    ----------
    output_dim : dimension of output layer / number of prediction classes
    model_class : name of model class; we use cnn1 for fashion_mnist, cnn2 for celeba, and resnet for cifar_10
    activation : activation function
    device : device on which to put the model

    Returns
    -------
    model : model used for training
    """
    if model_class == 'cnn1':
        model = CNN1(output_dim=output_dim, activation=activation).to(device)
    elif model_class == 'cnn2':
        model = CNN2(output_dim=output_dim, activation=activation).to(device)
    elif model_class == 'resnet':
        model = ResNet(BasicBlock, [3, 3, 3], output_dim=output_dim, activation=activation).to(device)
    else:
        raise NotImplementedError(f'invalid model name: {model_class}')

    if model_class == 'resnet':
        model.apply(init_weights_kaiming)
    else:
        model.apply(init_weights_glorot)
    return model


def init_weights_glorot(m, mean=0.0, std=0.02):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight)
    elif isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)


def init_weights_kaiming(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
