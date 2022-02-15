"""
Code adapted from https://github.com/facebookresearch/GradientEpisodicMemory
                    &
                  https://github.com/kuangliu/pytorch-cifar
"""
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.functional import relu, avg_pool2d

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out

class BasicBlockFilm(nn.Module):
    """
    Extension to standard ResNet block (https://arxiv.org/abs/1512.03385) with FiLM layer adaptation. After every batch
    normalization layer, we add a FiLM layer (which applies an affine transformation to each channel in the hidden
    representation). As we are adapting the feature extractor with an external adaptation network, we expect parameters
    to be passed as an argument of the forward pass.
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockFilm, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, gamma1, beta1, gamma2, beta2):
        """
        Implements a forward pass through the FiLM adapted ResNet block. FiLM parameters for adaptation are passed
        through to the method, one gamma / beta set for each convolutional layer in the block (2 for the blocks we are
        working with).
        :param x: (torch.tensor) Batch of images to apply computation to.
        :param gamma1: (torch.tensor) Multiplicative FiLM parameter for first conv layer (one for each channel).
        :param beta1: (torch.tensor) Additive FiLM parameter for first conv layer (one for each channel).
        :param gamma2: (torch.tensor) Multiplicative FiLM parameter for second conv layer (one for each channel).
        :param beta2: (torch.tensor) Additive FiLM parameter for second conv layer (one for each channel).
        :return: (torch.tensor) Resulting representation after passing through layer.
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self._film(out, gamma1, beta1)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self._film(out, gamma2, beta2)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    def _film(self, x, gamma, beta):
        gamma = gamma[None, :, None, None]
        beta = beta[None, :, None, None]
        return gamma * x + beta


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf, bias):
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.linear = nn.Linear(nf * 8 * block.expansion, num_classes, bias=bias)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def features(self, x):
        '''Features before FC layers'''
        out = relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

    def logits(self, x):
        '''Apply the last FC linear mapping to get logits'''
        x = self.linear(x)
        return x

    def forward(self, x):
        out = self.features(x)
        logits = self.logits(out)
        return logits
    
    def get_layer_output(self, x, param_dict, layer_to_return):
        if layer_to_return == 0:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            if self.initial_pool:
                x = self.maxpool(x)
            return x
        else:
            resnet_layers = [self.layer1, self.layer2, self.layer3, self.layer4]
            layer = layer_to_return - 1
            for block in range(self.layers[layer]):
                x = resnet_layers[layer][block](x, param_dict[layer][block]['gamma1'], param_dict[layer][block]['beta1'],
                                       param_dict[layer][block]['gamma2'], param_dict[layer][block]['beta2'])
            return x

    @property
    def output_size(self):
        return 512


class FilmResNet(ResNet):
    """
    Wrapper object around BasicBlockFilm that constructs a complete ResNet with FiLM layer adaptation. Inherits from
    ResNet object, and works with identical logic.
    """

    def __init__(self, block, layers):
        ResNet.__init__(self, block, layers)
        self.layers = layers

    def forward(self, x, param_dict):
        """
        Forward pass through ResNet. Same logic as standard ResNet, but expects a dictionary of FiLM parameters to be
        provided (by adaptation network objects).
        :param x: (torch.tensor) Batch of images to pass through ResNet.
        :param param_dict: (list::dict::torch.tensor) One dictionary for each block in each layer of the ResNet,
                           containing the FiLM adaptation parameters for each conv layer in the model.
        :return: (torch.tensor) Feature representation after passing through adapted network.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.initial_pool:
            x = self.maxpool(x)

        for block in range(self.layers[0]):
            x = self.layer1[block](x, param_dict[0][block]['gamma1'], param_dict[0][block]['beta1'],
                                   param_dict[0][block]['gamma2'], param_dict[0][block]['beta2'])
        for block in range(self.layers[1]):
            x = self.layer2[block](x, param_dict[1][block]['gamma1'], param_dict[1][block]['beta1'],
                                   param_dict[1][block]['gamma2'], param_dict[1][block]['beta2'])
        for block in range(self.layers[2]):
            x = self.layer3[block](x, param_dict[2][block]['gamma1'], param_dict[2][block]['beta1'],
                                   param_dict[2][block]['gamma2'], param_dict[2][block]['beta2'])
        for block in range(self.layers[3]):
            x = self.layer4[block](x, param_dict[3][block]['gamma1'], param_dict[3][block]['beta1'],
                                   param_dict[3][block]['gamma2'], param_dict[3][block]['beta2'])

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x


def film_resnet18(pretrained=False, pretrained_model_path=None, mt=False, **kwargs):
    """
        Constructs a FiLM adapted ResNet-18 model.
    """

    # if mt flag is set (mini/tiered imagenet scenario)
    if mt:
        return film_resnet18_mt(pretrained, pretrained_model_path, **kwargs)

    model = FilmResNet(BasicBlockFilm, [2, 2, 2, 2], **kwargs)
    if pretrained:
        ckpt_dict = torch.load(pretrained_model_path)
        model.load_state_dict(ckpt_dict['state_dict'])
    return model



def resnet18_mt(pretrained=False, pretrained_model_path=None, **kwargs):
    """
        Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        ckpt_dict = torch.load(pretrained_model_path)
        model.load_state_dict(ckpt_dict)
    return model


def film_resnet18_mt(pretrained=False, pretrained_model_path=None, **kwargs):
    """
        Constructs a FiLM adapted ResNet-18 model.
    """
    model = FilmResNet(BasicBlockFilm, [2, 2, 2, 2], **kwargs)
    model_state = model.state_dict()
    if pretrained:
        ckpt_dict = torch.load(pretrained_model_path)
        model.load_state_dict(ckpt_dict['resnet_dict'])
    return model


def Reduced_ResNet18(nclasses, nf=20, bias=True):
    """
    Reduced ResNet18 as in GEM MIR(note that nf=20).
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf, bias)

def ResNet18(nclasses, nf=64, bias=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf, bias)

'''
See https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
'''

def ResNet34(nclasses, nf=64, bias=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], nclasses, nf, bias)

def ResNet50(nclasses, nf=64, bias=True):
    return ResNet(Bottleneck, [3, 4, 6, 3], nclasses, nf, bias)


def ResNet101(nclasses, nf=64, bias=True):
    return ResNet(Bottleneck, [3, 4, 23, 3], nclasses, nf, bias)


def ResNet152(nclasses, nf=64, bias=True):
    return ResNet(Bottleneck, [3, 8, 36, 3], nclasses, nf, bias)


class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, dim_in=160, head='mlp', feat_dim=128):
        super(SupConResNet, self).__init__()
        self.encoder = Reduced_ResNet18(100)
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        elif head == 'None':
            self.head = None
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder.features(x)
        if self.head:
            feat = F.normalize(self.head(feat), dim=1)
        else:
            feat = F.normalize(feat, dim=1)
        return feat

    def features(self, x):
        return self.encoder.features(x)
