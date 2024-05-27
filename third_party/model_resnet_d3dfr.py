import os
import torch.nn as nn
from torch.nn import Linear
from torch.nn import Conv2d
from torch.nn import BatchNorm1d
from torch.nn import BatchNorm2d
from torch.nn import ReLU
from torch.nn import Dropout
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from torch.nn import MaxPool2d
from torch.nn import Sequential
from torch.nn import Module
import torch
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
}

def filter_state_dict(state_dict, remove_name='fc'):
    new_state_dict = {}
    for key in state_dict:
        if remove_name in key:
            continue
        new_state_dict[key] = state_dict[key]
    return new_state_dict

def conv3x3(in_planes, out_planes, stride=1):
    """ 3x3 convolution with padding
    """
    return Conv2d(in_planes,
                  out_planes,
                  kernel_size=3,
                  stride=stride,
                  padding=1,
                  bias=False)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """ 1x1 convolution
    """
    return Conv2d(in_planes,
                  out_planes,
                  kernel_size=1,
                  stride=stride,
                  bias=bias)

def conv3x3_(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1_(in_planes: int, out_planes: int, stride: int = 1, bias: bool = False) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


class Bottleneck(Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = BatchNorm2d(planes * self.expansion)
        self.relu = ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck_(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck_, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1_(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3_(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1_(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(Module):
    """ ResNet backbone
    """
    def __init__(self, input_size, block, layers, zero_init_residual=True):
        """ Args:
            input_size: input_size of backbone
            block: block function
            layers: layers in each block
        """
        super(ResNet, self).__init__()
        assert input_size[0] in [112, 224], \
            "input_size should be [112, 112] or [224, 224]"
        self.inplanes = 64
        self.conv1 = Conv2d(3, 64,
                            kernel_size=7,
                            stride=2,
                            padding=3,
                            bias=False)
        self.bn1 = BatchNorm2d(64)
        self.relu = ReLU(inplace=True)
        self.maxpool = MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.bn_o1 = BatchNorm2d(2048)
        self.dropout = Dropout()
        if input_size[0] == 112:
            self.fc = Linear(2048 * 4 * 4, 512)
        else:
            self.fc = Linear(2048 * 7 * 7, 512)
        self.bn_o2 = BatchNorm1d(512)

        # initialize_weights(self.modules)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn_o1(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn_o2(x)

        return x


class resNet(nn.Module): # ori resnet

    def __init__(
            self,
            block_: Type[Union[BasicBlock, Bottleneck_]],
            layers: List[int],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            use_last_fc: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(resNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.use_last_fc = use_last_fc
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block_, 64, layers[0])
        self.layer2 = self._make_layer(block_, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block_, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block_, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if self.use_last_fc:
            self.fc = nn.Linear(512 * block_.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck_):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block_: Type[Union[BasicBlock, Bottleneck_]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block_.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block_.expansion, stride),
                norm_layer(planes * block_.expansion),
            )

        layers = []
        layers.append(block_(self.inplanes, planes, stride, downsample, self.groups,
                             self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block_.expansion
        for _ in range(1, blocks):
            layers.append(block_(self.inplanes, planes, groups=self.groups,
                                 base_width=self.base_width, dilation=self.dilation,
                                 norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        if self.use_last_fc:
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

def ResNet_50(input_size, **kwargs):
    """ Constructs a ResNet-50 model.
    """
    model = ResNet(input_size, Bottleneck, [3, 4, 6, 3], **kwargs)

    return model


class ResNet50_nofc(Module):
    """ ResNet backbone
    """
    def __init__(self, input_size, output_dim, use_last_fc=False, init_path=None):
        """ Args:
            input_size: input_size of backbone
            block: block function
            layers: layers in each block
        """
        super(ResNet50_nofc, self).__init__()
        assert input_size[0] in [112, 224, 256], \
            "input_size should be [112, 112] or [224, 224]"
        func, last_dim = func_dict['resnet50']
        self.use_last_fc=use_last_fc
        backbone = func(use_last_fc=use_last_fc, num_classes=output_dim)
        if init_path and os.path.isfile(init_path):
            state_dict = filter_state_dict(torch.load(init_path, map_location='cpu'))
            backbone.load_state_dict(state_dict)
            print("Loading init recon %s from %s"%('resnet50', init_path))
        self.backbone = backbone
        if not use_last_fc:
            self.fianl_layers = nn.ModuleList([
                conv1x1(last_dim, 80, bias=True), # id
                conv1x1(last_dim, 64, bias=True), # exp
                conv1x1(last_dim, 80, bias=True), # tex
                conv1x1(last_dim, 3, bias=True), # angle
                conv1x1(last_dim, 27, bias=True), # gamma
                conv1x1(last_dim, 2, bias=True), # tx, ty
                conv1x1(last_dim, 1, bias=True), # tz
                conv1x1(last_dim, 4, bias=True) # pupil
            ])
            for m in self.fianl_layers:
                nn.init.constant_(m.weight, 0.)
                nn.init.constant_(m.bias, 0.)


    def forward(self, x):
        x = self.backbone(x)
        if not self.use_last_fc:
            output = []
            for layer in self.fianl_layers:
                output.append(layer(x))
            x = torch.flatten(torch.cat(output, dim=1), 1)
        return x


def _resnet(
        arch: str,
        block: Type[Union[BasicBlock, Bottleneck_]],
        layers: List[int],
        pretrained: bool,
        progress: bool,
        **kwargs: Any
) -> ResNet:
    model = resNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> resNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck_, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


func_dict = {
    'resnet50': (resnet50, 2048),
}


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
    

def fuse(conv, bn):
    w = conv.weight
    mean = bn.running_mean
    var_sqrt = torch.sqrt(bn.running_var + bn.eps)

    beta = bn.weight
    gamma = bn.bias

    if conv.bias is not None:
        b = conv.bias
    else:
        b = mean.new_zeros(mean.shape)

    w = w * (beta / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
    b = (b - mean) / var_sqrt * beta + gamma

    fused_conv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        conv.dilation,
        conv.groups,
        bias=True,
        padding_mode=conv.padding_mode
    )
    fused_conv.weight = nn.Parameter(w)
    fused_conv.bias = nn.Parameter(b)
    return fused_conv


def fuse_module(m):
    children = list(m.named_children())
    conv = None
    conv_name = None
    for name, child in children:
        if isinstance(child, nn.BatchNorm2d) and conv:
            bc = fuse(conv, child)
            m._modules[conv_name] = bc
            m._modules[name] = Identity()
            conv = None
        elif isinstance(child, nn.Conv2d):
            conv = child
            conv_name = name
        else:
            fuse_module(child)


def getd3dfr_res50(pretrained="./third_party/d3dfr_res50_nofc.pth"):
    model = ResNet50_nofc([256,256], 257+4, use_last_fc=False)
    for param in model.parameters():
        param.requires_grad=False
    if pretrained is not None and os.path.exists(pretrained):
        checkpoint_no_module = {}
        checkpoint = torch.load(pretrained, map_location=lambda storage, loc: storage)
        for k, v in checkpoint.items():
            if k.startswith('module'):
                k = k[7:]
            checkpoint_no_module[k] = v
        info = model.load_state_dict(checkpoint_no_module, strict=False)
        print(pretrained, info)
    model = model.eval()
    fuse_module(model)
    return model
