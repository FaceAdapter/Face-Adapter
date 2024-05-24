import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from huggingface_hub import PyTorchModelHubMixin
from transformers.modeling_utils import PreTrainedModel
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
# import model_resnet

INPLACE_RELU=True


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=INPLACE_RELU)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=INPLACE_RELU)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out




class UNet(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, fea_dims=[64,64,128,256,512], out_dims = [16,32,64,128,256], num_classes=1):
        super(UNet, self).__init__()

        self.backbone = ResNetBackBone(BasicBlock, [2, 2, 2, 2])
        

        self.decoder1 = nn.Sequential(nn.Conv2d(fea_dims[4]+fea_dims[3], out_dims[4], 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(out_dims[4]),
                                      nn.ReLU(INPLACE_RELU),
                                      nn.Conv2d(out_dims[4], out_dims[4], 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(out_dims[4]),
                                      nn.ReLU(INPLACE_RELU))

        self.decoder2 = nn.Sequential(nn.Conv2d(fea_dims[3]+fea_dims[2], out_dims[3], 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(out_dims[3]),
                                      nn.ReLU(INPLACE_RELU),
                                      nn.Conv2d(out_dims[3], out_dims[3], 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(out_dims[3]),
                                      nn.ReLU(INPLACE_RELU))

        self.decoder3 = nn.Sequential(nn.Conv2d(fea_dims[2]+fea_dims[1], out_dims[2], 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(out_dims[2]),
                                      nn.ReLU(INPLACE_RELU),
                                      nn.Conv2d(out_dims[2], out_dims[2], 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(out_dims[2]),
                                      nn.ReLU(INPLACE_RELU))

        self.decoder4 = nn.Sequential(nn.Conv2d(fea_dims[1]+fea_dims[0], out_dims[1], 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(out_dims[1]),
                                      nn.ReLU(INPLACE_RELU),
                                      nn.Conv2d(out_dims[1], out_dims[1], 3, 1, 1,bias=False),
                                      nn.BatchNorm2d(out_dims[1]),
                                      nn.ReLU(INPLACE_RELU))

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')


        self.decoder5 = nn.Sequential(nn.Conv2d(out_dims[1], out_dims[0], 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(out_dims[0]),
                                      nn.ReLU(INPLACE_RELU),
                                      nn.Conv2d(out_dims[0], out_dims[0], 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(out_dims[0]),
                                      nn.ReLU(INPLACE_RELU))
        self.final_conv = nn.Conv2d(out_dims[0], num_classes, 1, 1, bias=True)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # model_path = './pretrained/resnet18-5c106cde.pth'
        # model_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        # checkpoint_no_module = {}
        # for k, v in model_dict.items():            
        #     checkpoint_no_module[k] = v
        # info = self.backbone.load_state_dict(checkpoint_no_module, strict=False)
        # print(info)

    def forward(self, img):
        x0,x1,x2,x3,x4 = self.backbone(img)
        x4 = self.upsample(x4)
        x3 = self.decoder1(torch.cat([x3,x4],dim = 1))
        x3 = self.upsample(x3)
        x2 = self.decoder2(torch.cat([x2,x3],dim = 1))
        x2 = self.upsample(x2)
        x1 = self.decoder3(torch.cat([x1,x2],dim = 1))
        x1 = self.upsample(x1)
        x0 = self.decoder4(torch.cat([x0,x1],dim = 1))
        x0 = self.upsample(x0)
        x0 = self.decoder5(x0)
        out = self.final_conv(x0)
        return out.sigmoid()


class ResNetBackBone(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNetBackBone, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=INPLACE_RELU)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])  #1/4
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  #1/8
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  #1/16
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  #1/32

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):


        x = self.conv1(x)
        x = self.bn1(x)
        x0 = self.relu(x)
        
        x = self.maxpool(x0)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return [x0,x1,x2,x3,x4]


if __name__ =='__main__':
    model = UNet()
    img = torch.zeros((1,6,256,256))
    res = model(img)
    print(res.size())
    
    torch.save(model.state_dict(), 'seg_unet_res18.pth', _use_new_zipfile_serialization=False)
