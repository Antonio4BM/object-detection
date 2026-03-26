from torchvision import models
import torch.nn as nn


# (kernel size, out channels, strides, padding)

architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class YoloV1(nn.Module):

    def __init__(self, S=7, B=2, C=7, pretrained=True, freeze_backbone=True, **kwargs):
        super(YoloV1, self).__init__()
        #self.in_channels = in_channels
        #self.architecture = architecture_config

        #self.darknet = self._create_conv_layers(self.architecture)
        #self.fcs = self._create_fcs(**kwargs)
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)

        # remove avgpool and fc
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.head = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d((S,S)),
            nn.Flatten(),
            nn.Linear(1024 * S * S, 1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(1024, S * S * (B * 5 + C))
        )
        

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layers += [CNNBlock(in_channels, out_channels=x[1], kernel_size=x[0], stride=x[2], padding=x[3])]
                in_channels = x[1]
            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif type(x) == list:
                conv_1 = x[0]
                conv_2 = x[1]
                n = x[2]
                for _ in range(n):
                    layers += [CNNBlock(in_channels, out_channels=conv_1[1], kernel_size=conv_1[0], stride=conv_1[2], padding=conv_1[3])]
                    layers += [CNNBlock(conv_1[1], out_channels=conv_2[1], kernel_size=conv_2[0], stride=conv_2[2], padding=conv_2[3])]
                    in_channels = conv_2[1]
                    
        return nn.Sequential(*layers)


    def _create_fcs(self, split_size, num_boxes, num_classes):

        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Linear(1024 * S * S, 4096),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S * S * (B * 5 + C))
        )