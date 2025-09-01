from dataclasses import dataclass
from typing import List, Tuple
import torch.nn as nn
import torch
from ptflops import get_model_complexity_info

@dataclass
class FireConfig:
    """SqueezeNet Fire模块配置类"""
    squeeze_channels: int    # Squeeze层输出通道数
    expand_1x1_channels: int # Expand层1x1卷积通道数
    expand_3x3_channels: int # Expand层3x3卷积通道数
    use_bypass: bool = False # 是否使用残差连接

    @property
    def expand_channels(self) -> int:
        return self.expand_1x1_channels + self.expand_3x3_channels


def _squeezenet_conf(
        version: str,
        width_mult: float = 1.0,  # 宽度乘数（用于模型缩放）
        **kwargs
) -> Tuple[List[FireConfig], int]:
    """
    生成SqueezeNet配置的核心函数

    Args:
        version: 模型版本标识符 ('1_0', '1_1', 'custom')
        width_mult: 通道数缩放系数 (默认1.0)

    Returns:
        fire_configs: Fire模块配置列表
        init_channels: 初始卷积层通道数
    """

    # 基础配置（基于SqueezeNet 1.0）
    base_config = [
        FireConfig(16, 64, 64),
        FireConfig(16, 64, 64),
        FireConfig(32, 128, 128),
        FireConfig(32, 128, 128),
        FireConfig(48, 192, 192),
        FireConfig(48, 192, 192),
        FireConfig(64, 256, 256)
    ]

    # 版本适配
    if version == '1_0':
        fire_configs = base_config
        init_channels = 96
    elif version == '1_1':
        fire_configs = [
            FireConfig(16, 64, 64),
            FireConfig(16, 64, 64),
            FireConfig(32, 128, 128),
            FireConfig(32, 128, 128),
            FireConfig(48, 192, 192),
            FireConfig(48, 192, 192),
            FireConfig(64, 256, 256, use_bypass=True)  # 1.1版本增加残差连接
        ]
        init_channels = 64  # 1.1版本初始通道更小
    elif version == 'custom':
        # 自定义配置示例
        fire_configs = [
            FireConfig(int(16 * width_mult),
                       int(64 * width_mult),
                       int(64 * width_mult)),
            # ...其他自定义层
        ]
        init_channels = int(96 * width_mult)
    else:
        raise ValueError(f"Unsupported SqueezeNet version: {version}")

    # 应用宽度乘数
    if width_mult != 1.0:
        fire_configs = [
            FireConfig(
                int(cfg.squeeze_channels * width_mult),
                int(cfg.expand_1x1_channels * width_mult),
                int(cfg.expand_3x3_channels * width_mult),
                cfg.use_bypass
            ) for cfg in fire_configs
        ]
        init_channels = int(init_channels * width_mult)

    return fire_configs, init_channels


class SqueezeNet1(nn.Module):
    def __init__(
            self,
            version: str = "1_0",
            num_classes: int = 1000,
            width_mult: float = 1.0,
            dropout: float = 0.5
    ):
        super().__init__()
        # 初始卷积（输入通道必须为3）
        self.initial_conv = nn.Conv2d(3, 96, kernel_size=7, stride=2)

        # 特征层
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireModule(in_channels=96, squeeze_channels=16, expand_1x1_channels=64, expand_3x3_channels=64),
            FireModule(in_channels=128, squeeze_channels=16, expand_1x1_channels=64, expand_3x3_channels=64),
            nn.MaxPool2d(3, stride=2, ceil_mode=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # 27 →13
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),  # 13 →7
            nn.ReLU(inplace=True)
        )


        # self.features = nn.Sequential(
        #     # nn.Conv2d(3, 96, kernel_size=7, stride=2),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        #     FireModule(96, 16, 64, 64),
        #     FireModule(128, 16, 64, 64),
        #     FireModule(128, 32, 128, 128),
        #     nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        #     FireModule(256, 32, 128, 128),
        #     FireModule(256, 48, 192, 192),
        #     FireModule(384, 48, 192, 192),
        #     FireModule(384, 64, 256, 256),
        #     nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        #     FireModule(512, 64, 256, 256),
        # )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(128, num_classes, kernel_size=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )



    def forward(self, x):
        conv_out = self.initial_conv(x)
        features_out = self.features(conv_out)
        cls_out = self.classifier(features_out)
        return conv_out, features_out, cls_out

    def _make_fire_layers(
            self,
            configs: List[FireConfig],
            initial_out_channels: int  # 新增参数
    ) -> nn.Sequential:
        layers = []
        in_channels = initial_out_channels  # 使用传入的初始通道数

        for i, cfg in enumerate(configs):
            layers.append(
                FireModule(
                    in_channels=in_channels,
                    squeeze_channels=cfg.squeeze_channels,
                    expand_1x1_channels=cfg.expand_1x1_channels,
                    expand_3x3_channels=cfg.expand_3x3_channels,
                    use_bypass=cfg.use_bypass
                )
            )
            in_channels = cfg.expand_channels

            if i in [2, 4]:
                layers.append(nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))

        return nn.Sequential(*layers)


class FireModule(nn.Module):
    def __init__(
            self,
            in_channels: int,
            squeeze_channels: int,
            expand_1x1_channels: int,
            expand_3x3_channels: int,
            use_bypass: bool = False
    ):
        super().__init__()
        self.use_bypass = use_bypass

        # Squeeze层
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)

        # Expand层
        self.expand_1x1 = nn.Conv2d(squeeze_channels, expand_1x1_channels, kernel_size=1)
        self.expand_3x3 = nn.Conv2d(squeeze_channels, expand_3x3_channels, kernel_size=3, padding=1)
        self.expand_activation = nn.ReLU(inplace=True)

        # 残差连接
        if use_bypass:
            self.bypass = nn.Conv2d(in_channels, expand_1x1_channels + expand_3x3_channels, kernel_size=1)

    def forward(self, x):
        identity = x

        x = self.squeeze_activation(self.squeeze(x))
        x = torch.cat([
            self.expand_activation(self.expand_1x1(x)),
            self.expand_activation(self.expand_3x3(x))
        ], 1)

        if self.use_bypass:
            x += self.bypass(identity)

        return x

def squeezenet1_0(pretrained: bool = False, **kwargs) -> SqueezeNet1:
    model = SqueezeNet1(version='1_0', **kwargs)
    if pretrained:
        load_pretrained_weights(model, 'squeezenet1_0')
    return model

def squeezenet1_1(pretrained: bool = False, **kwargs) -> SqueezeNet1:
    model = SqueezeNet1(version='1_1', **kwargs)
    if pretrained:
        load_pretrained_weights(model, 'squeezenet1_1')
    return model

def squeezenet_custom(width_mult: float = 1.0, **kwargs) -> SqueezeNet1:
    return SqueezeNet1(version='custom', width_mult=width_mult, **kwargs)
def count_parameters(model):
    # 计算输入为224x224x3时的FLOPs和参数数量
    with torch.cuda.device(0):  # 如果有GPU可用，选择GPU计算
        flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True)

    return flops,params
