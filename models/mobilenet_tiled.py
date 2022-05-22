# -*- coding: utf-8 -*-
# @Author  : DevinYang(pistonyang@gmail.com)
# @Author  : RobertBiehl(robeffect@gmail.com)
__all__ = ['MobileNetV2Tiled']

from typing import List, Tuple

from torchtoolbox.nn import Activation
from torch import nn

from models.mobilenet import SE_Module, MobileNetBottleneck
from module.tiling_layers import ExtractTiles, MergeTiles


class MobileNetBottleneckTiled(nn.Module):
    def __init__(self, in_c, expansion, out_c, kernel_size, stride, se=False,
                 activation='relu6', first_conv=True, skip=True, linear=True,
                 input_szs: List[Tuple[int]] = [],
                 extract_tiles: bool = False,
                 merge_tiles: bool = False
                 ):
        super(MobileNetBottleneckTiled, self).__init__()

        assert merge_tiles is False or stride == 1, "Tiles cannot be merged if stride is != 1"
        assert extract_tiles is False or stride == 1, "Tiles cannot be extracted if stride is != 1"

        self.act = Activation(activation, auto_optimize=True)  # [bug]no use when linear=True
        hidden_c = round(in_c * expansion)
        self.linear = linear
        self.skip = stride == 1 and in_c == out_c and skip

        # prepare tile_sz and grid_sz
        tile_sz = [16, 16] if (extract_tiles or merge_tiles) and len(input_szs) > 0 else None

        if tile_sz is not None:
            shape = input_szs[-1]
            while tile_sz[0] > 1 and shape[1] % tile_sz[0] != 0:
                tile_sz[0] //= 2
            while tile_sz[1] > 1 and shape[2] % tile_sz[1] != 0:
                tile_sz[1] //= 2

        grid_sz = (shape[1] // tile_sz[0], shape[2] // tile_sz[0]) if merge_tiles else None

        if grid_sz is not None and (grid_sz[0] <= 1 or grid_sz[1] <= 1 or tile_sz[0] <= 4 or tile_sz[1] <= 4):
            tile_sz = None
            grid_sz = None

        seq = []
        if extract_tiles and tile_sz is not None:
            seq.append(ExtractTiles(tile_sz, flatten=True))
            print(f"MobileNetBottleneckTiled ExtractTiles tile_sz={tile_sz}")

        if first_conv and in_c != hidden_c:
            seq.append(nn.Conv2d(in_c, hidden_c, 1, 1, bias=False))
            seq.append(nn.BatchNorm2d(hidden_c))
            seq.append(Activation(activation, auto_optimize=True))

        seq.append(nn.Conv2d(hidden_c, hidden_c, kernel_size, stride,
                             kernel_size // 2, groups=hidden_c, bias=False))
        seq.append(nn.BatchNorm2d(hidden_c))
        seq.append(Activation(activation, auto_optimize=True))
        if se:
            seq.append(SE_Module(hidden_c))
        seq.append(nn.Conv2d(hidden_c, out_c, 1, 1, bias=False))
        seq.append(nn.BatchNorm2d(out_c))

        if merge_tiles and grid_sz is not None:
            seq.append(MergeTiles(grid_sz))
            print(f"MobileNetBottleneckTiled MergeTiles grid_sz={tile_sz}")

        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        skip = x
        x = self.seq(x)
        if self.skip:
            x = skip + x
        if not self.linear:
            x = self.act(x)
        return x


class MobileNetV2Tiled(nn.Module):
    def __init__(self, num_classes=1000, small_input=False, input_sz: Tuple[int] = (224, 224)):
        super(MobileNetV2Tiled, self).__init__()
        self.first_block = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2 if not small_input else 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            nn.Conv2d(32, 16, 1, 1, bias=False),
            nn.BatchNorm2d(16),
        )

        input_szs = [input_sz]

        self.mb_block = nn.Sequential(
            MobileNetBottleneck(16, 6, 24, 3, 2, input_szs=input_szs),
            MobileNetBottleneckTiled(24, 6, 24, 3, 1, input_szs=input_szs, extract_tiles=True, merge_tiles=True),
            MobileNetBottleneck(24, 6, 32, 3, 2, input_szs=input_szs),
            MobileNetBottleneckTiled(32, 6, 32, 3, 1, input_szs=input_szs, extract_tiles=True),
            MobileNetBottleneckTiled(32, 6, 32, 3, 1, input_szs=input_szs, merge_tiles=True),
            MobileNetBottleneck(32, 6, 64, 3, 2, input_szs=input_szs),
            MobileNetBottleneckTiled(64, 6, 64, 3, 1, input_szs=input_szs, extract_tiles=True),
            MobileNetBottleneck(64, 6, 64, 3, 1, input_szs=input_szs),
            MobileNetBottleneck(64, 6, 64, 3, 1, input_szs=input_szs),
            MobileNetBottleneck(64, 6, 96, 3, 1, input_szs=input_szs),
            MobileNetBottleneck(96, 6, 96, 3, 1, input_szs=input_szs),
            MobileNetBottleneckTiled(96, 6, 96, 3, 1, input_szs=input_szs, merge_tiles=True),
            MobileNetBottleneck(96, 6, 160, 3, 2, input_szs=input_szs),
            MobileNetBottleneckTiled(160, 6, 160, 3, 1, input_szs=input_szs),  # TODO: extract tiles here again
            MobileNetBottleneck(160, 6, 160, 3, 1, input_szs=input_szs),
            MobileNetBottleneck(160, 6, 320, 3, 1, input_szs=input_szs),
        )
        self.last_block = nn.Sequential(
            nn.Conv2d(320, 1280, 1, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.output = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.first_block(x)
        x = self.mb_block(x)
        x = self.last_block(x)
        x = self.output(x)
        return x
