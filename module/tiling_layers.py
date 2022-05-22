__all__ = ['ExtractTiles', 'MergeTiles']

from typing import Tuple
from torch import nn
import torch


class ExtractTiles(nn.Module):
    def __init__(self, tile_sz: Tuple[int], flatten: bool):
        super().__init__()
        self.tile_sz = tile_sz
        self.flatten = flatten

    def forward(self, inputs: torch.Tensor):
        input_shape = inputs.get_shape()
        tile_sz = self.tile_sz

        # NCHW -> N C GH TH GW TW
        patches_grid_shape_a = [
            input_shape[0],
            input_shape[1],
            input_shape[2] // tile_sz[0],
            tile_sz[0],
            input_shape[3] // tile_sz[1],
            tile_sz[1]]
        print(
            f"{inputs.name} ExtractTiles input_shape={input_shape} tile_sz={tile_sz} "
            f"patches_grid_shape_a={patches_grid_shape_a}")
        patches = inputs
        patches = torch.reshape(patches, patches_grid_shape_a)
        # N C GH TH GW TW -> N GH GW C TH TW
        patches = torch.transpose(patches, (0, 2, 4, 1, 3, 5))

        if self.flatten:
            # N GH GW C TH TW -> N*GH*GW C TH TW
            patches = torch.reshape(patches, [-1, tile_sz[0], tile_sz[1], input_shape[-1]])

        return patches


class MergeTiles(nn.Module):
    def __init__(self, grid_sz: Tuple[int]):
        super().__init__()
        self.grid_sz = grid_sz

    def call(self, inputs: torch.Tensor):
        grid_sz = self.grid_sz
        input_shape = inputs.get_shape()
        patches = inputs

        # N*GH*GW C TH TW -> N GH GW C TH TW
        patches_grid_shape = [-1, grid_sz[0], grid_sz[1], input_shape[1], input_shape[-2], input_shape[-1]]
        merged = torch.reshape(patches, patches_grid_shape)

        # N GH GW C TH TW -> N C GH TH GW TW
        merged = torch.transpose(merged, (0, 3, 1, 4, 2, 5))

        # N C GH TH GW TW -> N C H W
        output_shape = [-1, input_shape[1], grid_sz[0] * input_shape[-2], grid_sz[1] * input_shape[-1]]
        merged = torch.reshape(merged, output_shape)
        return merged