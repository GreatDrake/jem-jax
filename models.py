from functools import partial
from typing import Any, Callable, Sequence, Tuple

from flax import linen as nn
import jax.numpy as jnp

ModuleDef = Any
Init = nn.initializers.variance_scaling(1/3, "fan_in", "uniform")
Conv = partial(nn.Conv, use_bias=True, kernel_init=Init, bias_init=nn.initializers.zeros)

class WideBlock(nn.Module):
    filters: int
    conv: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)
    strides_helper = jnp.zeros(strides)

    @nn.compact
    def __call__(self, x,):
        residual = x
        x = self.act(x)
        x = self.conv(self.filters, (3, 3), padding=[(1,1), (1,1)])(x)
        x = self.act(x)
        x = self.conv(self.filters, (3, 3), self.strides, padding=[(1,1), (1,1)])(x)

        if residual.shape != x.shape or self.strides_helper.shape != (1, 1):
            residual = self.conv(self.filters, (1, 1),
                                 self.strides, name='conv_proj')(residual)

        return residual + x

class WideResNet(nn.Module):
    num_classes: int
    depth: int
    widen_factor: int
    act: Callable = partial(nn.leaky_relu, negative_slope=0.2)
    block_cls: ModuleDef = WideBlock

    @nn.compact
    def __call__(self, x):
        conv = Conv

        n = (self.depth-4)//6
        k = self.widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        x = conv(nStages[0], (3, 3), (1, 1), padding=[(1,1), (1,1)])(x)    
    
        def _wide_layer(x, planes, num_blocks, stride):
            strides = [stride] + [(1, 1)]*(num_blocks-1)
            for stride in strides:
                x = self.block_cls(filters=planes, strides=stride, act=self.act, conv=conv)(x)
            return x

        x = _wide_layer(x, nStages[1], n, (1, 1))
        x = _wide_layer(x, nStages[2], n, (2, 2))
        x = _wide_layer(x, nStages[3], n, (2, 2))  
    
        x = self.act(x)

        x = nn.avg_pool(x, window_shape=(8, 8))
        x = x.reshape((x.shape[0], -1))

        x = nn.Dense(self.num_classes, kernel_init=Init)(x)
    
        return x

class CNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.leaky_relu(x, 0.05)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    
        x = Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.leaky_relu(x, 0.05)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    
        x = x.reshape((x.shape[0], -1)) # Flatten
        x = nn.Dense(features=256, kernel_init=Init)(x)
        x = nn.leaky_relu(x, 0.05)


        x = nn.Dense(features=10, kernel_init=Init)(x)    # There are 10 classes in MNIST
        return x

