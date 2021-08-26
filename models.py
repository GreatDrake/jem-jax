from functools import partial
from typing import Any, Callable, Sequence, Tuple

from flax import linen as nn
import jax.numpy as jnp

ModuleDef = Any

class WideBlock(nn.Module):
    filters: int
    conv: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x,):
        residual = x
        y = self.act(x)
        y = self.conv(self.filters, (3, 3), padding=[(1,1), (1,1)])(y)
        y = self.act(x)
        y = self.conv(self.filters, (3, 3), self.strides, padding=[(1,1), (1,1)])(y)

        #if residual.shape != y.shape:
        residual = self.conv(self.filters, (1, 1),
                             self.strides, name='conv_proj')(residual)

        return residual + y

class WideResNet(nn.Module):
    num_classes: int
    depth: int
    widen_factor: int
    act: Callable = partial(nn.leaky_relu, negative_slope=0.2)
    block_cls: ModuleDef = WideBlock

    @nn.compact
    def __call__(self, x):
        conv = partial(nn.Conv, use_bias=True)

        n = (self.depth-4)//6
        k = self.widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        out = conv(nStages[0], (3, 3), (1, 1), padding=[(1,1), (1,1)])(x)    
    
        def _wide_layer(x, planes, num_blocks, stride):
            strides = [stride] + [(1, 1)]*(num_blocks-1)
            for stride in strides:
                x = self.block_cls(filters=planes, strides=stride, act=self.act, conv=conv)(x)
            return x

        out = _wide_layer(out, nStages[1], n, (1, 1))
        out = _wide_layer(out, nStages[2], n, (2, 2))
        out = _wide_layer(out, nStages[3], n, (3, 3))  
    
        out = self.act(out)

        #out = nn.avg_pool(out, window_shape=(8, 8))
        out = jnp.mean(out, axis=(1, 2))    

        out = nn.Dense(self.num_classes)(out)
    
        return out


class CNN2(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=16, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2))
        #x = nn.Dropout(0.1)(x, deterministic=True)
        
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2))
        #x = nn.Dropout(0.1)(x, deterministic=True)

        
        x = x.reshape((x.shape[0], -1)) # Flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)

        x = nn.Dropout(0.2)(x, deterministic=True)    

        x = nn.Dense(features=10)(x)    # There are 10 classes in MNIST
        return x

class CNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.leaky_relu(x, 0.05)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    
        x = nn.Dropout(0.1)(x, deterministic=True)

        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.leaky_relu(x, 0.05)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    
        x = nn.Dropout(0.1)(x, deterministic=True)

        x = x.reshape((x.shape[0], -1)) # Flatten
        x = nn.Dense(features=256)(x)
        x = nn.leaky_relu(x, 0.05)

        x = nn.Dropout(0.2)(x, deterministic=True)    

        x = nn.Dense(features=10)(x)    # There are 10 classes in MNIST
        return x

class CNNSimple(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    
        x = nn.Dropout(0.1)(x, deterministic=True)

        x = x.reshape((x.shape[0], -1)) # Flatten
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)

        x = nn.Dropout(0.2)(x, deterministic=True)    

        x = nn.Dense(features=10)(x)    # There are 10 classes in MNIST
        return x

