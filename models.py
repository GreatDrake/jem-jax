from flax import linen as nn

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
    # Provide a constructor to register a new parameter 
    # and return its initial value
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    
        x = nn.Dropout(0.1)(x, deterministic=True)

        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    
        x = nn.Dropout(0.1)(x, deterministic=True)

        x = x.reshape((x.shape[0], -1)) # Flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)

        x = nn.Dropout(0.2)(x, deterministic=True)    

        x = nn.Dense(features=10)(x)    # There are 10 classes in MNIST
        return x

