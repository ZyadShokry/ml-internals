import numpy as np

### Sigmoid Activation Function

class Sigmoid:
    def __init__(self):
        # y: output
        self.y = None

    def forward(self, x):
        self.y = 1.0 / (1.0 + np.exp(-x))
        return self.y

    def backward(self, dy):
        # dy: gradient from the upper layer
        return dy * self.y * (1.0 - self.y)

    def __str__(self):
        return 'Sigmoid'

### ReLU Activation Function

class ReLU:
    def __init__(self):
        # x: input
        self.x = None

    def forward(self, x):
        self.x = x
        return np.maximum(0.0, x)

    def backward(self, dy):
        # dy: gradient from the upper layer
        return dy * (self.x > 0.0)

    def __str__(self):
        return 'ReLU'
