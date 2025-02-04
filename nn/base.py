from .layers import Dense, Conv2D
from collections import OrderedDict
import prettytable as pt

class Model:
    def __init__(self):
        self.layers = OrderedDict()
        self.loss = None
        self.params = []

    def add(self, layer):
        layer_name = layer.__class__.__name__ + str(len(self.layers))
        self.layers[layer_name] = layer

        if hasattr(layer, 'params'):
            for param in layer.params:
                self.params.append(param)

    def init_weights(self, value):
        for layer in self.layers.values():
            if isinstance(layer, Dense) or isinstance(layer, Conv2D):
                layer.init_weights(value)
                layer.init_biases(value)

    def set_loss(self, loss):
        self.loss = loss

    def __call__(self, x):
        return self.predict(x)

    def predict(self, x):
        for layer in self.layers.values():
            x = layer(x)
        return x

    def backward(self):
        dy = self.loss.backward()
        for layer in reversed(self.layers.values()):
            dy = layer.backward(dy)
            
    @property
    def summary(self):
        table = pt.PrettyTable()
        table.field_names = ['Layer', 'Type', 'Input', 'Output', 'Activation']
        for name, layer in self.layers.items():
            _type, _input, _output, _activation = 'N/A', 'N/A', 'N/A', 'N/A'
            if 'Type' in layer.details:
                _type = layer.details['Type']
            if 'Input' in layer.details:
                _input = layer.details['Input']
            if 'Output' in layer.details:
                _output = layer.details['Output']
            if 'Activation' in layer.details:
                _activation = layer.details['Activation']
            
            table.add_row([name, _type, _input, _output, _activation])
        return table