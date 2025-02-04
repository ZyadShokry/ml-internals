import numpy as np
import uuid

### Parameter class

class Parameter:
    def __init__(self, value):
        """
        Initializes a Parameter object to wrap around a value (like weight or gradient).
        
        Args:
            value (np.ndarray): The value of the parameter (e.g., weight or gradient).
        """
        self.value = value
        self._grad = np.zeros_like(value)  # Initialize gradient to zeros
        self._id = uuid.uuid4()  # Unique identifier for the parameter

    def set_grad(self, grad):
        self._grad = grad

    @property
    def grad(self):
        return self._grad
    
    @property
    def id(self):
        return self._id

    def fill(self, value):
        """Fill the parameter with a scalar value."""
        self.value.fill(value)
    
    def zero_grad(self):
        """Reset the gradient to zero."""
        self.grad.fill(0)

    def __repr__(self):
        return f"Parameter(value={self.value.shape}, grad={self.grad.shape})"


### Base Optimizer class

class Optimizer:
    def __init__(self, params, lr=0.001):
        self.lr = lr
        self.params = params

    def zero_grad(self):
        """
        Reset the gradients of all parameters.
        """
        for param in self.params:
            param.zero_grad()

    def step(self):
        """
        Update the parameters based on the computed gradients.
        """
        raise NotImplementedError("This method should be implemented by a subclass.")


### SGD Optimizer (Stochastic Gradient Descent)

class SGDOptimizer(Optimizer):
    def __init__(self, params=None, lr=0.001, momentum=0.0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.prev_v = {}  # Previous velocity for each parameter

    def step(self):
        for param in self.params:
            if param.id not in self.prev_v:
                self.prev_v[param.id] = np.zeros_like(param.value)

            # Update velocity
            self.prev_v[param.id] = self.momentum * self.prev_v[param.id] - self.lr * param.grad

            # Update parameter
            param.value += self.prev_v[param.id]
