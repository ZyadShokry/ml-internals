import numpy as np
from .optimizers import Parameter

### Dense (Fully Connected) Layer

class Dense:
    def __init__(self, input_size, output_size, activation=None):
        """
        Initializes the Dense layer (fully connected layer).

        Args:
            input_size (int): Number of input features (neurons).
            output_size (int): Number of output neurons.
            activation (callable, optional): Activation function to apply after the linear transformation.
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.W = Parameter(np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size))
        self.b = Parameter(np.zeros(output_size))
        self.x = None # Placeholder for input data
        self.y = None # Placeholder for output data
        self.params = [self.W, self.b]
        self.info()

    def init_weights(self, value):
        self.W.fill(value)

    def init_biases(self, value):
        self.b.fill(value)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Performs the forward pass of the Dense layer.

        Args:
            x (numpy.ndarray): Input data with shape (batch_size, input_size).

        Returns:
            numpy.ndarray: Output after linear transformation (Wx + b) and activation (if provided).
        """
        self.batch_size = x.shape[0]
        self.x = x
        W, b = self.W.value, self.b.value
        self.y = np.dot(x, W) + b
        if self.activation is not None:
            self.y = self.activation.forward(self.y)
        return self.y

    def backward(self, dy):
        """
        Performs the backward pass to compute gradients with respect to input, weights, and biases.

        Args:
            dy (numpy.ndarray): Gradient of the loss with respect to the output.

        Returns:
            numpy.ndarray: Gradient with respect to the input data.
        """
        if self.activation is not None:
            dy = self.activation.backward(dy)
        self.W.set_grad(np.dot(self.x.T, dy))
        self.b.set_grad(np.sum(dy, axis=0))
        dx = np.dot(dy, self.W.value.T)
        return dx

    def info(self):
        self.details = {
            'Type': 'Dense',
            'Input': self.input_size,
            'Output': self.output_size,
            'Activation': self.activation
        }

### 2D Convolutional Layer

class Conv2D:
    def __init__(self, input_size=(3), kernel_size=(3, 3), num_kernels=1, padding=(1, 1), stride=(1, 1), activation=None):
        """
        Initializes the Conv2D layer.

        Args:
            input_size (tuple): The size of the input depth.
            kernel_size (tuple): The size of the convolutional filter (height, width).
            num_kernels (int): Number of kernels (filters) to apply.
            padding (tuple): Padding to be added to input (height, width).
            stride (tuple): Stride of the convolution (height, width).
            activation (callable, optional): Activation function to be applied after convolution.
        """
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels
        self.padding = padding
        self.stride = stride
        self.activation = activation
        self.W = Parameter(np.random.randn(num_kernels, input_size[0], kernel_size[0], kernel_size[1]) * np.sqrt(2.0 / (input_size[0] * kernel_size[0] * kernel_size[1])))
        self.b = Parameter(np.zeros(num_kernels))
        self.params = [self.W, self.b]
        self.x = None # Placeholder for input data
        self.y = None # Placeholder for output data
        # self.optim = None
        self.info()

    def init_weights(self, value):
        self.W.fill(value)

    def init_biases(self, value):
        self.b.fill(value)

    def __call__(self, x):
        return self.forward(x)

    @staticmethod
    def dilate(image, di_h=1, di_v=1):
        """
        Dilates the input image (used for backward pass to account for stride).

        Args:
            image (numpy.ndarray): Input image to be dilated.
            di_h (int): Horizontal dilation factor.
            di_v (int): Vertical dilation factor.

        Returns:
            numpy.ndarray: Dilated image.
        """
        dilated_image = np.zeros((image.shape[0] * (di_v + 1) - di_v, image.shape[1] * (di_h + 1) - di_h))
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                dilated_image[i * (di_v + 1), j * (di_h + 1)] = image[i, j]
        return dilated_image

    @staticmethod
    def convolve(image, kernel, padding=(1, 1), stride=(1, 1)):
        """
        Performs 2D convolution on the input image using the given kernel.

        Args:
            image (numpy.ndarray): Input image.
            kernel (numpy.ndarray): Convolution kernel (filter).
            padding (tuple): Padding to apply to the input image (height, width).
            stride (tuple): Stride for moving the kernel across the image.

        Returns:
            numpy.ndarray: Convolved image.
        """
        padded_image = np.zeros((image.shape[0] + 2 * padding[0], image.shape[1] + 2 * padding[1]))
        if padding[0] == 0 and padding[1] == 0:
            padded_image = image
        else:
            padded_image[padding[0]:-padding[0], padding[1]:-padding[1]] = image

        convolved_image = np.zeros((int((padded_image.shape[0] - kernel.shape[0]) / stride[0] + 1),
                                    int((padded_image.shape[1] - kernel.shape[1]) / stride[1] + 1)))
        for i in range(convolved_image.shape[0]):
            for j in range(convolved_image.shape[1]):
                convolved_image[i, j] = np.sum(padded_image[i * stride[0]:i * stride[0] + kernel.shape[0],
                                                            j * stride[1]:j * stride[1] + kernel.shape[1]] * kernel)
        return convolved_image

    def forward(self, x):
        """
        Performs the forward pass of the convolutional layer.

        Args:
            x (numpy.ndarray): Input data with shape (batch_size, channels, height, width).

        Returns:
            numpy.ndarray: Output of the convolution after activation.
        """
        self.x = x
        self.y = np.zeros((x.shape[0], self.num_kernels, int((x.shape[2] + 2 * self.padding[0] - self.kernel_size[0]) / self.stride[0] + 1),
                           int((x.shape[3] + 2 * self.padding[1] - self.kernel_size[1]) / self.stride[1] + 1)))

        bias, W = self.b.value, self.W.value
        for b in range(x.shape[0]):
            for i in range(self.num_kernels):
                for j in range(x.shape[1]):
                    self.y[b, i] += self.convolve(x[b, j], W[i, j], padding=self.padding, stride=self.stride)
                self.y[b, i] += bias[i]

        if self.activation is not None:
            self.y = self.activation.forward(self.y)

        return self.y

    def backward(self, dy):
        """
        Performs the backward pass to calculate gradients with respect to input and weights.

        Args:
            dy (numpy.ndarray): Gradient of the loss with respect to the output of this layer.

        Returns:
            numpy.ndarray: Gradient with respect to the input.
        """
        if self.activation is not None:
            dy = self.activation.backward(dy)

        dx = np.zeros(self.x.shape)
        W = self.W.value
        for b in range(self.x.shape[0]):
            for i in range(self.x.shape[1]):
                for j in range(self.num_kernels):
                    dx[b, i] += self.convolve(self.dilate(dy[b, j], self.stride[1] - 1, self.stride[0] - 1),
                                         np.rot90(W[j, i], 2),
                                         padding=(self.kernel_size[1] - 1, self.kernel_size[0] - 1))

        dW = np.zeros(W.shape)
        
        for b in range(self.x.shape[0]):
            for i in range(self.num_kernels):
                for j in range(self.x.shape[1]):
                    dW[i, j] += self.convolve(self.x[b, j],
                                                self.dilate(dy[b, i], self.stride[1] - 1,
                                                self.stride[0] - 1),
                                                padding=self.padding)
        self.W.set_grad(dW)
        self.b.set_grad(np.sum(dy, axis=(0, 2, 3)))
        return dx

    def info(self):
        self.details = {
            'Type': 'Conv2D',
            'Input': self.input_size,
            'Activation': self.activation
        }

### Maximum Pooling Layer

class MaxPool:
    def __init__(self, pool_size=(2, 2), stride=(2, 2), activation=None):
        """
        Initializes the MaxPool layer.

        Args:
            pool_size (tuple): Size of the pooling window, typically (height, width).
            stride (tuple): Stride for moving the pooling window, typically (height, width).
            activation (callable, optional): Activation function to be applied after pooling.
        """
        self.pool_size = pool_size
        self.stride = stride
        self.x = None # Placeholder for input data
        self.y = None # Placeholder for output data
        self.activation = activation
        self.info()

    def __call__(self, x):
        self.x = x
        self.y = np.zeros((x.shape[0], x.shape[1], x.shape[2] // self.pool_size[0], x.shape[3] // self.pool_size[1]))
        if self.activation is not None:
            return self.activation.forward(self.forward(x))
        return self.forward(x)

    def forward(self, x):
        """
        Performs the forward pass of the MaxPool layer, applying max pooling.

        Args:
            x (numpy.ndarray): Input array with shape (batch_size, channels, height, width).

        Returns:
            numpy.ndarray: Output of the max pooling operation.
        """
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for k in range(x.shape[2] // self.pool_size[0]):
                    for l in range(x.shape[3] // self.pool_size[1]):
                        self.y[i, j, k, l] = np.max(x[i, j, k * self.stride[0]:k * self.stride[0] + self.pool_size[0],
                                                    l * self.stride[1]:l * self.stride[1] + self.pool_size[1]])
        return self.y

    def backward(self, dy):
        """
        Performs the backward pass to compute gradients with respect to input.

        Args:
            dy (numpy.ndarray): Gradient of the loss with respect to the output of this layer.

        Returns:
            numpy.ndarray: Gradient with respect to the input.
        """
        if self.activation is not None:
            dy = self.activation.backward(dy)

        dx = np.zeros_like(self.x)
        for i in range(self.x.shape[0]):
            for j in range(self.x.shape[1]):
                for k in range(self.x.shape[2] // self.pool_size[0]):
                    for l in range(self.x.shape[3] // self.pool_size[1]):
                        max_index = np.argmax(self.x[i, j, k * self.stride[0]:k * self.stride[0] + self.pool_size[0],
                                                    l * self.stride[1]:l * self.stride[1] + self.pool_size[1]])
                        max_row = max_index // self.pool_size[1]
                        max_col = max_index % self.pool_size[1]
                        dx[i, j, k * self.stride[0] + max_row, l * self.stride[1] + max_col] = dy[i, j, k, l]
        return dx

    def info(self):
        self.details = {
            'Type': 'MaxPool',
            'Pool': self.pool_size,
            'Stride': self.stride
        }

### Flatten Layer

class Flatten:
    def __init__(self):
        self.x = None # Placeholder for input data
        self.y = None # Placeholder for output data
        self.info()

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Forward pass for flattening the input.

        This reshapes the input x (batch_size, channels, height, width) into a 2D array
        with shape (batch_size, -1), where -1 means that the dimension is inferred
        based on the other dimensions. Essentially, it flattens all spatial dimensions
        into a single vector per sample.

        Args:
            x (numpy.ndarray): Input array with shape (batch_size, channels, height, width).

        Returns:
            numpy.ndarray: Flattened array with shape (batch_size, -1).
        """
        self.x = x
        self.y = x.reshape(x.shape[0], -1)
        return self.y

    def backward(self, dy):
        """
        Backward pass for flattening the input.

        Since the forward pass simply reshapes the input, the backward pass does the
        same operation to match the original shape of x.

        Args:
            dy (numpy.ndarray): Gradient with respect to the output of the layer.

        Returns:
            numpy.ndarray: Gradient with respect to the input, reshaped back to the original input shape.
        """
        return dy.reshape(self.x.shape)

    def info(self):
        self.details = {
            'Type': 'Flatten'
        }