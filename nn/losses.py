# -*- coding: utf-8 -*-
import numpy as np

### Cross Entropy Loss Function

class CrossEntropy:
    def __init__(self):
        self.y = None # y: predicted output (model's output)
        self.t = None # t: true target (correct labels)

    def __call__(self, y, t):
        self.y = y
        self.t = t
        return self.forward(y, t)

    def stable_softmax(self, x):
        """
        Computes the softmax of vector x with numerical stability (by subtracting the max value).
        
        Arguments:
            x: Input array (logits).
        
        Returns:
            softmax_result: Softmax probabilities for each class.
        """
        # Subtract max for numerical stability to prevent overflow issues in exp()
        x = x - np.max(x, axis=1, keepdims=True)
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    
    def softmax(self, x):
        """
        Softmax function placeholder (not implemented).
        In case needed for other usages, this method can be implemented.
        """
        pass

    def forward(self, y, t):
        """
        Computes the forward pass for cross-entropy loss calculation.
        
        Arguments:
            y: Predicted output (logits) from the model.
            t: Ground truth target labels.
        
        Returns:
            loss: The computed cross-entropy loss.
        """
        m = y.shape[0]  # Number of samples
        prob = self.stable_softmax(y)
        log_likelihood = -np.log(prob[range(m), t])
        loss = np.sum(log_likelihood)
        return loss

    def backward(self):
        """
        Computes the gradient of the loss with respect to the inputs (backpropagation).
        
        Returns:
            grad: The gradient of the cross-entropy loss with respect to the input logits.
        """
        m = self.t.shape[0]  # Number of samples
        prob = self.stable_softmax(self.y)
        prob[range(m), self.t] -= 1
        return prob
