import numpy as np
import trax   
from trax import layers as tl

from config import *
from process import Process

class Model:
    def __init__(self):
        pass


    def reversible_layer_forward(self, x, f, g):
        """
        Args:
            x (np.array): an input vector or matrix
            f (function): a function which operates on a vector/matrix
            g (function): a function which operates on a vector/matrix
        Returns:
            y (np.array): an output vector or matrix whose form is determined by 'x', f and g
        """
        # split the input vector into two (* along the last axis because it is the depth dimension)
        x1, x2 = np.split(x, 2, axis=-1)

        y1 = x1 + f(x2)

        y2 = x2 + g(y1)

        # concatenate y1 and y2 along the depth dimension. be sure output is of type np.ndarray
        y = np.concatenate((y1, y2), axis=-1)

        return y



    def reversible_layer_reverse(self, y, f, g):
        """
        Args:
            y (np.array): an input vector or matrix
            f (function): a function which operates on a vector/matrix of the form of 'y'
            g (function): a function which operates on a vector/matrix of the form of 'y'
        Returns:
            y (np.array): an output vector or matrix whose form is determined by 'y', f and g
        """

        # split the input vector into two (* along the last axis because it is the depth dimension)
        y1, y2 = np.split(y, 2, axis=-1)
        x2 = y2 - g(y1)
        x1 = y1 - f(x2)

        # concatenate x1 and x2 along the depth dimension
        x = np.concatenate((x1, x2), axis=-1)

        return x


    def ReformerLM(vocab_size=33000, n_layers=2, mode='train', attention_type=tl.SelfAttention):
        """
        Args:
            vocab_size (int): size of the vocabulary
            n_layers (int): number of decoder layers
            mode (string): setting of the model which can be 'train', 'eval', or 'predict'
            attention_type(class): attention class to use
        Returns:
            model (ReformerLM): a reformer language model implemented in Trax
        """

        # initialize an instance of Trax's ReformerLM class
        model = trax.models.reformer.ReformerLM(
            # set vocab size
            vocab_size=vocab_size,
            # set number of layers
            n_layers=n_layers,
            # set mode
            mode=mode,
            # set attention type
            attention_type=attention_type
        )

        return model

