import math
from random import randint
import numpy as np

class myNeuralNetLayer:
    def __init__(self, node_count: int) -> None:
        self.node_count: int = node_count
        self.weights: np.ndarray = []

        for i in range(node_count):
            self.weights.append(randint(0,6))

class myNeuralNetwork:
    def __init__(self, inp_nodes: int, out_nodes: int, activ_func: str) -> None:
        self.inp = myNeuralNetLayer(inp_nodes)
        self.out = myNeuralNetLayer(out_nodes)

    def weightedSum(self, x: np.ndarray) -> float:
        weights_transpose = self.out.weights.T
        return np.dot(x, weights_transpose)

    def activation_function(self, weighted_sum: float, activ_func: str) -> float:
        if type == "sigmoid":
            return 1 / (1 + math.exp(-weighted_sum))
        elif type == "relu":
            return max(0, weighted_sum) 
        elif type == "leakyrelu":
            return max(0.01 * weighted_sum, weighted_sum)         # hardcoding alpha as 0.01 for simplicity
        elif type == "tanh":
            return (math.exp(weighted_sum) - math.exp(-weighted_sum)) / (math.exp(weighted_sum) + math.exp(-weighted_sum))
        else:
            return f"No such activation function defined: {type}"

    





###################################
#
# Number of neurons, 
# Loss function, optimizer
#
###################################

def forward_pass(inputs: np.ndarray, weights: np.ndarray, bias: float, activ_func: str):
    z = np.dot(inputs, weights) + bias
    return activation_function(activ_func, z)

def activation_function(type: str, y: float):       # only supports sigmoid, tanh, relu, leakyrelu
    if type == "sigmoid":
        return 1 / (1 + math.exp(-y))
    elif type == "relu":
        return max(0, y) 
    elif type == "leakyrelu":
        return max(0.01 * y, y)         # hardcoding alpha as 0.01 for simplicity
    elif type == "tanh":
        return (math.exp(y) - math.exp(-y)) / (math.exp(y) + math.exp(-y))
    else:
        return f"No such activation function defined: {type}"
    
def backward_pass():
    pass

print(activation_function("tanh", 5))