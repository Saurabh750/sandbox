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

    # def activation_function(self, weighted_sum: float, activ_func: str) -> float:
    #     if type == "sigmoid":
    #         return 1 / (1 + math.exp(-weighted_sum))
    #     elif type == "relu":
    #         return max(0, weighted_sum) 
    #     elif type == "leakyrelu":
    #         return max(0.01 * weighted_sum, weighted_sum)         # hardcoding alpha as 0.01 for simplicity
    #     elif type == "tanh":
    #         return (math.exp(weighted_sum) - math.exp(-weighted_sum)) / (math.exp(weighted_sum) + math.exp(-weighted_sum))
    #     else:
    #         return f"No such activation function defined: {type}"
        
class simpleNeuralNetwork():
    def __init__(self) -> None:
        np.random.seed(1)
        # For simplicity, first trying to model a single neuron with 3 inputs and 1 output
        # Random weights assigned to 3 x 1 matrix
        self.weights = 2 * np.random.random((3, 1)) - 1

    def __sigmoid(self, z: float):
        return 1 / (1 + math.exp(-z))
    
    def __sigmoid_derivative(self, a: float):
        return a * (1 - a)
    
    def forwardPass(self, x, count):
        print(f"Forward Pass: {count+1}")

        prod = np.dot(x, self.weights)
        for i in range(len(prod)):
            prod[i] = self.__sigmoid(prod[i])

        return prod
    
    # For adjusting the weights during backprop, I have used "Error Weighted Derivative" for simplicity

    def train(self, input_training_data, output_training_data, epochs):
        for epoch in range(epochs):
            output = self.forwardPass(input_training_data, epoch)

            error = np.zeros(output.shape)
            for i in range(len(output)):
                error[i][0] = output_training_data.T[i][0] - output[i][0]

            print(error)

            delta = np.dot(input_training_data.T, error * self.__sigmoid_derivative(output))

            self.weights += delta

if __name__ == "__main__":
    nn = simpleNeuralNetwork()
    print("Random weights are:")
    print(nn.weights)

    input = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    output = np.array([0,1,1,0]).reshape(-1,1).T

    nn.train(input, output, 100)
