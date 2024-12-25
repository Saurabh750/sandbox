import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNN(nn.Module):
    def __init__(self, input_size: int, hidden_size1: int, hidden_size2: int, output_size: int):
        super(SimpleNN, self).__init__()
        
        # Defining the layers
        self.hidden_layer1 = nn.Linear(input_size, hidden_size1)  # First hidden layer
        self.hidden_layer2 = nn.Linear(hidden_size1, hidden_size2)  # Second hidden layer
        self.output_layer = nn.Linear(hidden_size2, output_size)  # Output layer
        
        # Activation functions
        self.relu = nn.ReLU()   # ReLU activation for the first hidden layer
        self.tanh = nn.Tanh()   # Tanh activation for the second hidden layer
    
    def forward(self, x):
        # Define the forward pass with the specified activation functions
        x = self.relu(self.hidden_layer1(x))  # Apply hidden layer 1 + ReLU
        x = self.tanh(self.hidden_layer2(x))  # Apply hidden layer 2 + Tanh
        x = self.output_layer(x)  # Output layer (no activation function here)
        return x

# Define the model with appropriate input size, hidden layer sizes, and output size
input_size = 784  # Example input size (e.g., for MNIST images: 28x28 flattened)
hidden_size1 = 128  # Number of neurons in the first hidden layer
hidden_size2 = 64   # Number of neurons in the second hidden layer
output_size = 10    # Output size (e.g., for 10 classes in MNIST)

# Create the model
model = SimpleNN(input_size, hidden_size1, hidden_size2, output_size)

# Print the model architecture
print(model)
