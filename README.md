# Backpropagation from Scratch - Regression

This project demonstrates how to implement a simple neural network with backpropagation for a regression task. The implementation is completely from scratch using NumPy and Pandas.

## Dataset

The dataset used in this project is a simple tabular dataset with three columns:
- CGPA (Cumulative Grade Point Average)
- ATS Score (Aptitude Test Score)
- Package(LPA) (Salary package in Lakhs Per Annum)

This is a small dataset with only 4 rows, making it perfect for understanding the inner workings of neural networks.

## Project Overview

This implementation builds a neural network with:
- Input layer: 2 neurons (for CGPA and ATS Score)
- Hidden layer: 2 neurons
- Output layer: 1 neuron (predicted Package)

The implementation includes:
1. Parameter initialization
2. Forward propagation
3. Manual backpropagation update
4. Training over multiple epochs

## Neural Network Architecture

The neural network implemented in this project has the following structure:

- **Input Layer**: 2 neurons (CGPA and ATS Score)
- **Hidden Layer**: 2 neurons with linear activation
- **Output Layer**: 1 neuron (Package LPA prediction)

### Architecture Diagram

```
[Input Layer]       [Hidden Layer]       [Output Layer]
                        ┌─────┐
                    ────┤ H1  ├────┐
  ┌─────┐          │    └─────┘    │     ┌─────┐
  │CGPA ├──────────┤               ├─────┤     │
  └─────┘          │    ┌─────┐    │     │ LPA │
                    ────┤ H2  ├────┘     │     │
  ┌─────┐          │    └─────┘          └─────┘
  │ATS  ├──────────┘
  └─────┘      
```

### Parameter Details

- **W1**: Weights connecting input layer to hidden layer (shape: 2×2)
- **b1**: Biases for hidden layer neurons (shape: 2×1)
- **W2**: Weights connecting hidden layer to output layer (shape: 2×1)
- **b2**: Bias for output layer neuron (shape: 1×1)

## How the Network Works

1. **Parameter Initialization**: Weights are initialized to small values (0.1) and biases to zeros.
2. **Forward Propagation**: 
   - Input features X = [CGPA, ATS Score] are fed into the network
   - Hidden layer computation: A1 = W1.T @ X + b1
   - Output layer computation: y_hat = W2.T @ A1 + b2
3. **Parameter Update**: The weights and biases are updated based on the prediction error using gradient descent.
4. **Epochs**: The entire dataset is processed multiple times to improve the model's accuracy.

## Loss Function

The implementation uses Mean Squared Error (MSE) as the loss function:
- Loss = (y - y_hat)²
- Learning rate = 0.001

## Key Functions

- `initialize_parameters`: Sets up the initial weights and biases for each layer
- `linear_forward`: Performs the linear transformation part of forward propagation
- `L_layer_forward`: Implements the complete forward pass through all layers
- `update_parameters`: Updates weights and biases using gradient descent

## Instructions

To run this implementation:

1. Set up a Jupyter notebook or Google Colab environment
2. Import the required libraries (NumPy and Pandas)
3. Create the dataset as shown in the code
4. Run the code blocks sequentially to see the step-by-step execution
5. Observe how the loss decreases with each epoch as the model learns

## Implementation Notes

- The backpropagation is implemented manually (not using automatic differentiation)
- The learning rate is fixed at 0.001
- No activation functions are used in this simple implementation
- The model is trained for 5 epochs in the example

## Extensions

This basic implementation can be extended in several ways:
- Add activation functions (like ReLU, sigmoid, etc.)
- Implement mini-batch gradient descent
- Add regularization to prevent overfitting
- Increase the number of hidden layers
- Increase the dataset size

## Learning Outcomes

By studying this implementation, you can gain insights into:
- The fundamentals of neural networks
- How parameter updates work in backpropagation
- The relationship between forward propagation and backpropagation
- How neural networks learn from data through iterative training
