# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Regression problems involve predicting a continuous output variable based on input features. Traditional linear regression models often struggle with complex patterns in data. Neural networks, specifically feedforward neural networks, can capture these complex relationships by using multiple layers of neurons and activation functions. In this experiment, a neural network model is introduced with a single linear layer that learns the parameters weight and bias using gradient descent.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: Generate Dataset

Create input values  from 1 to 50 and add random noise to introduce variations in output values .

### STEP 2: Initialize the Neural Network Model

Define a simple linear regression model using torch.nn.Linear() and initialize weights and bias values randomly.

### STEP 3: Define Loss Function and Optimizer

Use Mean Squared Error (MSE) as the loss function and optimize using Stochastic Gradient Descent (SGD) with a learning rate of 0.001.

### STEP 4: Train the Model

Run the training process for 100 epochs, compute loss, update weights and bias using backpropagation.

### STEP 5: Plot the Loss Curve

Track the loss function values across epochs to visualize convergence.

### STEP 6: Visualize the Best-Fit Line

Plot the original dataset along with the learned linear model.

### STEP 7: Make Predictions

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: Sai Sanjiv R

### Register Number: 212223230179

```python
class Model(nn.Module):
      def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features,out_features)

    def forward(self, x):
        return self.linear(x)
```
# Initialize the Model, Loss Function, and Optimizer
```
torch.manual_seed(59)  # Ensure same initial weights
model = Model(1, 1)
initial_weight = model.linear.weight.item()
initial_bias = model.linear.bias.item()
print("\nName: Sai Sanjiv R")
print("Register No: 212223230179")
print(f'Initial Weight: {initial_weight:.8f}, Initial Bias: {initial_bias:.8f}\n')
```
# Define Loss Function & Optimizer
```
loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.001)
```
# Train the Model
```
epochs = 50
losses = []
for epoch in range(1, epochs + 1):  # Loop over epochs
    y_pred = model(X)
    loss = loss_function(y_pred,y)
    losses.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
      print(f'epoch: {epoch:2}  loss: {loss.item():10.8f}  '
          f'weight: {model.linear.weight.item():10.8f}  '
          f'bias: {model.linear.bias.item():10.8f}')
```
### Dataset Information
<img width="487" height="176" alt="image" src="https://github.com/user-attachments/assets/7daaf5f3-a71d-484b-bf38-e97391c6eca8" />
<img width="812" height="151" alt="image" src="https://github.com/user-attachments/assets/05760cbb-3630-4b0b-b3b3-72a78af17bc5" />
<img width="712" height="565" alt="image" src="https://github.com/user-attachments/assets/8a88a98a-7896-4593-92ee-d4da68f0d3ac" />


### OUTPUT
#### Training Loss Vs Iteration Plot
<img width="580" height="455" alt="download" src="https://github.com/user-attachments/assets/6839c997-292f-4a02-a1b7-df9e6e81f350" />

#### Best Fit line plot

<img width="711" height="565" alt="image" src="https://github.com/user-attachments/assets/01851a5d-e9cf-4bf7-bac5-4adf5a152ec1" />


### New Sample Data Prediction
<img width="805" height="257" alt="image" src="https://github.com/user-attachments/assets/1e9c818b-c44d-4c9f-8fc1-9d85cb57adb5" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
