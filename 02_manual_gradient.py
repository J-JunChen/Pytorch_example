import numpy as np
import matplotlib.pyplot as plt

# Training Data
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0  # a random guess: random value


# our model for forward pass
def forward(x):
    return x * w


# Loss function
def loss(x, y):
    y_pred = forward(x)
    return pow((y_pred - y), 2)


# compute gradient
def gradient(x, y):  # d_loss/d_w
    return 2 * x * (x * w - y)


#Before training
print("Prediction (before training)", 4, forward(4))
 
# Training loop
for epoch in range(20):
    for x_val, y_val in zip(x_data, y_data):
        # Compute derivative w.r.t to the learning weights
        # update the weights
        # Compute the loss and print progress
        grad = gradient(x_val, y_val)
        w = w - 0.01 * grad  # set the manual learing rate: 0.01
        print("\tgrad: ", x_val, y_val, round(grad, 2))
        l = loss(x_val, y_val)
    print("progress:", epoch, "w=", round(w, 2), "loss=", round(l, 2))

# After training
print("Predicted score (after training)", "4 hours of studying: ", forward(4))
