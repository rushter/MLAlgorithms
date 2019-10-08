
# We'll be using the contour plot to visualize descent on advertising dataset

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Note - we will perform the gradient descent on the advertising data


data = pd.read_csv("Advertising.csv")

data = data.drop(columns=["Unnamed: 0"], axis=1)

Y = data["Sales"]

X = np.column_stack((data["TV"],data["Radio"]))

# converting the values into normal form i.e., mean = 0 and S.D. = 1
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# Calculating Gradient and Mean Squared Error(MSE)

def gradient_descent(W, x, y):
    y_hat = x.dot(W).flatten()
    error = (y - y_hat)
    mse = (1.0 / len(x)) * np.sum(np.square(error))
    gradient = -(1.0 / len(x)) * error.dot(x)
    return gradient, mse

w = np.array((-40, -40))
alpha = 0.1
tolerance = 1e-3 # this thing is convergence tolerance

#creating two more arrays, one for storing all the intermediate w and mse
old_w = []
errors = []

# Perform Gradient Descent
iterations = 1
for i in range(200):
    gradient, error = gradient_descent(w, X_scaled, Y)
    new_w = w - alpha * gradient
 
    # Print error every 10 iterations
    if iterations % 10 == 0:
        print("Iteration: %d - Error: %.4f" % (iterations, error))
        old_w.append(new_w)
        errors.append(error)
 
    # Stopping Condition
    if np.sum(abs(new_w - w)) < tolerance:
        print('Gradient Descent has converged')
        break
 
    iterations += 1
    w = new_w

    
all_ws = np.array(old_w)
levels = np.sort(np.array(errors))

w0 = np.linspace(-w[0] * 5, w[0] * 5, 100)
w1 = np.linspace(-w[1] * 5, w[1] * 5, 100)
mse_vals = np.zeros(shape=(w0.size, w1.size))

for i, value1 in enumerate(w0):
    for j, value2 in enumerate(w1):
        w_temp = np.array((value1,value2))        
        mse_vals[i, j] = gradient_descent(w_temp, X_scaled, Y)[1]


# Call the plt.annotate() function in loops to create the arrow which shows the convergence path of the gradient descent
plt.contourf(w0, w1, mse_vals, levels,alpha=0.7)

for i in range(len(old_w) - 1):
    plt.annotate('', xy=all_ws[i + 1, :], xytext=all_ws[i, :],# the start and the end points of the arrows pointing towards the centre
                 arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
                 va='center', ha='center')
 
CS = plt.contour(w0, w1, mse_vals, levels, linewidths=1,colors='black')
plt.clabel(CS, inline=1, fontsize=8)
plt.title("Contour Plot of Gradient Descent")
plt.xlabel("w0")
plt.ylabel("w1")
plt.show()

