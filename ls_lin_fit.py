import numpy as np
import matplotlib.pyplot as plt

# Random data
N = 10
M = 2
input = np.random.random((N, M))
# print(input)

# Setup matrices
m = np.shape(input)[0]
X = np.matrix([np.ones(m), input[:, 0]]).T
y = np.matrix(input[:, 1]).T

# Solve for projection matrix
p_mat = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
print(p_mat)

# Find regression line
xx = np.linspace(0, 1, 2)
yy = np.array(p_mat[0] + p_mat[1] * xx)

# Plot data, regression line
plt.figure(1)
plt.plot(xx, yy.T, color='b')
plt.scatter(input[:, 0], input[:, 1], color='r')
plt.show()
