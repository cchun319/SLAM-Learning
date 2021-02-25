import numpy as np
import math
import random

X = [];
x_0 = np.random.normal(loc = 1.0, scale = math.sqrt(2.0), size=None)
X.append(x_0)
a = -1;

Y = [];
y_0 = math.sqrt(x_0 ** 2) + np.random.normal(loc = 0.0, scale = math.sqrt(0.5), size=None)
Y.append(y_0)

for _ in range(100):
	eps = np.random.normal(loc = 0.0, scale = math.sqrt(1.0), size=None)
	x_k = a * X[-1] + eps
	X.append(x_k)
	y_k = math.sqrt(x_k ** 2) + np.random.normal(loc = 0.0, scale = math.sqrt(0.5), size=None)
	Y.append(y_k)
	
print(Y);