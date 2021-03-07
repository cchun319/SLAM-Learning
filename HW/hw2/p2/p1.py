import numpy as np
import math
import random
import matplotlib.pyplot as plt

X = [];
x_0 = np.random.normal(loc = 1.0, scale = math.sqrt(2.0), size=None)
X.append(x_0)
a = -1;

Y = [];
y_0 = math.sqrt(x_0 ** 2) + np.random.normal(loc = 0.0, scale = math.sqrt(0.5), size=None)

for _ in range(100):
	eps = np.random.normal(loc = 0.0, scale = math.sqrt(1.0), size=None)
	x_k = a * X[-1] + eps
	X.append(x_k)
	y_k = math.sqrt(x_k ** 2) + np.random.normal(loc = 0.0, scale = math.sqrt(0.5), size=None)
	Y.append(y_k)

# print(Y);

cov_ = np.eye(2,2);
state_ = np.zeros((2,1))
state_[0, 0] = np.random.normal(loc = 1.0, scale = math.sqrt(2.0), size=None);
pro_noise = np.diag([1, 0.1])
Q = np.diag([0.5, 0.1])

states = []
for i in range(100):
	A = np.array([[state_[1, 0], 0],[0, 1]]);

	# A[a, 0]
	#  [0, 1]

	noise = np.zeros((2,1));
	noise[0,0] = np.random.normal(loc = 0, scale = math.sqrt(1), size=None)
	noise[1,0] = np.random.normal(loc = 0, scale = math.sqrt(0.1), size=None)

	# noise[eps]
	#      [a_noise]

	predict_state = A @ state_ + noise;
	predict_cov = A @ cov_ @ A.T + pro_noise;
	# pro_noise[eps_deviation, 0]
	#          [0, a_noise]
	C = np.zeros((2,2));
	C[0, 0] = 2 * predict_state[0, 0] / np.sqrt(predict_state[0, 0] ** 2 + 1);

	# C[partial derivative of g, 0]
	#  [0, 0]

	K = predict_cov @ C.T @ np.linalg.pinv(C @ predict_cov @ C.T + Q);
	print(C.shape)
	print(predict_state.shape)
	cov_ = (np.eye(2,2) - K @ C) @ predict_cov; 
	state_[0,0] = predict_state[0,0] + K[0,0] * (Y[i] - np.sqrt(predict_state[0, 0] ** 2 + 1))
	state_[1,0] = predict_state[1,0]
	states.append(state_)
	print(state_.T)

states = np.asarray(states).T;
print(states.shape)
fig, ((ax1, ax2)) = plt.subplots(1,2)
fig.suptitle('filtered angular velocity')
ax1.plot(range(100), states[0, 0,:], label = "track Y")
ax1.plot(range(100), Y, label = "TrueY")
ax2.plot(range(100), states[0, 1,:], label = "A")
ax1.legend()
ax2.legend()
ax1.set_title('ome_x')
ax2.set_title('ome_y')
plt.show();