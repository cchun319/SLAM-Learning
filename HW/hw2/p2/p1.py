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
# S[x]
#  [a]

state_[0, 0] = 1;
state_[1, 0] = -40
pro_noise = np.diag([1, 0.1])
Q = 1;
cov = [];
states = []
for i in range(100):
	A = np.array([[state_[1, 0], state_[0, 0]],[0, 1]]);

	# A[a, x_k]
	#  [0, 1]

	# noise[eps]
	#      [a_noise]

	predict_state = np.array([[state_[0, 0] * state_[1, 0]],[state_[1, 0]]])
	predict_cov = A @ cov_ @ A.T + pro_noise;
	# pro_noise[eps_deviation, 0]
	#          [0, a_noise]
	C = np.zeros((1,2));
	C[0, 0] = predict_state[0, 0] / np.sqrt(predict_state[0, 0] ** 2 + 1);

	# C[partial derivative of g, 0]
	#  [0, 0]

	K = predict_cov @ C.T @ np.linalg.pinv(C @ predict_cov @ C.T + Q);

	cov_ = (np.eye(2,2) - K @ C) @ predict_cov; 
	state_ = predict_state + K * (Y[i] - np.sqrt(predict_state[0, 0] ** 2 + 1))
	# state_[1,0] = predict_state[1,0]
	states.append(state_)
	cov.append([state_[1,0] - np.sqrt(cov_[1,1]), state_[1,0] + np.sqrt(cov_[1,1])])

states = np.asarray(states).T;
cov = np.asarray(cov).T;
fig, ((ax1, ax2)) = plt.subplots(1,2)
ax1.plot(range(100), states[0, 0,:], label = "track Y")
ax1.plot(range(100), Y, label = "TrueY")
ax2.plot(range(100), states[0, 1,:], label = "A")
ax2.vlines(range(100),cov[0,:], cov[1,:], alpha=.3)
ax2.scatter(range(100),cov[1,:], marker='.', alpha=.1)
ax2.scatter(range(100),cov[0,:], marker='.', alpha=.1)
ax1.legend()
ax2.legend()
plt.show();