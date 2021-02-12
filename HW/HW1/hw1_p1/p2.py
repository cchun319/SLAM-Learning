import numpy as np

T = 0.5 * np.ones((2,2), dtype = float);
M = np.array([[0.4, 0.1, 0.5], [0.1, 0.5, 0.4]]);

alpha_0 = np.array([[0.5], [0.5]]);
Y = np.array([2, 0, 0, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 0, 0, 1]);

alpha_1 = alpha_0 * M[:,Y[0]].reshape(2,1)

# print(alpha_1)
beta = np.array([[1], [1]])
alpha = alpha_1
for i in range(1, 20):
	###
	# print("alpha1\n" + str(alpha_1.reshape(2,1)))
	# print("M\n" + str( M[:,Y[i]].reshape(2,1)))
	# print("T*M\n" + str(T * M[:,Y[i]].reshape(2,1)))
	###
	alpha_1 = (T * M[:,Y[i]].reshape(2,1)) @ alpha_1.reshape(2,1)
	###
	# print("new alpha1\n" + str(alpha_1.reshape(2,1)) + "\n")
	###
	alpha = np.hstack((alpha, alpha_1))

print("alpha: \n" + str(alpha.T) + "\n")

total_observation = np.sum(alpha[:, -1], axis = 0);

betas = beta;
for i in range(0, 20):
	###
	# print("beta\n" + str(beta.reshape(2,1)))
	# print("M\n" + str( M[:,Y[19 - i]].reshape(2,1)))
	# print("beta*M\n" + str(beta.reshape(2,1) * M[:,Y[19 - i]].reshape(2,1)))
	###
	beta = T.T @ (beta.reshape(2,1) * M[:,Y[19 - i]].reshape(2,1))
	# print("new beta\n" + str(beta) + "\n")
	betas = np.hstack((beta, betas))
betas = betas[:,0:-1]
print("beta: \n" + str(betas.T) + "\n")


gamma_ = alpha * betas
gamma_sum = np.sum(gamma_, axis = 0)
gamma_f = gamma_ / gamma_sum;
print("gamma: \n" + str(gamma_f.T) + "\n")
print("Most possible state: \n" + str(np.argmax(gamma_f, axis= 0)) + "\n")
gamma_sum_col = np.sum(gamma_f[:, :-1], axis = 1)
# print("gamma_sum_col: \n" + str(gamma_sum_col))


T_new = np.zeros((2,2)); 
for i in range(0, 20 - 1):
	
	xi_k = alpha[:,i].reshape(2,1) @ (betas[:,i + 1].reshape(1,2) * M[:, Y[i + 1]].reshape(1,2)) * T.T
	xi_sum = np.sum(xi_k); 
	xi_k /= xi_sum;
	print("XI" + str(i + 1) + ":")
	print(xi_k)

	T_new += xi_k.T;

print()
# gamma 2 x 20
T_new /= gamma_sum_col

print("T': \n" + str(T_new) + str("\n"));

M_new = np.zeros((2,3));
Indicator = np.zeros((20, 3));

for i in range(3):
	Indicator[ Y == i,i] = 1

gamma_sum_col_full = np.sum(gamma_f, axis = 1)

M_new = gamma_f @ Indicator / gamma_sum_col_full.reshape(2,1)

print("M': \n" + str(M_new) + "\n")


alpha_1 = gamma_f[:,0].reshape(2,1) * M[:,Y[0]].reshape(2,1)

# print(alpha_1)
beta = np.array([[1], [1]])
alpha_new = alpha_1
for i in range(1, 20):
	###
	# print("alpha1\n" + str(alpha_1.reshape(2,1)))
	# print("M\n" + str( M[:,Y[i]].reshape(2,1)))
	# print("T*M\n" + str(T * M[:,Y[i]].reshape(2,1)))
	###
	alpha_1 = (T_new * M_new[:,Y[i]].reshape(2,1)) @ alpha_1.reshape(2,1)
	###
	# print("new alpha1\n" + str(alpha_1.reshape(2,1)) + "\n")
	###
	alpha_new = np.hstack((alpha_new, alpha_1))

total_observation_new = np.sum(alpha_new[:, -1], axis = 0);

print("total_observation: \n" + str(total_observation) + "\n")
print("total_observation_new: \n" + str(total_observation_new))




