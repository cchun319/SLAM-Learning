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
	print("alpha1\n" + str(alpha_1.reshape(2,1)))
	print("M\n" + str( M[:,Y[i]].reshape(2,1)))
	print("T*M\n" + str(T * M[:,Y[i]].reshape(2,1)))
	###
	alpha_1 = (T * M[:,Y[i]].reshape(2,1)) @ alpha_1.reshape(2,1)
	###
	print("new alpha1\n" + str(alpha_1.reshape(2,1)) + "\n")
	###
	alpha = np.hstack((alpha, alpha_1))

print(alpha)
alpha_sum = np.sum(alpha[:, -1], axis = 0);
print("alpha_sum\n" + str(alpha_sum))

betas = beta;
for i in range(0, 20):
	###
	print("beta\n" + str(beta.reshape(2,1)))
	print("M\n" + str( M[:,Y[19 - i]].reshape(2,1)))
	print("beta*M\n" + str(beta.reshape(2,1) * M[:,Y[19 - i]].reshape(2,1)))
	###
	beta = T.T @ (beta.reshape(2,1) * M[:,Y[19 - i]].reshape(2,1))
	print("new beta\n" + str(beta) + "\n")
	betas = np.hstack((beta, betas))
betas = betas[:,0:-1]
print(betas)
beta_sum = np.sum(betas[:,0] * M[:, Y[0]] /2 )
print("beta_sum\n" + str(beta_sum))


gamma_ = alpha * betas
# # gamma_ /= alpha_sum
print("gamma: \n" + str(gamma_ / alpha_sum))
print(np.argmax(gamma_, axis= 0))

gamma_sum = np.sum(gamma_, axis = 1).reshape((2,1))
# ss
pi_0 = gamma_[:,0];
T_new = np.zeros((2,2)); 
for i in range(0, 20 - 1):
	xi_k = alpha[:,i].reshape(2,1) @ (betas[:,i + 1].reshape(1,2) * M[:, Y[i + 1]].reshape(1,2)) * T.T  
	T_new += xi_k.T;
	print(T_new/gamma_sum)
	
# T_new /= gamma_sum
# print(T_new)




