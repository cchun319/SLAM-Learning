import numpy as np;
from numpy import linalg as LA
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

board = np.zeros((10, 10));

# make the board
board[9 - 1, 8] = 2;
board[9 - 6, 3] = 1;
board[9 - 7, 5] = -10;
board[4:6, 7] = -10;
board[2:6, 4] = -10;
board[9 - 2, 3:7] = -10;
board[[0,-1], :] = -10;
board[:,[0,-1]] = -10;
board_vec = board.flatten('C');
print(board)
# board = board.reshape((100,1))
# print(board.reshape((100,1)));

def generateQT(p):

	q = np.zeros_like(p);
	T = np.zeros((100, 100));
	np.fill_diagonal(T, 0.1);
	for i in range(len(p)):
		r = (int)(i / 10);
		c = (int)(i % 10);
		destr = 0;
		destc = 0;
		stay1r = r;
		stay1c = c;
		stay2r = r;
		stay2c = c;
        
		control = p[i];

		# 0: n, 1:e, 2:s, 3:w
		if(p[i] == 0):
			stay1c += 1;
			stay2c += -1;
			r += 1; c += 0;
		if(p[i] == 1):
			stay1r += 1;
			stay2r += -1;
			r += 0; c += 1;
		if(p[i] == 2):
			stay1c += 1;
			stay2c += -1;
			r += -1; c += 0;
		if(p[i] == 3):
			stay1r += 1;
			stay2r += -1;
			r += 0; c += -1;

		
		if(board_vec[i] == 2):
			T[i,i] = 1
			q[i] = 9;
			continue;
		elif(r < 0 or r > 9 or c < 0 or c > 9 or board[r][c] < 0 or board_vec[i] < 0):
			q[i] = -11;
		else:
			q[i] = -1;

		if(board_vec[i] < 0): # border
			T[i,i] = 1
			continue;
		
		destr = r;
		destc = c;
		dest = r * 10 + c;
		stay1 = stay1r * 10 + stay1c;
		stay2 = stay2r * 10 + stay2c;
		T[i, dest] = 0.7;
		T[i, stay1] = 0.1;
		T[i, stay2] = 0.1;


	return q, T;

v = [1, 0, -1, 0];
h = [0, 1, 0, -1];

def newPolicy(J):
	ctl_policy = np.ones((100,1));
	for i in range(1,9):
		for j in range(1,9):
			maxValue = -100;
			maxd = 0;
			for k in range(4):
				nr = i + v[k];
				nc = j + h[k]
				if(J[nr][nc] > maxValue):
					maxValue = J[nr][nc];
					maxd = k;
			ctl_policy[i*10 + j] = maxd;

	return ctl_policy;

def printJ(J):
	for i in range(10):
		for j in range(10):
			print("%.2f" % round(J[i][j], 2), end = "\t")
		print("\n");


# print(board);

# transision matrix



ctl_policy = np.ones((100,1));
ctl_policy_prev = np.zeros((100,1));
J_opt_10x10 = np.zeros((10,10))
ct = 0;
while(LA.norm(ctl_policy - ctl_policy_prev) > 1e-5 and ct < 200):
	# use current policy to genereate q vector;
	q, T = generateQT(ctl_policy)

	J_opt = LA.pinv(np.diag([1]*100) - 0.9 * T) @ q;
	J_opt_10x10 = np.reshape(J_opt, (10,10), 'C')
	if(ct == 0):
		printJ(J_opt_10x10);
		ax = sns.heatmap(J_opt_10x10)
	ctl_policy_prev = ctl_policy;
	ctl_policy = newPolicy(J_opt_10x10)
	ct += 1;

print(np.reshape(ctl_policy, (10,10), 'C'))
printJ(J_opt_10x10)



X = np.array((0))
Y= np.array((0.5))
U = np.array((2))
V = np.array((0))

fig, ax = plt.subplots()
q = ax.quiver(X, Y, U, V,units='xy' ,scale=1)

# plt.grid()
major_ticks = np.arange(0, 11, 1)
ax.set_xticks(major_ticks);
ax.set_yticks(major_ticks);
ax.set_aspect('equal')
ax.grid(which='both');
plt.xlim(0,10)
plt.ylim(0,10)

plt.title('How to plot a vector in matplotlib ?',fontsize=10)

# plt.savefig('how_to_plot_a_vector_in_matplotlib_fig3.png', bbox_inches='tight')
plt.show()