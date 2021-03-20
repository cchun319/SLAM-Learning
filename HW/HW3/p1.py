import numpy as np;
from numpy import linalg as LA

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
			stay1c +=1;
			stay2c += -1;
			r += 1; c +=0;
		if(p[i] == 1):
			stay1r += 1;
			stay2r += -1;
			r += 0; c += 1;
		if(p[i] == 2):
			stay1c +=1;
			stay2c += -1;
			r += -1; c +=0;
		if(p[i] == 3):
			stay1r += 1;
			stay2r += -1;
			r += 0; c += -1;

		if(r < 0 or r > 9 or c < 0 or c > 9 or board[r][c] < 0):
			q[i] = -10;
		elif(board[r][c] == 2):
			q[i] = 10;
		else:
			q[i] = -1;

		if(board_vec[i] < 0): # border
			T[i,:] = 0
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

v = [1,0,-1,0];
h = [0,1,0,-1];
def newPolicy(J):
	ctl_policy = np.ones((100,1));
	for i in range(1,9):
		for j in range(1,9):
			maxValue = 0;
			maxd = 0;
			for k in range(4):
				nr = i + v[k];
				nc = j + h[k]
				if(J[nr][nc] > maxValue):
					maxValue = J[nr][nc];
					maxd = k;
			ctl_policy[i*10 + j] = maxd;

	return ctl_policy;


# print(board);

# transision matrix


ctl_policy = np.ones((100,1));
ctl_policy_prev = np.zeros((100,1));
J_opt_10x10 = np.zeros((10,10))
while(LA.norm(ctl_policy - ctl_policy_prev) > 1e-5):
	# use current policy to genereate q vector;
	q, T = generateQT(ctl_policy)

	J_opt = LA.pinv(np.diag([1]*100) - 0.9 * T) @ q;
	J_opt_10x10 = np.reshape(J_opt, (10,10), 'C')
	ctl_policy_prev = ctl_policy;
	ctl_policy = newPolicy(J_opt_10x10)

print(np.reshape(ctl_policy, (10,10), 'C'))
print(J_opt_10x10)

