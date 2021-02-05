import numpy as np


class HistogramFilter(object):
	"""
	Class HistogramFilter implements the Bayes Filter on a discretized grid space.
	"""

	def histogram_filter(self, cmap, belief, action, observation):
		'''
		Takes in a prior belief distribution, a colormap, action, and observation, and returns the posterior
		belief distribution according to the Bayes Filter.
		:param cmap: The binary NxM colormap known to the robot.
		:param belief: An NxM numpy ndarray representing the prior belief.
		:param action: The action as a numpy ndarray. [(1, 0), (-1, 0), (0, 1), (0, -1)]
		:param observation: The observation from the color sensor. [0 or 1].
		:return: The posterior distribution.
		Use starter.npz data given in /data to debug and test your code. starter.npz has sequence of actions and 
		corresponding observations. True belief is also given to compare your filters results with actual results. 
		cmap = arr_0, actions = arr_1, observations = arr_2, true state = arr_3
		'''
		### Your Algorithm goes Below.

		
		n, m = cmap.shape;
		# print("belief:\n" + str(np.round(belief, 5)));
		stay_ = np.eye(m, dtype = float) / 10;

		T = np.copy(belief);
		T *= 0.9;
		if action[0] == 0: # up and down
			T = np.roll(T, action[1], axis = 0)
			if(action[1] > 0):
				T[0,:] = 0;
				T[-1,:] += belief[-1,:];
			else:
				T[-1,:] = 0;
				T[0,:] += belief[0,:];
		else: # left and right
			T = np.roll(T, action[0], axis = 1)
			if(action[0] > 0):
				T[:,0] = 0
				T[:,-1] += belief[:,-1];
			else:
				T[:,-1] = 0
				T[:,0] += belief[:,0];

		belief_ = T + belief @ stay_;
		
		M = np.ones((n, m), dtype = float);
		
		M[cmap == observation] = 9

		M /= 10;
		belief_ *= M;
				
		belief_ /= np.sum(belief_);	
		
		print("belief_:\n" + str(np.round(belief_, 3)));

		return belief_;