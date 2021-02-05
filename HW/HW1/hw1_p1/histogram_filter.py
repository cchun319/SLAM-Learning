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

		T = np.copy(belief);
		T *= 0.9;
		action_ = np.zeros_like(action)
		action_[0] = -action[1]
		action_[1] = action[0]

		border = np.zeros_like(cmap, dtype = float)
		# print("actionOOO:\n" + str(action));
		# print("action_:\n" + str(action_));
		# print("val:\n" + str(action_[np.where(action_ != 0)][0]));
		# print("aid:\n" + str(np.where(action_ != 0)[0][0]));
		# print("T:\n" + str(np.round(T, 3)));
		T = np.roll(T, action_[np.where(action_ != 0)][0], axis = np.where(action_ != 0)[0][0])
		
		if np.where(action_ != 0)[0][0] == 0:
			if action_[np.where(action_ != 0)][0] > 0:
				T[0,:] = 0
				border[-1, :] = belief[-1, :];
				# print("T:\n" + str(np.round(T, 3)));
			else:
				T[-1,:] = 0
				border[0, :] = belief[0, :];
				# print("T:\n" + str(np.round(T, 3)));
		else:
			if action_[np.where(action_ != 0)][0] > 0:
				T[:,0] = 0
				border[:, -1] = belief[:, -1];
			else:
				T[:, -1] = 0
				border[:, 0] = belief[:, 0];


		belief_ = T + belief * 0.1 + border;
		
		M = np.ones((n, m), dtype = float);
		
		M[cmap == observation] = 9

		M /= 10;
		belief_ *= M;
				
		belief_ /= np.sum(belief_);	
		
		# print("belief_:\n" + str(np.round(belief_, 3)));

		return belief_;