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
		pb = np.round(belief, 3);
		print("belief:\n" + str(pb));
		# print("%.2f" % np.round(belief, 2))
		n, m = cmap.shape;
		# print(str(n) + " and " + str(m));
		# print("action:\n" + str(action));
		# print("action[0]:\n" + str(action[0]));
		# print("action[1]:\n" + str(action[1]));
		belief_ = np.zeros_like(belief)
		for i in range(n):
			for j in range(m):

				nr = i + action[0];
				nc = j + action[1];
				nr = min(n - 1, nr)
				nr = max(0, nr)
				nc = min(m - 1, nc)
				nc = max(0, nc)
				belief_[nr][nc] += belief[i][j] * 0.9;
				belief_[i][j] += belief[i][j] * 0.1;
		norm = 0;
		
		for i in range(n):
			for j in range(m):
				if(observation == cmap[i][j]):
					belief_[i][j] *= 0.9;
				else:
					belief_[i][j] *= 0.1;
				norm += belief_[i][j];

				
		belief_ /= norm;	

		
		# print("belief_:\n" + str(belief_));
		# print("observation:\n" + str(observation));

		return belief_;