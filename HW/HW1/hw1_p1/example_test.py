import numpy as np
import matplotlib.pyplot as plt
from histogram_filter import HistogramFilter
import random


if __name__ == "__main__":

	# Load the data
	data = np.load(open('data/starter.npz', 'rb'))
	cmap = data['arr_0']
	actions = data['arr_1']
	observations = data['arr_2']
	belief_states = data['arr_3']


	#### Test your code here
	filter = HistogramFilter();
	belief_ = np.ones_like(cmap) / (cmap.shape[0] * cmap.shape[1]);
	

	for i in range(len(actions)):
		belief_ = filter.histogram_filter(cmap, belief_, actions[i], observations[i]);
		ind = np.unravel_index(np.argmax(belief_, axis=None), belief_.shape)
		print("iter: " + str(i) + " \nBAYES:" + str(ind)  + " PROB: "+ str(belief_[ind]) + " TRUE:" + str(belief_states[i]) + "\n\n");