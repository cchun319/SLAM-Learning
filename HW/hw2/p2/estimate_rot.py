import numpy as np
from scipy import io
from quaternion import Quaternion
import matplotlib.pyplot as plt
import math

#data files are numbered on the server.
#for exmaple imuRaw1.mat, imuRaw2.mat and so on.
#write a function that takes in an input number (1 through 6)
#reads in the corresponding imu Data, and estimates
#roll pitch and yaw using an unscented kalman filter

alpha = 200;
beta = 0.5;

def digitalToAnalog(raw, alpha_, beta_):

	ret = (raw - beta_) * 3330 / (1023 * alpha_)
	return ret;


def estimate_rot(data_num=1):
	#load data
	imu = io.loadmat('imu/imuRaw'+str(data_num)+'.mat')
	vicon = io.loadmat('vicon/viconRot'+str(data_num)+'.mat')
	accel = imu['vals'][0:3,:]
	gyro = imu['vals'][3:6,:]
	T = np.shape(imu['ts'])[1]

	accel_ana = digitalToAnalog(accel, alpha, beta);
	x = range(accel.shape[-1]);

	fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
	fig.suptitle('Sharing x per column, y per row')
	ax1.plot(x, accel_ana[0,:], label = "ax")
	ax1.plot(x, accel_ana[1,:], label = "ay")
	ax1.plot(x, accel_ana[2,:], label = "az")
	ax2.plot(x, gyro[0,:], label = "ax")
	ax3.plot(x, gyro[1,:], label = "ay")
	ax4.plot(x, gyro[2,:], label = "az")
	ax1.legend()
	ax2.legend()
	ax3.legend()
	ax4.legend()
	ax1.set_title('acc')
	ax2.set_title('yaw')
	ax3.set_title('pitch')
	ax4.set_title('roll')
	plt.show();
	# for ax in fig.get_axes():
	# 	ax.label_outer()
	print(accel.shape)
	# plt.plot(x, accel_ana[0,:], label = "ax")
	# plt.plot(x, accel_ana[1,:], label = "ay")
	# plt.plot(x, accel_ana[2,:], label = "az")
	# # plt.plot(x, np.sqrt(accel_ana[2,:]**2 + accel_ana[0,:]**2 + accel_ana[1,:]**2), label = "line 1")
	# plt.show();
	# your code goes here
	print(vicon['rots'].shape);


	# roll, pitch, yaw are numpy arrays of length T
	return roll,pitch,yaw

estimate_rot();