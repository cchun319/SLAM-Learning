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

def CholeskyMatrix(P_prev):
	return None;


def estimate_rot(data_num=1):
	#load data
	imu = io.loadmat('imu/imuRaw'+str(data_num)+'.mat')
	vicon = io.loadmat('vicon/viconRot'+str(data_num)+'.mat')
	accel = imu['vals'][0:3,:]
	gyro = imu['vals'][3:6,:]
	T = np.shape(imu['ts'])[1]

	ax = digitalToAnalog(accel[0, :], 33.0, 510.0);
	ay = digitalToAnalog(accel[1, :], 33.0, 497.0);
	az = digitalToAnalog(accel[2, :], 33.0, 510.0);

	t = imu['ts'].T;

	quaterions = [];
	for i in range(vicon['rots'].shape[-1]):
		rot = vicon['rots'][:,:,i].reshape(3,3)
		q = rotationMatrixToEulerAngles(rot)
		quaterions.append(q)
	quaterions = np.asarray(quaterions);
	quaterions = quaterions.T;


	yaw, pitch, roll = accToRPY(-ax, -ay, az);

	gyro_dig = digitalToAnalog(gyro, 280.0, 370.0);

	# your code goes here
	X = np.zeros((7, 1))
	P_prev = np.zeros((6,6))
	Q = np.random.rand(6,6)
	S = CholeskyMatrix(P_prev + Q)
	W = 




	# roll, pitch, yaw are numpy arrays of length T
	return roll,pitch,yaw

estimate_rot();