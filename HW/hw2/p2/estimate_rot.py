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

def accToRPY(accelerationX, accelerationY, accelerationZ) :
	pitch =  np.arctan2(accelerationX, np.sqrt(accelerationY*accelerationY + accelerationZ*accelerationZ));
	roll =  np.arctan2(accelerationY, np.sqrt(accelerationX*accelerationX + accelerationZ*accelerationZ));
	yaw =  np.arctan2(accelerationZ, np.sqrt(accelerationX*accelerationX + accelerationZ*accelerationZ));
	return yaw, pitch, roll

def isRotationMatrix(R) :
	Rt = np.transpose(R)
	shouldBeIdentity = np.dot(Rt, R)
	I = np.identity(3, dtype = R.dtype)
	n = np.linalg.norm(I - shouldBeIdentity)
	return n < 1e-6

def rotationMatrixToEulerAngles(R) :
	assert(isRotationMatrix(R))
	sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
	singular = sy < 1e-6
	if  not singular :
		x = math.atan2(R[2,1] , R[2,2])
		y = math.atan2(-R[2,0], sy)
		z = math.atan2(R[1,0], R[0,0])
	else :
		x = math.atan2(-R[1,2], R[1,1])
		y = math.atan2(-R[2,0], sy)
		z = 0
	return np.array([x, y, z])

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

	ax = digitalToAnalog(accel[0, :], 33.0, 510.0);
	ay = digitalToAnalog(accel[1, :], 33.0, 497.0);
	az = digitalToAnalog(accel[2, :], 33.0, 510.0);
	# accel_ana = accel;
	x = imu['ts'].T;

	quaterions = [];
	for i in range(vicon['rots'].shape[-1]):
		rot = vicon['rots'][:,:,i].reshape(3,3)
		q = rotationMatrixToEulerAngles(rot)
		quaterions.append(q)
	quaterions = np.asarray(quaterions);
	quaterions = quaterions.T;
	print(quaterions.shape);


	yaw, pitch, roll = accToRPY(ax, ay, -az);
	patch = np.zeros((3, 84));
	quaterions = np.hstack((quaterions, patch))

	fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
	fig.suptitle('Sharing x per column, y per row')
	ax1.plot(x, ax, label = "ax")
	ax1.plot(x, ay, label = "ay")
	ax1.plot(x, az, label = "az")
	ax2.plot(x, roll, label = "ax")
	ax3.plot(x, pitch, label = "ay")
	ax4.plot(x, yaw, label = "az")
	ax2.plot(x, quaterions[0,:], label = "viconx")
	ax3.plot(x, quaterions[1,:], label = "vicony")
	ax4.plot(x, quaterions[2,:], label = "viconz")
	ax1.legend()
	ax2.legend()
	ax3.legend()
	ax4.legend()
	ax1.set_title('acc')
	ax2.set_title('yaw')
	ax3.set_title('pitch')
	ax4.set_title('roll')
	plt.show();

	print(np.amin(gyro[0,:]));
	print(np.amax(gyro[0,:]));
	print(gyro)
	gyro_dig = digitalToAnalog(gyro, 200.0, 370.0);
	gyro_accu = np.array([[roll[0]],[pitch[0]],[yaw[0]]]);
	p_time = x[0];
	for i in range(1,T):
		new_gyro = gyro_accu[:,-1] + (x[i] - p_time) * gyro_dig[:,i];
		new_gyro = np.reshape(new_gyro, (3,1))
		gyro_accu = np.hstack((gyro_accu, new_gyro))
		p_time = x[i]

	print(gyro_accu.shape)	
	fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
	ax1.plot(x, gyro_accu[0,:], label = "ax")
	ax2.plot(x, gyro_accu[1,:], label = "ax")
	ax3.plot(x, gyro_accu[2,:], label = "vicony")
	ax1.legend()
	ax2.legend()
	ax3.legend()
	ax1.set_title('acc')
	ax2.set_title('yaw')
	ax3.set_title('pitch')
	plt.show();
	# your code goes here
	


	# roll, pitch, yaw are numpy arrays of length T
	return roll,pitch,yaw

estimate_rot();