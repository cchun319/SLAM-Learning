import numpy as np
from scipy import io
from quaternion import Quaternion
import matplotlib.pyplot as plt
import math
from numpy import linalg as LA

#data files are numbered on the server.
#for exmaple imuRaw1.mat, imuRaw2.mat and so on.
#write a function that takes in an input number (1 through 6)
#reads in the corresponding imu Data, and estimates
#roll pitch and yaw using an unscented kalman filter

def CholeskyMatrix(P_prev):
	return np.linalg.cholesky(P_prev);

def propagateState(X, W):
	q_x = Quaternion(X[0,0], X[1:4, 0]);
	X_p = np.zeros((7, 12))
	for i in range(W.shape[-1]):
		q_w = Quaternion(0, W[0:3, i]);
		omega_k = X[4:7, 0] + W[3:6, i]
		q_k = q_x.__mul__(q_w);
		x_k = np.zeros((7, 1));
		x_k[4:7, 0] = omega_k;
		x_k[0,0] = q_k.scalar();
		x_k[1:4, 0] = q_k.vec();
		X_p[:, i] = x_k[:, 0];
	return X_p;

def processA(X_p, prev, timeStamp):
	# Todo
	delta_t = timeStamp - prev;
	for i in range(X_p.shape[-1]):
		q_ome = Quaternion();
		q_ome.from_axis_angle(X_p[4:7, i] * delta_t); # alpha_delta_t
		q_cur = Quaternion(X_p[0,i], X_p[1:4, i]);
		q_predict = q_cur.__mul__(q_ome);
		X_p[0, i] = q_predict.scalar();
		X_p[1:4, i] = q_predict.vec();
	return X_p;

def getSigmaPts(S):
	W = np.zeros((6, 12))
	a = np.sqrt(12);
	for i in range(S.shape[0]):
		W[:, 2*i] = a * S[:,i];
		W[:, 2*i + 1] = -a * S[:,i];
	return W;

def getBaryMean(B):
	return np.mean(B, axis = 1)

def getQtError(q, q_barinv): # q_bar is Quaternion instance, q is vector 
	q_i = Quaternion(q[0], q[1:4])
	return q_i.__mul__(q_barinv)


def getMeanQuaternion(q_cur):
	q = Quaternion(0, [0,0,0]);
	q_prev = Quaternion(-1, [-1,-1,-1]);
	threshould = 0.001;
	ct = 0;
	err_ = [q.scalar() - q_prev.scalar(), (q.vec() - q_prev.vec())[0], (q.vec() - q_prev.vec())[1], (q.vec() - q_prev.vec())[2]];
	while(LA.norm(err_) < threshould or ct < 1000):
		q_inv = q.inv();
		E_i = np.zeros((4, 12));
		for i in range(q_cur.shape[-1]):
			er = getQtError(q_cur[0:4, i], q_inv);
			E_i[0, i] = er.scalar();
			E_i[1:4, i] = er.vec();

		e_bar = getBaryMean(E_i);
		e_q = Quaternion(e_bar[0], e_bar[1:4])
		q_prev = q;
		q = e_q.__mul__(q);
		err_ = [q.scalar() - q_prev.scalar(), (q.vec() - q_prev.vec())[0], (q.vec() - q_prev.vec())[1], (q.vec() - q_prev.vec())[2]]
		ct += 1;
	
	return q;

def getMean(Y_i):
	Y_mean = np.zeros((7,1));
	Y_mean[4:7, 0] = getBaryMean(Y_i[4:7, :])
	Y_mean_q = getMeanQuaternion(Y_i[0:4, :]);
	Y_mean[0, 0] = Y_mean_q.scalar();
	Y_mean[1:4, 0] = Y_mean_q.vec();
	return Y_mean;

def getWDeviation(W, Y_i):
	W_d = np.zeros((6, 12));
	q_bar = Quaternion(Y_i[0,0],Y_i[1:4, 0]);
	q_barinv = q_bar.inv()
	for i in range(W.shape[-1]):
		new_col = np.zeros((6, 1));
		new_col[0:3, 0] = getQtError([0, W[0, i], W[1, i], W[2, i]], q_barinv).vec(); # the scalar part might be wrong
		new_col[3:6, 0] = W[3:6, i] - Y_i[4:7, 0];
		W_d[:, i] = new_col[:, 0];
	return W_d;

def rotateVector(q, g):
	q_i = Quaternion(q[0], q[1:4]);

	return q_i.inv().__mul__(g.__mul__(q_i));

def getMeasureEstimate(X_propagate):
	Z = np.zeros((6,12));
	g_ = Quaternion(0,[0,0,9.8]); # TODO
	for i in range(X_propagate.shape[-1]):
		z_i = np.zeros((6, 1))
		z_i[0:3, 0] = X_propagate[4:7, i];
		z_i[3:6, 0] = rotateVector(X_propagate[0:4, i], g_).vec(); # might be wrong
		Z[:, i] = z_i[:, 0]
	return Z;

def digitalToAnalog(raw, alpha_, beta_):

	ret = (raw - beta_) * 3330 / (1023 * alpha_)
	return ret;

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

def accToRPY(accelerationX, accelerationY, accelerationZ) :
	pitch =  np.arctan2(-accelerationX, np.sqrt(accelerationY*accelerationY + accelerationZ*accelerationZ));
	roll =  np.arctan2(accelerationY, accelerationZ);
	yaw =  np.arctan2(accelerationZ, np.sqrt(accelerationX*accelerationX + accelerationZ*accelerationZ));
	return yaw, pitch, roll

def isRotationMatrix(R) :
	Rt = np.transpose(R)
	shouldBeIdentity = np.dot(Rt, R)
	I = np.identity(3, dtype = R.dtype)
	n = np.linalg.norm(I - shouldBeIdentity)
	return n < 1e-6

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
	roll_ = []
	pitch_ = []
	yaw_ = []
	R = np.zeros((6, 6))
	X = np.zeros((7, 1))
	P_prev = np.zeros((6,6))
	Q = np.eye(6,6)
	t_prev = t[0];
	for i in range(1, T):
		print("P_prev: \n" + str(P_prev + Q) + "\n\n");
		t_cur = t[i];
		S = CholeskyMatrix(P_prev + Q)
		W = getSigmaPts(S) 
		print("W: \n" + str(W.shape) + "\n\n");

		X_propagate = propagateState(X, W) # (X_(k+1))
		print("X_propagate: \n" + str(X_propagate.shape) + "\n\n");
		Y_i = processA(X_propagate, t_prev, t_cur);
		print("Y_i: \n" + str(Y_i.shape) + "\n\n");

		Y_bar = getMean(Y_i)
		print("Y_bar: \n" + str(Y_bar.shape) + "\n\n");

		W_devia = getWDeviation(W, Y_bar);
		print("W_devia: \n" + str(W_devia.shape) + "\n\n");

		P_kbar = W_devia @ W_devia.T / 12;
		print("P_kbar: \n" + str(P_kbar.shape) + "\n\n");

		Z_i = getMeasureEstimate(X_propagate)
		print("Z_i: \n" + str(Z_i.shape) + "\n\n");

		Z_kbar = getBaryMean(Z_i);
		print("Z_kbar: \n" + str(Z_kbar.shape) + "\n\n");
		# quaterion_actual = Quaternion();
		# quaterion_actual.from_axis_angle([yaw[i], pitch[i], roll[i]]);
		Z_actual = np.zeros((6, 1))
		Z_actual[0:3, 0] = gyro_dig[:,i];
		Z_actual[3:6, 0] = np.array([ax[i], ay[i], az[i]]) # TODO X(q, ome) ? 
		print("Z_actual: \n" + str(Z_actual.shape) + "\n\n");

		Z_kbar = Z_kbar.reshape((6,1))
		# Z_deiva = getWDeviation(Z_i, Z_kbar)
		Z_deiva = Z_i - Z_kbar
		print("Z_deiva: \n" + str(Z_deiva.shape) + "\n\n");

		P_zz = Z_deiva @ Z_deiva.T / 12;
		print("P_zz: \n" + str(P_zz.shape) + "\n\n");

		innovation = Z_actual - Z_kbar;
		print("innovation: \n" + str(innovation.shape) + "\n\n");

		P_vv = P_zz + R;
		P_xz = W_devia @ Z_deiva.T / 12;

		K_k = P_xz @ P_vv.T

		knk = K_k @ innovation;
		dq = Quaternion();
		dq.from_axis_angle(knk[0:3, 0])
		X_k = np.zeros((7, 1)) 

		new_q = dq.__mul__(Quaternion(Y_bar[0,0], Y_bar[1:4,0]));
		X_k[0, 0] = new_q.scalar();
		X_k[1:4, 0] = new_q.vec();
		X_k[4:7, 0] = Y_bar[4:7, 0] + knk[3:6, 0];

		P_k = P_kbar - K_k @ P_vv @ K_k.T
		P_prev = P_k
		X = X_k
		t_prev = t_cur;
		rpy = new_q.euler_angles()
		roll_.append(rpy[0])
		pitch_.append(rpy[1])
		yaw_.append(rpy[2])
	# roll, pitch, yaw are numpy arrays of length T
	return roll_, pitch_, yaw_

estimate_rot();