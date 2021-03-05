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
		q_w = Quaternion();
		q_w.from_axis_angle(W[0:3, i]);
		omega_k = X[4:7, 0] + W[3:6, i]
		q_k = q_w.__mul__(q_x);
		q_k.normalize();
		# print("q_k: " + str(q_k))
		# print("q_x: " + str(q_x))
		# print("q_w: " + str(q_w))

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
		q_ome.from_axis_angle(X_p[4:7, i] * delta_t);
		q_cur = Quaternion(X_p[0,i], X_p[1:4, i]);
		q_predict = q_ome.__mul__(q_cur);
		q_predict.normalize();
		X_p[0, i] = q_predict.scalar();
		X_p[1:4, i] = q_predict.vec();
	return X_p;

def getBaryMean(B):
	return np.mean(B, axis = 1)

def getQtError(q, q_barinv): # q_bar is Quaternion instance, q is vector 

	q_i = Quaternion(q[0], q[1:4])
	ret = q_i.__mul__(q_barinv)
	ret.normalize()
	return ret


def getMeanQuaternion(q_cur):
	q = Quaternion(1, [0,0,0]);
	q_prev = Quaternion(-1, [-1,-1,-1]);
	threshould = 0.01;
	ct = 0;
	err_ = [q.scalar() - q_prev.scalar(), (q.vec() - q_prev.vec())[0], (q.vec() - q_prev.vec())[1], (q.vec() - q_prev.vec())[2]];
	while(LA.norm(err_) > threshould or ct < 50):
		q_inv = q.inv();
		# print("err: " + str(LA.norm(err_)))
		# print("qinv: " + str(q_inv))
		E_i = np.zeros((3, 12));
		for i in range(q_cur.shape[-1]):
			# print("erQcur: " + str(q_cur[0:4, i]))
			er = getQtError(q_cur[0:4, i], q_inv);
			# print("erQ: " + str(er))
			E_i[:, i] = er.axis_angle();

		e_bar = getBaryMean(E_i);

		e_q = Quaternion();
		e_q.from_axis_angle(e_bar)
		q_prev = q;
		q = e_q.__mul__(q);
		q.normalize();
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

def getWDeviation(Y, Y_i):
	W_d = np.zeros((6, 12));
	q_bar = Quaternion(Y_i[0,0],Y_i[1:4, 0]);
	q_barinv = q_bar.inv()
	for i in range(Y.shape[-1]):
		new_col = np.zeros((6, 1));
		err_q = getQtError(Y[0:4, i], q_barinv);
		new_col[0:3, 0] = err_q.axis_angle(); # the scalar part might be wrong
		new_col[3:6, 0] = Y[4:7, i] - Y_i[4:7, 0];
		W_d[:, i] = new_col[:, 0];

	return W_d;

def rotateVector(q, g):
	q_i = Quaternion(q[0], q[1:4]);
	return q_i.inv()*g*q_i;

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

	gx = digitalToAnalog(gyro[1, :], 208.0, 373.7);
	gy = digitalToAnalog(gyro[2, :], 208.0, 375.7);
	gz = digitalToAnalog(gyro[0, :], 200.0, 370.1);

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

	patch = np.zeros((3, 84));
	quaterions = np.hstack((quaterions, patch))

	yaw, pitch, roll = accToRPY(-ax, -ay, az);
	bebug = False;

	# your code goes here
	roll_ = [0]
	pitch_ = [0]
	yaw_ = [0]
	X = np.zeros((7, 1))
	X[0, 0] = 1;
	P_prev = np.eye(6,6)
	Q = 0.01 * np.eye(6,6)
	R = 0.01 * np.eye(6,6)
	t_prev = t[0];
	for i in range(1, T):
		if(bebug):
			print("P_prev: \n" + str(P_prev + Q));
			print("state: \n" + str(X.T))
		
		t_cur = t[i];
		S = CholeskyMatrix(np.sqrt(12) * (P_prev + Q))
		if(bebug):
			print("S: \n" + str(S.shape) + "\n");
			print(str(S) + "\n");
		
		W = np.hstack((S, -S))
		if(bebug):
			print("W: \n" + str(W.shape) + "\n");
			print(str(W) + "\n");


		X_propagate = propagateState(X, W) # (X_(k+1))
		if(bebug):
			print("X_propagate: \n" + str(X_propagate.shape) + "\n");
			print(str(X_propagate) + "\n");


		Y_i = processA(X_propagate, t_prev, t_cur);
		if(bebug):
			print("Y_i: \n" + str(Y_i.shape) + "\n");
			print(str(Y_i) + "\n");


		Y_bar = getMean(Y_i)
		if(bebug):
			print("Y_bar: \n" + str(Y_bar.shape) + "\n");
			print(str(Y_bar) + "\n");


		W_devia = getWDeviation(X_propagate, Y_bar);
		if(bebug):
			print("W_devia: \n" + str(W_devia.shape) + "\n");
			print(str(W_devia) + "\n");

		W_p_devia = getWDeviation(Y_i, Y_bar);
		if(bebug):
			print("W_p_devia: \n" + str(W_p_devia.shape) + "\n");
			print(str(W_p_devia) + "\n");


		P_kbar = W_devia @ W_devia.T / 12;
		if(bebug):
			print("P_kbar: \n" + str(P_kbar.shape) + "\n");
			print(str(P_kbar) + "\n");


		Z_i = getMeasureEstimate(X_propagate)
		if(bebug):
			print("Z_i: \n" + str(Z_i.shape));
			print(str(Z_i) + "\n");


		Z_kbar = getBaryMean(Z_i);
		Z_kbar = Z_kbar.reshape((6,1))
		if(bebug):
			print("Z_kbar: \n" + str(Z_kbar.shape))
			print(str(Z_kbar.T) + "\n");

		Z_actual = np.zeros((6, 1))
		Z_actual[0:3, 0] = np.array([gx[i], gy[i], gz[i]])
		Z_actual[3:6, 0] = np.array([ax[i], ay[i], az[i]]) # TODO X(q, ome) ? 
		if(bebug):
			print("Z_actual: \n" + str(Z_actual.shape));
			print(str(Z_actual.T) + "\n");

		# Z_deiva = getWDeviation(Z_i, Z_kbar)
		Z_deiva = Z_i - Z_kbar
		if(bebug):
			print("Z_deiva: \n" + str(Z_deiva.shape));
			print(str(Z_deiva) + "\n");


		P_zz = Z_deiva @ Z_deiva.T / 12;
		if(bebug):
			print("P_zz: \n" + str(P_zz.shape) + "\n");
			print(str(P_zz) + "\n");

		innovation = Z_actual - Z_kbar;
		if(bebug):
			print("innovation: \n" + str(innovation.shape) + "\n");
			print(str(innovation) + "\n");

		P_vv = P_zz + R;
		if(bebug):
			print("P_vv: \n" + str(P_vv.shape) + "\n");
			print(str(P_vv) + "\n");


		P_xz = W_p_devia @ Z_deiva.T / 12;
		if(bebug):
			print("P_xz: \n" + str(P_xz.shape) + "\n");
			print(str(P_xz) + "\n");


		K_k = P_xz @ LA.inv(P_vv)
		if(bebug):
			print("K_k: \n" + str(K_k.shape) + "\n");
			print(str(K_k) + "\n");

		# innovation = np.zeros((6, 1))
		K_innno = K_k @ innovation;
		if(bebug):
			print("K_innno: \n" + str(K_innno.shape) + "\n");
		
		kq = Quaternion();
		kq.from_axis_angle(K_innno[0:3, 0]);

		Y_bar[4:7, 0] += K_innno[3:6, 0];
		qua_y = Quaternion(Y_bar[0, 0], Y_bar[1:4, 0]);
		qua_update = kq.__mul__(qua_y);
		qua_update.normalize();
		Y_bar[0, 0] = qua_update.scalar();
		Y_bar[1:4, 0] = qua_update.vec();
		X_k = Y_bar

		P_k = P_kbar - K_k @ P_vv @ K_k.T
		P_prev = P_k
		X = X_k
		t_prev = t_cur;

		rpy = kq.euler_angles()
		roll_.append(rpy[0])
		pitch_.append(rpy[1])
		yaw_.append(rpy[2])


	fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
	ax1.plot(t, yaw_, label = "yaw")
	ax2.plot(t, pitch_, label = "pitch")
	ax3.plot(t, roll_, label = "roll")
	ax1.plot(t, quaterions[2,:], label = "true yaw")
	ax2.plot(t, quaterions[1,:], label = "true pitch")
	ax3.plot(t, quaterions[0,:], label = "true roll")
	ax1.legend()
	ax2.legend()
	ax3.legend()
	ax1.set_title('yaw')
	ax2.set_title('pitch')
	ax3.set_title('roll')
	plt.show();
	# roll, pitch, yaw are numpy arrays of length T
	return roll_, pitch_, yaw_

estimate_rot();