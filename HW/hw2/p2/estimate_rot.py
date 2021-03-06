import numpy as np
from scipy import io
from quaternion import Quaternion
import matplotlib.pyplot as plt
import math
from numpy import linalg as LA
import os

#data files are numbered on the server.
#for exmaple imuRaw1.mat, imuRaw2.mat and so on.
#write a function that takes in an input number (1 through 6)
#reads in the corresponding imu Data, and estimates
#roll pitch and yaw using an unscented kalman filter

def CholeskyMatrix(P_prev):
	return np.linalg.cholesky(P_prev);

def propagateState(X, W):
	q_x = Quaternion(X[0,0], X[1:4, 0]);
	q_x.normalize();
	X_p = np.zeros((7, 12))
	for i in range(W.shape[-1]):
		q_w = Quaternion();
		q_w.from_axis_angle(W[0:3, i]);
		q_k = q_w * q_x;
		q_k.normalize();
		# print("q_k: " + str(q_k))
		# print("q_x: " + str(q_x))
		# print("q_w: " + str(q_w))
		X_p[0:4, i] = q_k.q
		X_p[4:7, i] = X[4:7, 0] + W[3:6, i];
	return X_p;

def processA(X_p, prev, timeStamp):
	# Todo
	delta_t = (timeStamp - prev) / 1000.0;
	for i in range(X_p.shape[-1]):
		q_ome = Quaternion();
		q_ome.from_axis_angle(X_p[4:7, i] * delta_t);
		q_cur = Quaternion(X_p[0,i], X_p[1:4, i]);
		q_predict = q_ome * q_cur;
		q_predict.normalize();
		X_p[0:4, i] = q_predict.q
	return X_p;

def getBaryMean(B):
	return np.mean(B, axis = 1)

def getQtError(q, q_barinv): # q_bar is Quaternion instance, q is vector 

	q_i = Quaternion(q[0], q[1:4])
	ret = q_i * q_barinv
	ret.normalize()
	return ret


def getMeanQuaternion(q_cur, q_bar):
	q = Quaternion(q_bar[0], q_bar[1:4]);
	prev_e = 10
	threshould = 0.0001;
	ct = 0;
	e_bar = np.array([0,0,0])
	while(np.abs(LA.norm(e_bar) -  prev_e) > threshould and ct < 50):
		prev_e = LA.norm(e_bar);

		E_i = np.zeros((3, 12));
		for i in range(q_cur.shape[-1]):
			er = getQtError(q_cur[0:4, i], q.inv());
			E_i[:, i] = er.axis_angle();

		e_bar = getBaryMean(E_i);

		e_q = Quaternion();
		e_q.from_axis_angle(e_bar)
		q = e_q * q;
		q.normalize();

		ct += 1;
	# print("err: " + str(LA.norm(e_bar)))
	# print("ct: " + str(ct))

	return q;

def getMean(Y_i, X_q):
	Y_mean = np.zeros((7,1));
	Y_mean[4:7, 0] = getBaryMean(Y_i[4:7, :])
	Y_mean_q = getMeanQuaternion(Y_i[0:4, :], X_q);
	Y_mean[0:4, 0] = Y_mean_q.q
	return Y_mean;

def getWDeviation(Y, Y_bar):
	W_d = np.zeros((6, 12));
	q_bar = Quaternion(Y_bar[0, 0],Y_bar[1:4, 0]);
	for i in range(Y.shape[-1]):
		new_col = np.zeros((6, 1));
		err_q = getQtError(Y[0:4, i], q_bar.inv());
		new_col[0:3, 0] = err_q.axis_angle(); 
		new_col[3:6, 0] = Y[4:7, i] - Y_bar[4:7, 0];
		W_d[:, i] = new_col[:, 0];

	return W_d;

def rotateVector(q, g):
	q_i = Quaternion(q[0], q[1:4]);
	return q_i.inv() * g * q_i;

def getMeasureEstimate(Y_i):
	Z = np.zeros((6,12));
	g_ = Quaternion(0.0, [0.0, 0.0, 9.8]);
	for i in range(Y_i.shape[-1]):
		z_i = np.zeros((6, 1))
		z_i[3:6, 0] = Y_i[4:7, i];
		z_i[0:3, 0] = rotateVector(Y_i[0:4, i], g_).vec(); # might be wrong
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
	imu = io.loadmat(os.path.join(os.path.dirname(__file__), "imu/imuRaw" + str(data_num) + ".mat"))
	accel = imu['vals'][0:3,:]
	gyro = imu['vals'][3:6,:]
	T = np.shape(imu['ts'])[1]

	gx = digitalToAnalog(gyro[1, :], 208.0, 373.7); # 208
	gy = digitalToAnalog(gyro[2, :], 208.0, 375.7); # 208
	gz = digitalToAnalog(gyro[0, :], 200.0, 370.1); # 200

	ax = digitalToAnalog(accel[0, :], 32.5, 510.0);
	ay = digitalToAnalog(accel[1, :], 32.5, 498.0);
	az = digitalToAnalog(accel[2, :], 29.5, 516.0);

	t = imu['ts'].T;
	vicon = io.loadmat('vicon/viconRot'+str(data_num)+'.mat')

	quaterions = [];
	for i in range(vicon['rots'].shape[-1]):
		rot = vicon['rots'][:,:,i].reshape(3,3)
		q = rotationMatrixToEulerAngles(rot)
		quaterions.append(q)
	quaterions = np.asarray(quaterions);
	quaterions = quaterions.T;

	patch = np.zeros((3, T - quaterions.shape[1]));
	quaterions = np.hstack((quaterions, patch))

	yaw, pitch, roll = accToRPY(-ax, -ay, az);
	bebug = False;

	# your code goes here
	roll_ = [0]
	pitch_ = [0]
	yaw_ = [0]
	ax_p = [0]
	ay_p = [0]
	az_p = [0]
	ox = [0]
	oy = [0]
	oz = [0]
	p_ome = []
	p_ori = []
	X = np.zeros((7, 1))
	X[0, 0] = 1;
	P_prev = np.eye(6,6)
	Q = np.diag([0.05, 0.05, 0.05, 0.85, 0.85, 0.85]).astype(float) # [0.05, 0.05, 0.05, 0.8, 0.8, 0.8]
	R = 7 * np.diag([1, 1, 1, 50, 50, 50]).astype(float) # 5 * np.diag([1, 1 , 1, 15, 15, 15])
	# print("Q: \n" + str(Q))

	t_prev = t[0];
	for i in range(1, T):
		t_cur = t[i];

		# if(i % 200 == 0):
		# 	print("Trace: \n" + str(np.trace(P_prev + Q * (t_cur - t_prev))));

		if(bebug):
			print("P_prev: \n" + str(P_prev + Q));
			print("state: \n" + str(X.T))
			input();
		
		S = CholeskyMatrix(np.sqrt(6) * (P_prev + Q))
		if(bebug):
			print("S: \n" + str(S.shape) + "\n");
			print(str(S) + "\n");
			input();
		
		W = np.hstack((S, -S))
		if(bebug):
			print("W: \n" + str(W.shape) + "\n");
			print(str(W) + "\n");
			input();

		X_propagate = propagateState(X, W) # (orientation, ome)
		if(bebug):
			print("X_propagate: \n" + str(X_propagate.shape) + "\n");
			print(str(X_propagate) + "\n");
			input();

		Y_i = processA(X_propagate, t_prev, t_cur);
		if(bebug):
			print("Y_i: \n" + str(Y_i.shape) + "\n");
			print(str(Y_i) + "\n");
			input();

		Y_bar = getMean(Y_i, X[0:4, 0])
		if(bebug):
			print("Y_bar: \n" + str(Y_bar.shape) + "\n");
			print(str(Y_bar) + "\n");
			input();

		W_p_devia = getWDeviation(Y_i, Y_bar);
		if(bebug):
			print("W_p_devia: \n" + str(W_p_devia.shape) + "\n");
			print(str(W_p_devia) + "\n");
			input();
		
		P_kbar = W_p_devia @ W_p_devia.T / 12;
		if(bebug):
			print("P_kbar: \n" + str(P_kbar.shape) + "\n");
			print(str(P_kbar) + "\n");
			input();
		
		Z_i = getMeasureEstimate(Y_i) # (orientation, ome)
		if(bebug):
			print("Z_i: \n" + str(Z_i.shape));
			print(str(Z_i) + "\n");
			input();

		Z_kbar = getBaryMean(Z_i);
		Z_kbar = Z_kbar.reshape((6,1))
		if(bebug):
			print("Z_kbar: \n" + str(Z_kbar.shape))
			print(str(Z_kbar.T) + "\n");
			input();

		

		Z_actual = np.zeros((6, 1))
		Z_actual[0:3, 0] = np.array([ax[i], ay[i], az[i]])
		Z_actual[3:6, 0] = np.array([gx[i], gy[i], gz[i]])
		if(bebug):
			print("Z_actual: \n" + str(Z_actual.shape));
			print(str(Z_actual.T) + "\n");
			input();
		
		Z_deiva = Z_i - Z_kbar
		if(bebug):
			print("Z_deiva: \n" + str(Z_deiva.shape));
			print(str(Z_deiva) + "\n");
			input();

		P_zz = Z_deiva @ Z_deiva.T / 12;
		if(bebug):
			print("P_zz: \n" + str(P_zz.shape) + "\n");
			print(str(P_zz) + "\n");
			input();

		innovation = Z_actual - Z_kbar;
		if(bebug):
			print("innovation: \n" + str(innovation.shape) + "\n");
			print(str(innovation) + "\n");
			input();

		P_vv = P_zz + R;
		if(bebug):
			print("P_vv: \n" + str(P_vv.shape) + "\n");
			print(str(P_vv) + "\n");
			input();

		P_xz = W_p_devia @ Z_deiva.T / 12;
		if(bebug):
			print("P_xz: \n" + str(P_xz.shape) + "\n");
			print(str(P_xz) + "\n");
			input();

		K_k = P_xz @ LA.inv(P_vv)
		# breakpoint()

		if(bebug):
			print("K_k: \n" + str(K_k.shape) + "\n");
			print(str(K_k) + "\n");
			input();

		K_innno = K_k @ innovation;
		if(bebug):
			print("K_innno: \n" + str(K_innno.shape) + "\n");
			input();
		
		kq = Quaternion();
		kq.from_axis_angle(K_innno[3:6, 0]);

		Y_bar[4:7, 0] += K_innno[0:3, 0];
		
		qua_y = Quaternion(Y_bar[0, 0], Y_bar[1:4, 0]);
		qua_y = kq * qua_y;
		qua_y.normalize();
		Y_bar[0:4, 0] = qua_y.q;
		X_k = Y_bar

		P_k = P_kbar - K_k @ P_vv @ K_k.T

		P_prev = P_k
		P_k_dia = np.diag(P_k)
		qua_y_P_k = Quaternion(P_k_dia[0], P_k_dia[1:4])
		X = X_k
		t_prev = t_cur;

		rpy = qua_y.euler_angles()
		roll_.append(-rpy[0])
		pitch_.append(-rpy[1])
		yaw_.append(rpy[2])
		ax_p.append(Z_kbar[0,0])
		ay_p.append(Z_kbar[1,0])
		az_p.append(Z_kbar[2,0])
		ox.append(-Y_bar[4,0])
		oy.append(-Y_bar[5,0])
		oz.append(Y_bar[6,0])
		p_ori.append(qua_y_P_k.euler_angles());
		p_ome.append(P_k_dia[3:6]);


	p_ori = np.asarray(p_ori);
	p_ome = np.asarray(p_ome);
	p_ori = p_ori.T
	p_ome = p_ome.T


	fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
	fig.suptitle('filtered orientation')
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

	fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
	fig.suptitle('filtered gravitational direction')
	ax1.plot(t, ax, label = "ax")
	ax2.plot(t, ay, label = "ay")
	ax3.plot(t, az, label = "az")
	ax1.plot(t, ax_p, label = "ax_Z")
	ax2.plot(t, ay_p, label = "ay_Z")
	ax3.plot(t, az_p, label = "az_Z")
	ax1.legend()
	ax2.legend()
	ax3.legend()
	ax1.set_title('ax')
	ax2.set_title('ay')
	ax3.set_title('az')
	plt.show();

	fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
	fig.suptitle('filtered angular velocity')
	ax1.plot(t, ox, label = "ox")
	ax2.plot(t, oy, label = "oy")
	ax3.plot(t, oz, label = "oz")
	ax1.plot(t, gx, label = "true")
	ax2.plot(t, gy, label = "true")
	ax3.plot(t, gz, label = "true")
	ax1.legend()
	ax2.legend()
	ax3.legend()
	ax1.set_title('ome_x')
	ax2.set_title('ome_y')
	ax3.set_title('ome_z')
	plt.show();

	fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
	fig.suptitle('filtered angular velocity')
	ax1.plot(t[:-1], p_ori[0,:], label = "ox")
	ax2.plot(t[:-1], p_ori[1,:], label = "oy")
	ax3.plot(t[:-1], p_ori[2,:], label = "oz")
	ax1.plot(t[:-1], p_ome[0,:], label = "true")
	ax2.plot(t[:-1], p_ome[0,:], label = "true")
	ax3.plot(t[:-1], p_ome[0,:], label = "true")
	ax1.legend()
	ax2.legend()
	ax3.legend()
	ax1.set_title('ome_x')
	ax2.set_title('ome_y')
	ax3.set_title('ome_z')
	plt.show();

	# roll, pitch, yaw are numpy arrays of length T
	return roll_, pitch_, yaw_

estimate_rot();