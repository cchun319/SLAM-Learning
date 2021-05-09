

import numpy as np
import matplotlib.pyplot as plt
import os, sys, pickle, math
from copy import deepcopy

v = np.array([1, 1, 1, -1, -1, -1, 0, 0]);
h = np.array([1, -1, 0, -1, 0, 1, 1, -1]);
class Dijkstra:
	def __init__(s, occumap, source):
		s.costMap = np.zeros((384,384));

		sgx, sgy = s.worldToGrid(source[0], source[1])
		s.unsx = [];
		s.unsy = [];

		s.uncertainty = np.sum(occumap < 0);

		s.infoMap = np.asarray([0] * 384 ** 2);

		s.range = int(np.ceil(3.5 // 0.05))
		s.angles = np.arange(0, 360) * 180.0 / np.pi;

		s.shift_x = (s.range * np.cos(s.angles)).astype(int);
		s.shift_y = (s.range * np.sin(s.angles)).astype(int);

		s.radius = int(np.ceil(0.08 // 0.05)) + 2 # enlarge radius = the half size of the robot
		# s.radius = 5 # enlarge radius = the half size of the robot

		s.enlargeOccu(occumap)

		# q (encode index, cost)
		s.q = [];
		s.visited = set();
		s.source = s.encode(sgx, sgy)
		s.q.append(s.source)
		
		s.cost = np.asarray([1e20] * 384 ** 2);
		s.parent = np.asarray([-1] * 384 ** 2);
		s.parent[s.source] = s.source;
	def worldToGrid(s, x, y):
		gx = (-x) // 0.05 + 184
		gy = (-y) // 0.05 + 184		
		return gx, gy;

	def planning(s):
		dist = 0
		# print("dest: " + str(s.dest))
		while len(s.q) > 0:
			qsize = len(s.q)
			while qsize > 0:
				cur = s.q[0]
				# print("cur: " + str(cur))
				s.q.pop(0)
				qsize -= 1;
				if cur in s.visited:
					continue;
				s.visited.add(cur);
				# print('add' + str(cur))
				cx, cy = s.decode(cur);
				
				for i in range(8):
					nx = cx + v[i];
					ny = cy + h[i];
					ncd = s.encode(nx, ny);

					if nx < 0 or ny < 0 or nx >= 384 or ny >= 384 or not s.isValid(nx, ny) or ncd in s.visited:
						# print("con")
						continue; 

					if s.cost[ncd] > dist + 1:
						s.cost[ncd] = dist + 1;
						s.parent[ncd] = cur

					s.q.append(ncd);

	def getPath(s):
		px = []
		py = []
		cx, cy = s.decode(s.dest);
		px.insert(0, int(cx))
		py.insert(0, int(cy))

		par = s.dest

		while par != s.parent[par]:
			# print("mpt")
			# print(px[0])
			# print(py[0])
			par = s.parent[par]	
			cx, cy = s.decode(par);
			px.insert(0, int(cx))
			py.insert(0, int(cy))

		print("raw path len: " + str(len(px)))	

		s.unsx = np.asarray(px);
		s.unsy = np.asarray(py);
		sx, sy = s.smoothing(px, py);
		# print("path len: " + str(len(sx)))	
		return s.tf_to_world(sx, sy);

	def enlargeOccu(s, map):

		x, y = np.where(map > 0)
		s.map = deepcopy(map)
		for i in range(len(x)):
			# print(str(x[i]) + " and " + str(y[i]))
			for k in range(max(0, x[i] - s.radius), min(map.shape[0], x[i] + s.radius + 1)):
				for l in range(max(0, y[i] - s.radius), min(map.shape[1], y[i] + s.radius + 1)):
					s.map[k][l] = 1;

	def isValid(s, x, y):
		# print("grid: " + str(x) + ' ' + str(y) + " is " + str(s.map[x][y]))
		return s.map[x][y] == 0

	def findDest(s):
		# return the destination that brings the most information
		# rx, ry = np.random.randint(384), np.random.randint(384)

		# while not s.isValid(rx, ry):
		# 	rx, ry = np.random.randint(384), np.random.randint(384)
		
		# rx = int((-1 + 10) // 0.05)
		# ry = int((2 + 10) // 0.05)
		print("looking for dest")
		fx, fy = np.where(s.map == 0) # free grid

		for i in range(len(fx)):
			endx = fx[i] + s.shift_x 
			endy = fy[i] + s.shift_y
			range_x = np.linspace([fx[i]]*len(s.shift_x), endx, num = 70, endpoint=True).astype(int)
			range_y = np.linspace([fy[i]]*len(s.shift_y), endy, num = 70, endpoint=True).astype(int)
			range_x = np.clip(range_x, 0, 383)
			range_y = np.clip(range_y, 0, 383)

			raw_score = s.map[range_x, range_y]
			score = raw_score[~np.any(raw_score > 0, axis = 1)]

			s.infoMap[s.encode(fx[i], fy[i])] = -np.sum(score)

		H_map = s.infoMap - 100 * s.cost # 
		s.costMap = np.reshape(H_map, (384, 384))

		s.dest = np.argmax(H_map)
		if H_map[s.dest] == 0 or s.cost[s.dest] == 1e20:
			print("exploration done")
			s.dest = -1;
		else:
			print("dest find: " + str(s.decode(s.dest)) + ' cost: ' + str(H_map[s.dest]))
			print("from: " + str(s.decode(s.source)))
			print(s.isValid(s.decode(s.dest)[0], s.decode(s.dest)[1]))
			# plt.imshow(np.reshape(H_map, (384, 384)));
			# plt.show()


	def collisionFree(s, x1, y1, x2, y2):
		linepts_x, linepts_y = s.getLine(x1, y1, x2, y2);
		obs = s.map[linepts_x.astype(int), linepts_y.astype(int)]
		# np.all((obs == 0))
		return not obs.any();

	def getLine(s, x1, y1, x2, y2):
		return np.linspace(x1, x2), np.linspace(y1, y2); 

	def encode(s, x, y):
		return int(x * 384 + y);

	def decode(s, code):
		return code // 384, code % 384; 

	def smoothing(s, px, py):
	
		# shortcut

		nx = []
		ny = []
		nx.append(px[-1])
		ny.append(py[-1])

		start = len(px) - 1;

		while start > 0:
			con = 0;
			print('start: ' + str(start))
			while con < start - 1 and not s.collisionFree(px[con], py[con], nx[0], ny[0]):
				# print(str(con) + ' start: ' + str(start))
				con += 1

			nx.insert(0, px[con])
			ny.insert(0, py[con])

			start = con

		print("smoothing: " + str(len(nx)))
		return nx, ny;
	def tf_to_world(s, px, py):
		# print(type(px))
		# print(py)
		tx = -(np.asarray(px) - 184.0) * 0.05;
		ty = -(np.asarray(py) - 184.0) * 0.05;
		return tx, ty;

	def updateUncertainty(s):
		s.uncertainty.append(len(np.where(s.map < 0)));

	def explore(s):
		s.planning();
		s.findDest();
		if s.dest < 0:
			return [], [];
		else:
			return s.getPath();


if __name__ == '__main__':
	

	m = np.zeros((384,384), dtype=np.int8)

	m[150: 270, 150: 270] = 1

	planner = Dijkstra(m, np.array([-10, -10]));
	planner.dest = 384**2 - 1

	planner.planning();

	x, y = planner.getPath();

	plt.imshow(m);

	plt.plot(x, y, 'r.', markersize = 3);
	plt.show()