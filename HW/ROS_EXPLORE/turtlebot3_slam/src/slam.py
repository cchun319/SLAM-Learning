# Pratik Chaudhari (pratikac@seas.upenn.edu)
#!/usr/bin/env python

import os, sys, pickle, math
from copy import deepcopy

from scipy import io
import numpy as np
import matplotlib.pyplot as plt

# from load_data import load_lidar_data, load_joint_data, joint_name_to_index
from utils import *

import logging
logger = logging.getLogger()
logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))

rot = get_so2(-np.pi/2)

class map_t:
    """
    This will maintain the occupancy grid and log_odds. You do not need to change anything
    in the initialization
    """
    def __init__(s, resolution=0.05):
        s.resolution = resolution
        s.mapSize = 20;
        s.xmin, s.xmax = -s.mapSize, s.mapSize
        s.ymin, s.ymax = -s.mapSize, s.mapSize
        s.szx = int(np.ceil((s.xmax-s.xmin)/s.resolution))
        s.szy = int(np.ceil((s.ymax-s.ymin)/s.resolution))
        s.numOfGrid = s.szx * s.szy;
        # binarized map and log-odds
        s.cells = np.zeros((s.szx, s.szy), dtype=np.int8)
        s.log_odds = np.zeros(s.cells.shape, dtype=np.float64) 

        # value above which we are not going to increase the log-odds
        # similarly we will not decrease log-odds of a cell below -max
        s.log_odds_max = 5e6
        # number of observations received yet for each cell
        s.num_obs_per_cell = np.zeros(s.cells.shape, dtype=np.uint64)

        # we call a cell occupied if the probability of
        # occupancy P(m_i | ... ) is >= occupied_prob_thresh
        s.occupied_prob_thresh = 0.6
        s.log_odds_thresh = np.log(s.occupied_prob_thresh/(1-s.occupied_prob_thresh)) # log(1.2)

    def grid_cell_from_xy(s, x, y):
        """
        x and y are 1-dimensional arrays, compute the cell indices in the map corresponding
        to these (x,y) locations. You should return an array of shape 2 x len(x). Be
        careful to handle instances when x/y go outside the map bounds, you can use
        np.clip to handle these situations.
        """
        #### TODO: XXXXXXXXXXX
        numOfSample = len(x);
        x = np.clip(x, s.xmin, s.xmax);
        y = np.clip(y, s.ymin, s.ymax);
        r = (x - s.xmin) // s.resolution;
        c = (y - s.ymin) // s.resolution;
        # print("raw pos: " + str(x) + " || " + str(y))
        # print("converted pos: " + str(r) + " || " + str(c))
        r = r.astype(int);
        c = c.astype(int);
        return r, c;

class slam_t:
    """
    s is the same as self. In Python it does not really matter
    what we call self, s is shorter. As a general comment, (I believe)
    you will have fewer bugs while writing scientific code if you
    use the same/similar variable names as those in the mathematical equations.
    """
    def __init__(s, resolution=0.05, Q=1e-3*np.eye(3),
                 resampling_threshold=0.3):
        s.init_sensor_model()

        # dynamics noise for the state (x,y,yaw)
        s.Q = np.array([2e-4, 2e-4, 1e-3]) * np.eye(3) 

        s.cur = np.zeros((3,1), dtype = np.float64)
        s.curgrid = np.zeros((2, 1), dtype = np.uint64)

        # we resample particles if the effective number of particles
        # falls below s.resampling_threshold*num_particles
        s.resampling_threshold = resampling_threshold

        # initialize the map
        s.map = map_t(resolution)

        s.l = 0.16

        s.prevTime = -1;
        s.curTime = -1;

    def read_data(s, src_dir, idx=0, split='train'):
        """
        src_dir: location of the "data" directory
        """
        logging.info('> Reading data')
        s.idx = idx
   


    def init_sensor_model(s):
        # lidar height from the ground in meters
        s.lidar_height = 0.15

        # dmin is the minimum reading of the LiDAR, dmax is the maximum reading
        s.lidar_dmin = 0.12
        s.lidar_dmax = 3.5
        s.lidar_angular_resolution = 1;
        # these are the angles of the rays of the Hokuyo
        s.lidar_angles = np.arange(0, 360, s.lidar_angular_resolution) * np.pi / 180.0

        # sensor model lidar_log_odds_occ is the value by which we would increase the log_odds
        # for occupied cells. lidar_log_odds_free is the value by which we should decrease the
        # log_odds for free cells (which are all cells that are not occupied)
        s.lidar_log_odds_occ = np.log(9) # 0.9
        s.lidar_log_odds_free = np.log(1/9.)

    def init_particles(s, n=100, p=None, w=None, t0=0):
        """
        n: number of particles
        p: xy yaw locations of particles (3xn array)
        w: weights (array of length n)
        """
        s.n = n
        s.p = deepcopy(p) if p is not None else np.zeros((3,s.n))
        # s.p[0,:] = -3;
        # s.p[1,:] = 1;
        s.w = deepcopy(w) if w is not None else np.ones(n)/float(s.n)

    @staticmethod
    def stratified_resampling(p, w):
        """
        resampling step of the particle filter, takes p = 3 x n array of
        particles with w = 1 x n array of weights and returns new particle
        locations (number of particles n remains the same) and their weights
        """
        #### TODO: XXXXXXXXXXX

        numOfParticle = p.shape[1];
        newW = np.ones_like(w) / numOfParticle;
        newP = np.zeros_like(p, dtype=np.float64);
        i = 0
        c = w[0];
        r = np.random.uniform(0.0, 1.0 / numOfParticle)

        for m in np.arange(numOfParticle):
            u = r + (m - 1) / numOfParticle;
            while(u > c):
                i = i + 1;
                c = c + w[i % numOfParticle];

            newP[:, m] = p[:, i % numOfParticle];

        return newP, newW;

    @staticmethod
    def log_sum_exp(w):
        return w.max() + np.log(np.exp(w-w.max()).sum())

    def rays2world(s, p, d, angles=None):
        """
        p is the pose of the particle (x,y,yaw)
        angles = angle of each ray in the body frame (this will usually be simply s.lidar_angles for the different lidar rays)
        d = is an array that stores the distance of along the ray of the lidar, for each ray (the length of d has to be equal to that of angles, this is s.lidar[t]['scan'])
        Return an array 2 x num_rays which are the (x,y) locations of the end point of each ray
        in world coordinates
        """

        # make sure each distance >= dmin and <= dmax, otherwise something is wrong in reading
        # the data

        
        d = np.asarray(d)
        angles = angles[(d >= s.lidar_dmin) & (d <= s.lidar_dmax)];
        d = d[(d >= s.lidar_dmin) & (d <= s.lidar_dmax)];
        # 1. from lidar distances to points in the LiDAR frame, #(r, the) -> (x, y, 0) in lidar frame
        scan_in_lidar = np.zeros((3, len(angles)), dtype=float);
        scan_in_lidar[0,:] = d * np.cos(angles);
        scan_in_lidar[1,:] = d * np.sin(angles); # (3 x n)
        scan_in_lidar_H = make_homogeneous_coords_3d(scan_in_lidar);

        # 2. from LiDAR frame to the body frame
        R_bodyToLidar = euler_to_se3(np.pi, 0, np.pi/2, np.array([0.032, 0, 0])); # 4 x 4
        scan_in_body = np.matmul(R_bodyToLidar, scan_in_lidar_H); 


        # 3. from body frame to world frame (x, y, s.head_height, the)
        R_worldToBody = euler_to_se3(0, 0, p[2], np.array([p[0], p[1], s.lidar_height]));
        scan_in_world = np.matmul(R_worldToBody,scan_in_body);

        scan_in_world_planar = scan_in_world[:2, :]; 

        return scan_in_world_planar;


    def get_control(s, odom):
        """
        Use the pose at time t and t-1 to calculate what control the robot could have taken
        at time t-1 at state (x,y,th)_{t-1} to come to the current state (x,y,th)_t. We will
        assume that this is the same control that the robot will take in the function dynamics_step
        below at time t, to go to time t-1. need to use the smart_minus_2d function to get the difference of the two poses and we will simply set this to be the control (delta x, delta y, delta theta)
        """

        # return linearVelocity and angularVelocity at the current time stamp

        w_v = s.l * odom.twist.twist.angular.z / 2;
        vr = w_v + odom.twist.twist.linear.x;
        vl = -w_v + odom.twist.twist.linear.x;
        # R = float('inf');
        # # print("spped: " + str(vr) + " || " + str(vl))

        # if abs(abs(vl) - abs(vr)) > 1e-2:
        #     R = 0.5 * s.l * (vr + vl) / (vr - vl) 
        # print(odom.twist.twist.angular)
        # print(odom.twist.twist.linear) 
        # return R, odom.twist.twist.angular.z
        return vr, vl, odom.twist.twist.angular.z


    def dynamics_step(s, odom):
        """"
        Compute the control using get_control and perform that control on each particle to get the updated locations of the particles in the particle filter, remember to add noise using the smart_plus_2d function to each particle
        """

        vr, vl, ome = s.get_control(odom);

        s.curTime = odom.header.stamp.secs + odom.header.stamp.nsecs*1e-9
        # print(s.curTime)
        timeDiff = s.curTime - s.prevTime;

        if s.prevTime < 0:
            timDiff = 0.1

        # print(timeDiff)
        noise = np.random.multivariate_normal(np.zeros(s.Q.shape[0]), s.Q, (s.p.shape[1])).T;

        if abs(ome) > 1e-2:
            R = 0.5 * s.l * (vr + vl) / (vr - vl) 
            dw = ome * timeDiff;
            rot = np.array([[np.cos(dw), -np.sin(dw), 0],
                [np.sin(dw), np.cos(dw), 0],
                [0, 0, 1]]);
            # print("previous prediction: " + str(s.p.T) + " R: " + str(R)

            p_ic_origin = deepcopy(s.p);
            p_ic_origin[0,:] -= R * np.sin(s.p[2,:])
            p_ic_origin[1,:] += R * np.cos(s.p[2,:])
            p_ic_origin[2,:] = dw;

            p_ic = deepcopy(s.p);
            p_ic[0,:] -= p_ic_origin[0,:]
            p_ic[1,:] -= p_ic_origin[1,:]
            # print(p_ic_origin)
            s.p = np.matmul(rot, p_ic) + p_ic_origin;
            
            # print("rot: " + str(rot))
            # print("mat mul: " + str((np.matmul(rot, p_ic)).T))
            # print("p_ic_origin: " + str(p_ic_origin.T))
            # print("p_ic: " + str(p_ic.T))
            # print("prediction: " + str(s.p.T) + " R: " + str(R))
        else:
            vt = 0.5 * (vr + vl)
            s.p[0,:] += timeDiff * vt * np.cos(s.p[2,:]);
            s.p[1,:] += timeDiff * vt * np.sin(s.p[2,:]);

        # s.p += noise;
        s.prevTime = s.curTime;

        # s.curTime = odom.header.stamp.secs + odom.header.stamp.nsecs*1e-9

        # timeDiff = s.curTime - s.prevTime;

        # if s.prevTime < 0:
        #     timDiff = 0.1

        # vr, vl, ome = s.get_control(odom);

        # vt = 0.5 * (vr + vl)
        # s.p[0,:] += timeDiff * vt * np.cos(s.p[2,:]);
        # s.p[1,:] += timeDiff * vt * np.sin(s.p[2,:]);
        # s.p[2,:] += ome * timeDiff

        # s.prevTime = s.curTime;
        # print("prediction: " + str(s.p.T))

        cy = np.cos(s.p[2] * 0.5);
        sy = np.sin(s.p[2] * 0.5);
        cp = np.cos(0 * 0.5);
        sp = np.sin(0 * 0.5);
        cr = np.cos(0 * 0.5);
        sr = np.sin(0 * 0.5);

        q = cr * cp * sy - sr * sp * cy;
        
        cur = make_homogeneous_coords_3d(s.p);
        R_tf = euler_to_se3(0, 0, s.p[2], np.array([-3, 1, 0]));

        intf = np.matmul(R_tf, cur)
        intf[2,:] = q
        intf = intf[:3,:]
        print("prediction: " + str(intf.T))
    
    @staticmethod
    def update_weights(w, obs_logp): 
        """
        Given the observation log-probability and the weights of particles w, calculate the
        new weights as discussed in the writeup. Make sure that the new weights are normalized
        """
        #### TODO: XXXXXXXXXXX
        log_w = np.log(w) + obs_logp;
        newWeight = log_w - slam_t.log_sum_exp(log_w);

        return np.exp(newWeight);


    def observation_step(s, lidar):
        """
        This function does the following things
            1. updates the particles using the LiDAR observations
            2. updates map.log_odds and map.cells using occupied cells as shown by the LiDAR data

        Some notes about how to implement this.
            1. As mentioned in the writeup, for each particle
                (a) First find the head, neck angle at t (this is the same for every particle)
                (b) Project lidar scan into the world frame (different for different particles)
                (c) Calculate which cells are obstacles according to this particle for this scan,
                calculate the observation log-probability
            2. Update the particle weights using observation log-probability
            3. Find the particle with the largest weight, and use its occupied cells to update the map.log_odds and map.cells.
        You should ensure that map.cells is recalculated at each iteration (it is simply the binarized version of log_odds). map.log_odds is of course maintained across iterations.
        """

        numOfParticle = s.p.shape[1];

        similarities = np.zeros(numOfParticle);

        for i in np.arange(numOfParticle): # iterate all the 
            occu = s.rays2world(s.p[:, i], lidar.ranges, s.lidar_angles);
            ir, ic = s.map.grid_cell_from_xy(occu[0,:], occu[1,:]);
            # s.map.num_obs_per_cell[ir, ic] += 1

            similarities[i] = np.sum(s.map.cells[ir, ic]);

        s.w = s.update_weights(s.w, similarities);


        id_max = np.argmax(s.w);
        s.cur = s.p[:, id_max]
        occu_max = s.rays2world(s.p[:, id_max], lidar.ranges, s.lidar_angles);
        max_indices_r, max_indices_c = s.map.grid_cell_from_xy(occu_max[0,:], occu_max[1,:]);

        limitsx = np.array([s.cur[0] - s.lidar_dmax / 2, s.cur[0] + s.lidar_dmax / 2, s.cur[0]]);
        limitsy = np.array([s.cur[1] - s.lidar_dmax / 2, s.cur[1] + s.lidar_dmax / 2, s.cur[1]]);

        lr, lc = s.map.grid_cell_from_xy(limitsx, limitsy)

        s.curgrid = [lr[2], lc[2]];
        maxProposal = np.zeros((s.map.szx, s.map.szy), dtype= np.int8);

        x_f = np.ndarray.flatten(np.linspace([lr[2]]*len(max_indices_r), max_indices_r, endpoint=False).astype('int64'))
        y_f = np.ndarray.flatten(np.linspace([lc[2]]*len(max_indices_r), max_indices_c, endpoint=False).astype('int64'))


        maxProposal[max_indices_r, max_indices_c] = 1;
        s.map.num_obs_per_cell[max_indices_r, max_indices_c] += 1
        s.map.num_obs_per_cell[x_f, y_f] += 1 # non decreasing

        s.map.log_odds[maxProposal == 1] += s.lidar_log_odds_occ; # mask
        # s.map.log_odds[maxProposal == 0] += s.lidar_log_odds_free;
        s.map.log_odds[x_f, y_f] += s.lidar_log_odds_free / 5;

        s.map.log_odds = np.clip(s.map.log_odds, -s.map.log_odds_max, s.map.log_odds_max);

        s.map.cells[s.map.log_odds >= s.map.log_odds_thresh] = 1;
        s.map.cells[s.map.log_odds < s.map.log_odds_thresh] = 0;

        s.resample_particles(); # the new weight might be all zero

    def resample_particles(s):
        """
        Resampling is a (necessary) but problematic step which introduces a lot of variance
        in the particles. We should resample only if the effective number of particles
        falls below a certain threshold (resampling_threshold). A good heuristic to
        calculate the effective particles is 1/(sum_i w_i^2) where w_i are the weights
        of the particles, if this number of close to n, then all particles have about
        equal weights and we do not need to resample
        """
        e = 1/np.sum(s.w**2)
        logging.debug('> Effective number of particles: {}'.format(e))
        if e/s.n < s.resampling_threshold:
            s.p, s.w = s.stratified_resampling(s.p, s.w)
            logging.debug('> Resampling')

    def plotMap(s):
        ax = plt.subplot(111)
        px, py = np.nonzero(s.map.cells);
        ax.clear()
        ax.plot(px, py, '*r')
        plt.draw()
        plt.pause(0.01)
