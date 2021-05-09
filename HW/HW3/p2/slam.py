# Pratik Chaudhari (pratikac@seas.upenn.edu)
from matplotlib.path import Path
import os, sys, pickle, math
from copy import deepcopy

from scipy import io
import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from load_data import load_lidar_data, load_joint_data, joint_name_to_index
from utils import *

import logging
logger = logging.getLogger()
logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))

class map_t:
    """
    This will maintain the occupancy grid and log_odds. You do not need to change anything
    in the initialization
    """
    def __init__(s, resolution=0.05):
        s.resolution = resolution
        s.xmin, s.xmax = -20.0, 20.0
        s.ymin, s.ymax = -20.0, 20.0
        s.szx = int(np.ceil((s.xmax-s.xmin)/s.resolution+1))
        s.szy = int(np.ceil((s.ymax-s.ymin)/s.resolution+1))

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
        s.occupied_prob_thresh = 0.3
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
        r = (x - s.xmin) / s.resolution;
        c = (y - s.ymin) / s.resolution;
        r = r.astype(int)
        c = c.astype(int)
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
        s.Q = Q

        # we resample particles if the effective number of particles
        # falls below s.resampling_threshold*num_particles
        s.resampling_threshold = resampling_threshold

        # initialize the map
        s.map = map_t(resolution)

    def read_data(s, src_dir, idx=0, split='train'):
        """
        src_dir: location of the "data" directory
        """
        logging.info('> Reading data')
        s.idx = idx
        s.lidar = load_lidar_data(os.path.join(src_dir,
                                               'data/%s/%s_lidar%d'%(split,split,idx)))
        s.joint = load_joint_data(os.path.join(src_dir,
                                               'data/%s/%s_joint%d'%(split,split,idx)))

        # finds the closets idx in the joint timestamp array such that the timestamp
        # at that idx is t
        s.find_joint_t_idx_from_lidar = lambda t: np.argmin(np.abs(s.joint['t']-t))
        s.current = s.lidar[0]['xyth'];
    def init_sensor_model(s):
        # lidar height from the ground in meters
        s.head_height = 0.93 + 0.33
        s.lidar_height = 0.15

        # dmin is the minimum reading of the LiDAR, dmax is the maximum reading
        s.lidar_dmin = 1e-3
        s.lidar_dmax = 30
        s.lidar_angular_resolution = 0.25
        # these are the angles of the rays of the Hokuyo
        s.lidar_angles = np.arange(-135,135+s.lidar_angular_resolution,
                                   s.lidar_angular_resolution)*np.pi/180.0

        # sensor model lidar_log_odds_occ is the value by which we would increase the log_odds
        # for occupied cells. lidar_log_odds_free is the value by which we should decrease the
        # log_odds for free cells (which are all cells that are not occupied)
        thres = 9.0 
        s.lidar_log_odds_occ = np.log(thres) # 0.9
        s.lidar_log_odds_free = np.log(1/thres)

    def init_particles(s, n=100, p=None, w=None, t0=0):
        """
        n: number of particles
        p: xy yaw locations of particles (3xn array)
        w: weights (array of length n)
        """
        s.n = n
        s.p = deepcopy(p) if p is not None else np.zeros((3,s.n), dtype=np.float64)
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
        newW = np.ones_like(w, dtype=np.float64) / numOfParticle;
        newP = np.zeros_like(p, dtype=np.float64);
        i = 0
        c = w[0];
        r = np.random.uniform(0.0, 1.0 / numOfParticle)

        for m in np.arange(numOfParticle):
            u = r + m  / numOfParticle;
            while(u > c):
                i = i + 1;
                c = c + w[i % numOfParticle];

            newP[:, m] = p[:, i % numOfParticle];

        return newP, newW;


    @staticmethod
    def log_sum_exp(w):
        return w.max() + np.log(np.exp(w-w.max()).sum())

    def rays2world(s, p, d, head_angle=0, neck_angle=0, angles=None, t=0):
        """
        p is the pose of the particle (x,y,yaw)
        angles = angle of each ray in the body frame (this will usually be simply s.lidar_angles for the different lidar rays)
        d = is an array that stores the distance of along the ray of the lidar, for each ray (the length of d has to be equal to that of angles, this is s.lidar[t]['scan'])
        Return an array 2 x num_rays which are the (x,y) locations of the end point of each ray
        in world coordinates
        """

        # make sure each distance >= dmin and <= dmax, otherwise something is wrong in reading
        # the data


        # d = np.clip(d, s.lidar_dmin, s.lidar_dmax);
        indices = np.where((d <= s.lidar_dmax) & (d >= s.lidar_dmin));
        angles = angles[indices];
        d = d[indices];

        # 1. from lidar distances to points in the LiDAR frame, #(r, the) -> (x, y, 0) in lidar frame
        scan_in_lidar = np.zeros((3, len(angles)), dtype=np.float64);
        scan_in_lidar[0,:] = d * np.cos(angles);
        scan_in_lidar[1,:] = d * np.sin(angles); 
        scan_in_lidar_H = make_homogeneous_coords_3d(scan_in_lidar); # (4 x n) x, y, 0, 1

        # 2. from LiDAR frame to the body frame
        R_bodyToLidar = euler_to_se3(0, head_angle, neck_angle , np.array([0, 0, s.lidar_height])); # 4 x 4

        scan_in_body = R_bodyToLidar @ scan_in_lidar_H;        
        # 3. from body frame to world frame (x, y, s.head_height, the)
        R_worldToBody = euler_to_se3(0, 0, p[2], np.array([p[0], p[1], s.head_height]));
        scan_in_world = R_worldToBody @ scan_in_body;

        scan_in_world_planar = scan_in_world[:, scan_in_world[2,:] >= s.head_height - 0.1]
        scan_in_world_planar = scan_in_world_planar[:, scan_in_world_planar[2,:] < 2 * s.head_height]

        # scan_in_world_planar = scan_in_world

        return scan_in_world_planar[0:2,:];


    def get_control(s, t):
        """
        Use the pose at time t and t-1 to calculate what control the robot could have taken
        at time t-1 at state (x,y,th)_{t-1} to come to the current state (x,y,th)_t. We will
        assume that this is the same control that the robot will take in the function dynamics_step
        below at time t, to go to time t-1. need to use the smart_minus_2d function to get the difference of the two poses and we will simply set this to be the control (delta x, delta y, delta theta)
        """

        if t == 0:
            return np.zeros(3)

        control = smart_minus_2d(s.lidar[t]['xyth'], s.lidar[t - 1]['xyth']);
        # control[2] = s.lidar[t]['rpy'][2] - s.lidar[t - 1]['rpy'][2]
        return control


    def dynamics_step(s, t):
        """"
        Compute the control using get_control and perform that control on each particle to get the updated locations of the particles in the particle filter, remember to add noise using the smart_plus_2d function to each particle
        """

        control = s.get_control(t);
        for i in np.arange(s.p.shape[1]):
            s.p[:, i] = smart_plus_2d(s.p[:, i], control); # TODO: add noise
            noise = np.random.multivariate_normal(np.zeros(s.Q.shape[0]), s.Q);
            s.p[:, i] = smart_plus_2d(s.p[:, i], noise)
    
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

    def sampleTheNonOccupyArea(s, x, y):
        X = [];
        Y = [];
        direction = [x - s.current[0], y - s.current[1]];
        lent = np.linalg.norm(direction, axis=0)
        samplePts = lent // 0.05;
        for i in np.arange(len(x)):
            x_pts = np.linspace(s.current[0], x[1], int(samplePts[i]), endpoint = False);
            y_pts = np.linspace(s.current[1], y[1], int(samplePts[i]), endpoint = False);
            X.extend(x_pts)
            Y.extend(y_pts)
        return X, Y

    def bresenham(s, x1,y1,x2,y2):
 
        X = []
        Y = []
        if(abs(x1 - x2) < 1 and abs(y1 - y2) < 1):
            return X, Y

        if(abs(x2 - x1) > abs(y2 - y1)):
            m_new = 2 * (y2 - y1)
            slope_error_new = m_new - (x2 - x1)
            
            y = y1
            for x in range(x1, x2 + 1, int(abs(x2 - x1) / (x2 - x1))):
                X.append(x)
                Y.append(y)
                # Add slope to increment angle formed
                slope_error_new = slope_error_new + m_new
         
                # Slope error reached limit, time to
                # increment y and update slope error.
                if (slope_error_new >= 0):
                    y = y+1
                    slope_error_new =slope_error_new - 2 * (x2 - x1)
        else:
            m_new = 2 * (x2 - x1)
            slope_error_new = m_new - (y2 - y1)
                
            x = x1
            # print("(",abs(x1 - x2) ,",",abs(y1 - y2),"\n")
            for y in range(y1, y2, int(abs(y2 - y1) / (y2 - y1))):
                X.append(x)
                Y.append(y)
                # Add slope to increment angle formed
                slope_error_new =slope_error_new + m_new
         
                # Slope error reached limit, time to
                # increment y and update slope error.
                if (slope_error_new >= 0):
                    x = x +1
                    slope_error_new =slope_error_new - 2 * (y2 - y1)

        return X, Y


    def observation_step(s, t):
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

        tid = s.find_joint_t_idx_from_lidar(s.lidar[t]['t'])

        similarities = np.zeros(numOfParticle);

        neck_ang = s.joint['head_angles'][0][tid];
        head_ang = s.joint['head_angles'][1][tid];

        for i in np.arange(numOfParticle): # iterate all the 
            occu = s.rays2world(s.p[:, i], s.lidar[t]['scan'], head_ang, neck_ang, s.lidar_angles, t);
            ir, ic = s.map.grid_cell_from_xy(occu[0,:], occu[1,:]);

            similarities[i] = np.sum(s.map.cells[ir, ic]);

        s.w = s.update_weights(s.w, similarities);

        id_max = np.argmax(s.w);
        s.current = s.p[:,id_max];

        occu_max = s.rays2world(s.p[:, id_max], s.lidar[t]['scan'], head_ang, neck_ang, s.lidar_angles, t);
        max_indices_r, max_indices_c = s.map.grid_cell_from_xy(occu_max[0,:], occu_max[1,:]);

        limitsx = np.array([s.current[0] - s.lidar_dmax / 2, s.current[0] + s.lidar_dmax / 2, s.current[0]]);
        limitsy = np.array([s.current[1] - s.lidar_dmax / 2, s.current[1] + s.lidar_dmax / 2, s.current[1]]);

        lr, lc = s.map.grid_cell_from_xy(limitsx, limitsy)

        maxProposal = np.zeros((s.map.szx, s.map.szy), dtype= np.int8);

        # notR = [];
        # notC = [];

        # for i in range(len(max_indices_r)):
        #     x_, y_ = s.bresenham(lr[2], lc[2], max_indices_r[i] , max_indices_c[i]);
        #     notR.extend(x_);
        #     notC.extend(y_);

        x_f = np.ndarray.flatten(np.linspace([lr[2]]*len(max_indices_r), max_indices_r, endpoint=False).astype('int64'))
        y_f = np.ndarray.flatten(np.linspace([lc[2]]*len(max_indices_r), max_indices_c, endpoint=False).astype('int64'))

        # notX, notY = s.sampleTheNonOccupyArea(occu_max[0,:], occu_max[1,:])
        # notR, notC = s.map.grid_cell_from_xy(notX, notY)
        maxProposal[max_indices_r, max_indices_c] = 1;
        s.map.num_obs_per_cell[max_indices_r, max_indices_c] += 1
        s.map.num_obs_per_cell[x_f, y_f] += 1 # non decreasing

        s.map.log_odds[maxProposal == 1] += s.lidar_log_odds_occ; # mask
        # s.map.log_odds[maxProposal == 0] += s.lidar_log_odds_free / 10;

        s.map.log_odds[x_f, y_f] += s.lidar_log_odds_free * 0.25;

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