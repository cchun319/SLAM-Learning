import gym
import numpy as np
import torch as th
import torch.nn as nn
import random
from tqdm import tqdm
from matplotlib import pyplot as plt
"""
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.
    Source:
        This environment corresponds to the version of the cart-pole problem
        described by Barto, Sutton, and Anderson
    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf
    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right
        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it
    Reward:
        Reward is 1 for every step taken, including the termination step
    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]
    Episode Termination:
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.
"""

batchSize = 120
gma = 0.99
apa = 0.05


def rollout(e, q, eps=0, T=200):
    # get the control signal from current step, network(x) -> control signal -> step(control) -> next state
    traj = []

    x = e.reset()
    for t in range(T):
        u = q.control(th.from_numpy(x).float().unsqueeze(0), eps=eps)
        u = u.int().numpy().squeeze()

        xp,r,d,info = e.step(u) # next state, reward, terminal state
        t = dict(x=x,xp=xp,r=r,u=u,d=d,info=info)
        # print(t)
        x = xp
        traj.append(t)
        if d:
            break
    return traj

class q_t(nn.Module):
    def __init__(s, xdim, udim, hdim=16):
        super().__init__()
        s.xdim, s.udim = xdim, udim
        s.m = nn.Sequential(
                            nn.Linear(xdim, hdim),
                            nn.ReLU(True),
                            nn.Linear(hdim, udim),
                            )

    def forward(s, x):
        return s.m(x);

    def control(s, x, eps=0):
        # 1. get q values for all controls
        val = s.m(x)

        ### TODO: XXXXXXXXXXXX
        # eps-greedy strategy to choose control input
        # note that for eps=0
        # you should return the correct control u
        u = th.argmax(val, dim=1);

        prob = np.random.uniform();

        if( prob < eps):
            u = th.tensor([np.random.choice([0,1])])

        return u

def loss(q, ds, q_tar, ddqn):
    ### TODO: XXXXXXXXXXXX
    # 1. sample mini-batch from datset ds
    # 2. code up dqn with double-q trick
    # 3. return the objective f

    randId = np.random.choice(range(len(ds)), batchSize);

    randPts = [];

    for id in randId:
        rid = np.random.choice(range(len(ds[id])), 5)
        for j in rid:
            randPts.append(ds[id][j])

    f = 0;

    for id in range(len(randPts)):
        q_val = q(th.from_numpy(randPts[id]['x']).float().unsqueeze(0))[0, randPts[id]['u']]
        q_max = th.max(q_tar(th.from_numpy(randPts[id]['xp']).float().unsqueeze(0)))
        
        if ddqn == True:
            u_best = th.argmax(q(th.from_numpy(randPts[id]['x']).float().unsqueeze(0)), dim=1);
            q_max = q_tar(th.from_numpy(randPts[id]['xp']).float().unsqueeze(0))[0, u_best]
            # print(u_best)
        # print(int(randPts[id]['d']))
        f += (q_val - randPts[id]['r'] - gma * (1 - int(randPts[id]['d'])) * q_max.detach())**2

    f /= len(randPts)

    return f

def evaluate(q):
    ### TODO: XXXXXXXXXXXX
    # 1. create a new environment e
    # 2. run the learnt q network for 100 trajectories on
    # this new environment to take control actions. Remember that
    # you should not perform epsilon-greedy exploration in the evaluation
    # phase
    # and report the average discounted
    # return of these 100 trajectories

    T = 200
    numOfTraj = 100
    r = 0;

    e_eval = gym.make('CartPole-v1')
    
    for _ in range(numOfTraj):
        x = e_eval.reset();
        for t in range(T):
            u = q.control(th.from_numpy(x).float().unsqueeze(0), eps=0)
            u = u.int().numpy().squeeze()

            xp,tr,d,info = e_eval.step(u) # next state, reward, terminal state
            # print(t)
            r += tr;
            x = xp
            if d:
                break
    r /= numOfTraj

    return r

def evaluateLast10(e, q):
    
    T = 200
    numOfTraj = 10
    r = 0;
    
    for _ in range(numOfTraj):
        x = e.reset();
        for t in range(T):
            u = q.control(th.from_numpy(x).float().unsqueeze(0), eps=0)
            u = u.int().numpy().squeeze()

            xp,tr,d,info = e.step(u) # next state, reward, terminal state
            # print(t)
            r += tr;
            x = xp
            if d:
                break
    r /= numOfTraj

    return r


if __name__=='__main__':
    # th.set_default_dtype(th.float64)
    e = gym.make('CartPole-v1')
    # th.set_default_dtype(th.float64)
    xdim, udim =    e.observation_space.shape[0], \
                    e.action_space.n

    q = q_t(xdim, udim, 8)
    q2 = q_t(xdim, udim, 8)

    # Adam is a variant of SGD and essentially works in the
    # same way
    optim = th.optim.Adam(q.parameters(), lr=1e-3,
                          weight_decay=1e-4)
    training_iter = 20000;
    ds = []
    epss = 0.5
    decay_factor = 0.9999
    # collect few random trajectories with
    # eps=1

    samping_iter = 500
    train_rt = []
    losss = []
    rewards = []
    q_tar = q_t(xdim, udim, 8)
    q_tar2 = q_t(xdim, udim, 8)
    q_tar.eval();
    q_tar2.eval();

    interval = 100
    ddqn = True

    print("Sampling paths")
    for i in tqdm(range(samping_iter)):
        ds.append(rollout(e, q, eps=1, T=200))
        # e.render();

    print("Learning the controllers")
    for i in tqdm(range(1, training_iter + 1)):
        q.train()
        t = rollout(e, q, epss)
        ds.append(t) # replay buffer

        # perform weights updates on the q network
        # need to call zero grad on q function
        # to clear the gradient buffer
        q.zero_grad()
        f = loss(q, ds, q_tar, ddqn)
        f.backward()
        optim.step()
        for tar_para, q_para in zip(q_tar.parameters(), q.parameters()):
            tar_para.data = (1 - apa) * tar_para + apa * q_para
        losss.append(f.item())

        # evaluate the last 10 traj

        epss *= decay_factor;

        if i % interval == 0:
            rw = evaluate(q);
            rewards.append(rw)
            train_rt.append(evaluateLast10(e, q))
            # plt.subplot(1, 3, 1) # row 1, col 2 index 1
            # plt.plot(range(i), losss)
            # plt.title("loss")

            # plt.subplot(1, 3, 2) # index 2
            # plt.plot(range(len(train_rt)), train_rt)
            # plt.title("traning reward")

            # plt.subplot(1, 3, 3) # index 2
            # plt.plot(range(len(train_rt)), rewards)
            # plt.title("reward")

            # plt.show()

        # 1000 iter -> evalution for last 10 traj -> training error
        # 1000 iter -> generate 10 more traj test evaultion new envrioment 


        # plot the loss 
        # keep track of return by the eval function
    plt.subplot(1, 3, 1) # row 1, col 2 index 1
    plt.plot(range(training_iter), losss)
    plt.title("loss")

    plt.subplot(1, 3, 2) # index 2
    plt.plot(range(training_iter//interval), train_rt)
    plt.title("traning reward")

    plt.subplot(1, 3, 3) # index 2
    plt.plot(range(training_iter//interval), rewards)
    plt.title("reward")

    plt.show()



    # # on policy
    # policy gradient, sample trajectory(currrent policy) past traj is not valid
    # # off policy
    # pair(x_t, r, x_(t + 1))
    # # actor-critic -> 
    # # actor -> policy
    # # critic -> reward  


