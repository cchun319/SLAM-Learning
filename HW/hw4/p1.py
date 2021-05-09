import numpy as np
import torch as th
import torch.nn as nn
from torch.distributions import Normal
import matplotlib.pyplot as plt


class u_t(nn.Module):
    def __init__(s, xdim=2, udim=1):
        super().__init__()
        """
        Build two layer neural network
        We will assume that the variance of the stochastic
        controller is a constant, to the network simply
        outputs the mean. We will do a hack to code up the constraint
        on the magnitude of the control input. We use a tanh nonlinearity
        to ensure that the output of the network is always between [-1,1]
        and then add a noise to it. While the final sampled control may be larger
        than the constraint we desire [-1,1], this is a quick cheap way to enforce the constraint.
        """
        s.m = nn.Sequential(
                nn.Linear(xdim, 8),
                nn.ReLU(True),
                nn.Linear(8, udim),
                nn.Tanh(),
                )
        s.std = 1

    def forward(s, x, u=None):
        """
        This is a PyTorch function that runs the network
        on a state x to output a control u. We will also output
        the log probability log u_theta(u | x) because we want to take
        the gradient of this quantity with respect to the parameters
        """
        # mean control
        mu = s.m(x)
        # Build u_theta(cdot | x)
        n = Normal(mu, s.std)
        # sample a u if we are simulating the system, use the argument
        # we are calculating the policy gradient
        if u is None:
            u = n.rsample()
        logp = n.log_prob(u)
        return u, logp

def rollout(policy):
    """
    We will use the control u_theta(x_t) to take the control action at each
    timestep. You can use simple Euler integration to simulate the ODE forward
    for T = 200 timesteps with discretization dt=0.05.
    At each time-step, you should record the state x,
    control u, and the reward r
    """
    m = 1; l=1; b=0.1; g=9.8;
    gamma=0.99;
    get_rev = lambda z, zdot, u: -0.5*((np.pi-z)**2 + zdot**2 + 0.01*u**2)

    xs = [np.zeros(2)]; us = []; rs= [];
    dt = 0.05
    for t in np.arange(0, 10, dt):
        # The interface between PyTorch and numpy becomes a bit funny
        # but all that this line is doing is that it is running u(x) to get
        # a control for one state x
        u = policy(th.from_numpy(xs[-1]).view(1,-1).float())[0].detach().numpy().squeeze().item()

        z, zdot = xs[-1][0], xs[-1][1]
        zp = z + zdot*dt
        zdotp = zdot + dt*(u - m*g*l*np.sin(z) - b*zdot)/m/l**2

        rs.append(get_rev(z, zdot, u))
        us.append(u)
        xs.append(np.array([zp, zdotp]))

    R = sum([rr*gamma**k for k,rr in enumerate(rs)])
    return {'x': th.tensor(xs[:-1]).float(),
            'u': th.tensor(us).float(),
            'r': th.tensor(rs).float(), 'R': R}


def example_train():
    """
    The following code shows how to compute the policy gradient and update
    the weights of the neural network using one trajectory.
    """
    policy = u_t(xdim=2, udim=1)
    optim = th.optim.Adam(policy.parameters(), lr=1e-3)

    # 1. get a trajectory
    t = rollout(policy)
    """"
    2. We now want to calculate grad log u_theta(u | x), so
    we will feed all the states from the trajectory again into the network
    and this time we are interested in the log-probabilities. The following
    code shows how to update the weights of the model using one trajectory
    """
    logp = policy(th.from_numpy(t['x']).float())[1]
    f = (t['R']*logp).mean()

    # zero_grad is a PyTorch peculiarity that clears the backpropagation
    # gradient buffer before calling the next .backward()
    policy.zero_grad()
    # .backward computes the gradient of the policy gradient objective with respect
    # to the parameters of the policy and stores it in the gradient buffer
    f.backward()
    # .step() updates the weights of the policy using the computed gradient
    optim.step()

def train():
    """
    TODO: XXXXXXXXXXXX
    This is very similar to example_tran() above. You should sample
    multiple trajectory at each iteration and run the training for about 1000
    iterations. You should track the average value of the return across multiple
    trajectories and plot it as a function of the number of iterations.
    """
    policy = u_t(xdim=2, udim=1)
    optim = th.optim.Adam(policy.parameters(), lr=1e-3)

    num_sampling_traj = 50;
    iter_ = 1000

    R_t = [];
    ax = plt.subplot(111)
    for i in range(iter_):

        # 1. get a trajectory
        r = th.zeros([num_sampling_traj, 1], dtype=th.float64)
        f = th.zeros([num_sampling_traj, 200, 1], dtype=th.float64)
        for j in range(num_sampling_traj):
            t = rollout(policy)
            """"
            2. We now want to calculate grad log u_theta(u | x), so
            we will feed all the states from the trajectory again into the network
            and this time we are interested in the log-probabilities. The following
            code shows how to update the weights of the model using one trajectory
            """
            logp = policy(t['x'].view(-1,2), t['u'].view(-1,1))[1]
            f[j, :, :] = logp
            r[j] = t['R']

        b = r.mean();
        l = th.zeros([1, 1], dtype=th.float64)
        for j in range(num_sampling_traj):
            l += -((r[j] - b) * f[j]).mean()


        # zero_grad is a PyTorch peculiarity that clears the backpropagation
        # gradient buffer before calling the next .backward()
        policy.zero_grad()
        # .backward computes the gradient of the policy gradient objective with respect
        # to the parameters of the policy and stores it in the gradient buffer
        l.backward()
        # .step() updates the weights of the policy using the computed gradient
        optim.step()

        R_t.append(l);
        print("iter: " + str(i))

        ax.clear()
        ax.plot(range(i + 1), R_t, '-r')
        plt.draw()
        plt.pause(0.01)

    



if __name__ == "__main__":
    train();