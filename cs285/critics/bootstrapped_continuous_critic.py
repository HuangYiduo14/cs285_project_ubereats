#from .base_critic import BaseCritic
from torch import nn
from torch import optim
from cs285.envs.city import MAX_CAP, MAX_CAND_NUM
from cs285.infrastructure import pytorch_util as ptu
import numpy as np

class BootstrappedContinuousCritic(nn.Module):
    """
        Notes on notation:

        Prefixes and suffixes:
        ob - observation
        ac - action
        _no - this tensor should have shape (batch self.size /n/, observation dim)
        _na - this tensor should have shape (batch self.size /n/, action dim)
        _n  - this tensor should have shape (batch self.size /n/)

        Note: batch self.size /n/ is defined at runtime.
        is None
    """
    def __init__(self, hparams):
        super().__init__()
        self.ob_dim = 3+2*MAX_CAP
        self.ac_dim = 5
        self.is_city = hparams['is_city']
        self.size = hparams['size']
        self.n_layers = hparams['n_layers']
        self.learning_rate = hparams['learning_rate']
        self.n_drivers = hparams['n_drivers']
        # critic parameters
        self.num_target_updates = hparams['num_target_updates']
        self.num_grad_steps_per_target_update = hparams['num_grad_steps_per_target_update']
        self.gamma = hparams['gamma']
        self.critic_networks = []
        self.losses = []
        self.optimizers = []
        for i in range(self.n_drivers):
            self.critic_networks.append(ptu.build_mlp(
            self.ob_dim,
            1,
            n_layers=self.n_layers,
            size=self.size,
            ))
            self.critic_networks[i].to(ptu.device)
            self.losses.append(nn.MSELoss())
            self.optimizers.append(optim.Adam(
                self.critic_networks[i].parameters(),
                self.learning_rate,
            ))

    def forward(self, obs):
        observations = obs[:,:,0:self.ob_dim]
        rewards = []
        for i in range(self.n_drivers):
            rewards.append(self.critic_networks[i](observations[:,i,:]).squeeze(1))
        return rewards

    def forward_np(self, obs):
        obs = ptu.from_numpy(obs)
        predictions = self(obs)
        results = np.array([ptu.to_numpy(predictions[i]) for i in range(self.n_drivers)])
        return results.T

    def update(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
        """
            Update the parameters of the critic.

            let sum_of_path_lengths be the sum of the lengths of the paths sampled from
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories

            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                next_ob_no: shape: (sum_of_path_lengths, ob_dim). The observation after taking one step forward
                reward_n: length: sum_of_path_lengths. Each element in reward_n is a scalar containing
                    the reward for each timestep
                terminal_n: length: sum_of_path_lengths. Each element in terminal_n is either 1 if the episode ended
                    at that timestep of 0 if the episode did not end

            returns:
                training loss
        """
        # TODO: Implement the pseudocode below: do the following (
        # self.num_grad_steps_per_target_update * self.num_target_updates)
        # times:
        # every self.num_grad_steps_per_target_update steps (which includes the
        # first step), recompute the target values by
        #     a) calculating V(s') by querying the critic with next_ob_no
        #     b) and computing the target values as r(s, a) + gamma * V(s')
        # every time, update this critic using the observations and targets
        #
        # HINT: don't forget to use terminal_n to cut off the V(s') (ie set it
        #       to 0) when a terminal state is reached
        # HINT: make sure to squeeze the output of the critic_network to ensure
        #       that its dimensions match the reward

        for i in range(self.num_target_updates):
            value_s_next = self.forward_np(next_ob_no)
            value_s_next[terminal_n==1] = 0
            targets_d = []
            for d in range(self.n_drivers):
                targets = reward_n[:, d] + self.gamma * value_s_next[:,d]
                targets_d.append(ptu.from_numpy(targets))

            for j in range(self.num_grad_steps_per_target_update):
                value_s = self.forward(ptu.from_numpy(ob_no))
                losses = []
                for d in range(self.n_drivers):
                    losses.append(self.losses[d](targets_d[d], value_s[d]))
                    self.optimizers[d].zero_grad()
                    losses[d].backward()
                    self.optimizers[d].step()
        return losses[d].item()

"""
import numpy as np
ob_dim = 3+2*MAX_CAP
ac_dim = 5
n_drivers = 3
hparams= dict()
hparams['ob_dim'] = ob_dim
hparams['ac_dim'] = ac_dim
hparams['is_city'] = True
hparams['size'] = 20
hparams['n_layers'] = 2
hparams['learning_rate'] = 1e-3
hparams['n_drivers']=n_drivers
# critic parameters
hparams['num_target_updates'] = 10
hparams['num_grad_steps_per_target_update'] = 10
hparams['gamma'] = 0.9
aa =  BootstrappedContinuousCritic(hparams)
obs_lb_one_driver = [0, 0, 0] + [0 for _ in range(2 * MAX_CAP)] + \
                            [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0]*MAX_CAND_NUM
obs = np.array([[obs_lb_one_driver for _ in range(n_drivers)] for _ in range(2)])
cc = aa.forward_np(obs)
aa.update(obs,0,obs,cc,np.array([False, False]))
"""
