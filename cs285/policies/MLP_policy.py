import abc
import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.base_policy import BasePolicy
from cs285.envs.city import MAX_CAP, MAX_CAND_NUM

class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 shared_exp = False,
                 shared_exp_lambda = 1.,
                 is_city=True,
                 learning_rate=1e-4,
                 n_drivers = 1,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        # init vars
        self.ac_dim = 5
        self.ob_dim = 3+2*MAX_CAP
        self.n_layers = n_layers
        self.is_city = is_city
        self.n_drivers = n_drivers
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.shared_exp = shared_exp
        self.shared_exp_lambda = shared_exp_lambda
        self.nn_baseline = nn_baseline
        if self.is_city:
            self.agent_logits_nets = []
            #self.agent_logstds = []
            self.agent_optimizers = []
            #self.logits_na = None
            for i in range(self.n_drivers):
                self.agent_logits_nets.append(ptu.build_mlp(input_size=self.ob_dim+self.ac_dim,
                                               output_size=1,
                                               n_layers=self.n_layers,
                                               size=self.size))
                #self.agent_logstds.append(nn.Parameter(
                #    torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
                #))
                self.agent_logits_nets[i].to(ptu.device)
                #self.logstd.to(ptu.device)
                self.agent_optimizers.append(
                    optim.Adam(self.agent_logits_nets[i].parameters(),
                               self.learning_rate)
                )
            #self.logits_na = None
            self.baseline = None

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    # query the policy with observation(s) to get selected action(s)
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        # this obs has a dim of ob_dim + ac_dim*(1+MAX_CAND_NUM)
        if len(obs.shape) > 2:
            observation = obs
        else:
            observation = obs[None]
        observation = ptu.from_numpy(observation)
        action_distributions = self(observation)
        actions = []
        for i in range(self.n_drivers):
            action_i = ptu.to_numpy(action_distributions[i].sample())  # don't bother with rsample
            actions.append(action_i)
        actions = np.array(actions)
        return actions.T

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, obs: torch.FloatTensor):
        # this obs has a dim of n_exp * n_drivers * (ob_dim + ac_dim*(1+MAX_CAND_NUM))
        if len(obs.shape) > 2:
            observation = obs
        else:
            observation = obs[None]
        if self.is_city:
            action_distributions = []
            #import pdb; pdb.set_trace()
            for i in range(self.n_drivers):
                logit_this_driver = []
                for j in range(MAX_CAND_NUM+1):
                    index_slice = ptu.from_numpy(np.array(list(range(self.ob_dim))
                                         +list(range(self.ob_dim+j*self.ac_dim, self.ob_dim+(j+1)*self.ac_dim))), is_int=True)
                    logit_this_driver.append(
                        self.agent_logits_nets[i](
                            observation[:,i,:].index_select(1,index_slice)
                        )
                    )
                    #import pdb; pdb.set_trace()
                logit_this_driver_ts = torch.stack(logit_this_driver)
                logit_this_driver_ts.transpose_(1,0)
                logit_this_driver_ts.squeeze_()
                action_distributions.append(distributions.Categorical(logits=logit_this_driver_ts))
        return action_distributions
    
    def shared_forward(self, obs):
        if len(obs.shape) > 2:
            observation = obs
        else:
            observation = obs[None]
        action_distributions = dict()
        for i in range(self.n_drivers):
            for k in range(self.n_drivers):
                logit_i_net_k_obs = []
                for j in range(MAX_CAND_NUM+1):
                    index_slice = ptu.from_numpy(np.array(list(range(self.ob_dim))
                                         +list(range(self.ob_dim+j*self.ac_dim, self.ob_dim+(j+1)*self.ac_dim))), is_int=True)
                    logit_i_net_k_obs.append(
                        self.agent_logits_nets[i](
                            observation[:,k,:].index_select(1,index_slice)
                        )
                    )
                logit_this_driver_ts = torch.stack(logit_i_net_k_obs)
                logit_this_driver_ts.transpose_(1,0)
                logit_this_driver_ts.squeeze_()
                action_distributions[(i,k)]= distributions.Categorical(logits=logit_this_driver_ts)
        return action_distributions
        


#####################################################


#####################################################


class MLPPolicyAC(MLPPolicy):
    def update(self, observations, actions, adv_n=1.):
        # TODO: update the policy and return the loss
        if isinstance(adv_n, np.ndarray):
            adv_n = ptu.from_numpy(adv_n)
        if isinstance(observations, np.ndarray):
            observations = ptu.from_numpy(observations)
        if isinstance(actions, np.ndarray):
            actions = ptu.from_numpy(actions)
        if self.shared_exp:
            action_distributions = self.shared_forward(observations)
        else:
            action_distributions = self(observations)
        losses = []
        for i in range(self.n_drivers):
            if not self.shared_exp:
                losses.append((-action_distributions[i].log_prob(actions[:,i]) * adv_n[:,i]).mean())
            else:
                losses_i = -(action_distributions[i,i].log_prob(actions[:,i]) * ptu.from_numpy(adv_n[(i,i)]))
                for k in range(self.n_drivers):
                    if k != i:
                        losses_i = losses_i -self.shared_exp_lambda * \
                        torch.div(torch.exp(action_distributions[(i,k)].log_prob(actions[:,k])), torch.exp(action_distributions[(k,k)].log_prob(actions[:,k]))) *\
                        action_distributions[(k,k)].log_prob(actions[:,k]) * ptu.from_numpy(adv_n[(i,k)])
                losses.append(losses_i.mean())
                
            # -action_distributions[i].log_prob(actions[:,i]) =
            # tensor([log pro for driver i in test 0, ...in test 1,...])
            # import pdb; pdb.set_trace()
            self.agent_optimizers[i].zero_grad()
            if self.shared_exp:
                losses[i].backward(retain_graph=True)
            else:
              losses[i].backward()
            self.agent_optimizers[i].step()
        return losses[i].item()

"""
import numpy as np
ob_dim = 3+2*MAX_CAP
ac_dim = 5
n_drivers = 3
aa = MLPPolicyAC(ac_dim,ob_dim,2,10,n_drivers=n_drivers)
obs_lb_one_driver = [0, 0, 0] + [0 for _ in range(2 * MAX_CAP)] + \
                            [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0]*MAX_CAND_NUM
obs = np.array([[obs_lb_one_driver for _ in range(n_drivers)] for _ in range(2)])
cc = aa.get_action(obs)
adv_n = np.array([[0.01038937, 0.18045077, 0.04163878],
       [0.01038937, 0.18045075, 0.04163878]])
aa.update(obs,cc,adv_n=adv_n)
"""

