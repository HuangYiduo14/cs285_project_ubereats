from collections import OrderedDict
import numpy as np
from cs285.critics.bootstrapped_continuous_critic import \
    BootstrappedContinuousCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import *
from cs285.policies.MLP_policy import MLPPolicyAC
from cs285.infrastructure import pytorch_util as ptu
from .base_agent import BaseAgent


class ACAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super().__init__()

        self.env = env
        self.agent_params = agent_params

        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']
        self.n_drivers = self.agent_params['n_drivers']

        self.actor = MLPPolicyAC(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['is_city'],
            self.agent_params['learning_rate'],
            self.agent_params['n_drivers']
        )

        self.critic = BootstrappedContinuousCritic(self.agent_params)

        self.replay_buffer = ReplayBuffer()

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        # TODO Implement the following pseudocode:
        # for agent_params['num_critic_updates_per_agent_update'] steps,
        #     update the critic
        loss = OrderedDict()
        for i in range(self.agent_params['num_critic_updates_per_agent_update']):
            loss['Critic_Loss'] = self.critic.update(ob_no, ac_na, next_ob_no, re_n, terminal_n)
        # advantage = estimate_advantage(...)
        advantage = self.estimate_advantage(ob_no,next_ob_no, re_n, terminal_n)

        # for agent_params['num_actor_updates_per_agent_update'] steps,
        #     update the actor
        for i in range(self.agent_params['num_actor_updates_per_agent_update']):
            loss['Actor_Loss'] = self.actor.update(ob_no, ac_na, advantage)
        return loss

    def estimate_advantage(self, ob_no, next_ob_no, re_n, terminal_n):
        # TODO Implement the following pseudocode:
        # 1) query the critic with ob_no, to get V(s)
        # 2) query the critic with next_ob_no, to get V(s')
        # 3) estimate the Q value as Q(s, a) = r(s, a) + gamma*V(s')
        # HINT: Remember to cut off the V(s') term (ie set it to 0) at terminal states (ie terminal_n=1)
        # 4) calculate advantage (adv_n) as A(s, a) = Q(s, a) - V(s)
        value_s = self.critic.forward_np(ob_no)
        value_next_s = self.critic.forward_np(next_ob_no)
        value_next_s[terminal_n==1] = 0
        adv_n = re_n + self.gamma*value_next_s - value_s
        if self.standardize_advantages:
            for i in range(self.n_drivers):
                adv_n[:,i] = (adv_n[:,i] - np.mean(adv_n[:,i])) / (np.std(adv_n[:,i]) + 1e-8)
        return adv_n

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size)
