from collections import OrderedDict
import pickle
import os
import sys
import time
import matplotlib.pyplot as plt
import gym
from gym import wrappers
import numpy as np
import torch
from cs285.infrastructure import pytorch_util as ptu

from cs285.infrastructure import utils
from cs285.infrastructure.logger import Logger

from cs285.agents.dqn_agent import DQNAgent
from cs285.infrastructure.dqn_utils import (
        get_wrapper_by_name,
        register_custom_envs,
)
from cs285.envs.city import City, MAX_CAND_NUM, MAX_CAP
# how many rollouts to save as videos to tensorboard
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40 # we overwrite this in the code below


class RL_Trainer(object):

    def __init__(self, params):

        #############
        ## INIT
        #############

        # Get params, create logger
        self.params = params
        self.logger = Logger(self.params['logdir'])

        # Set random seeds
        seed = self.params['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        ptu.init_gpu(
            use_gpu=not self.params['no_gpu'],
            gpu_id=self.params['which_gpu']
        )

        #############
        ## ENV
        #############

        # Make the gym environment
        #register_custom_envs()
        self.env = City((self.params['width'],self.params['height']),self.params['n_drivers'],self.params['n_restaurants'])
        """
        if 'env_wrappers' in self.params:
            # These operations are currently only for Atari envs
            self.env = wrappers.Monitor(self.env, os.path.join(self.params['logdir'], "gym"), force=True)
            self.env = params['env_wrappers'](self.env)
            self.mean_episode_reward = -float('nan')
            self.best_mean_episode_reward = -float('inf')
        if 'non_atari_colab_env' in self.params and self.params['video_log_freq'] > 0:
            self.env = wrappers.Monitor(self.env, os.path.join(self.params['logdir'], "gym"), force=True)
            self.mean_episode_reward = -float('nan')
            self.best_mean_episode_reward = -float('inf')
        """
        self.env.seed(seed)

        # import plotting (locally if 'obstacles' env)
        if not(self.params['env_name']=='obstacles-cs285-v0'):
            import matplotlib
            matplotlib.use('Agg')

        # Maximum length for episodes
        self.params['ep_len'] = self.params['ep_len'] or self.env.spec.max_episode_steps
        global MAX_VIDEO_LEN
        MAX_VIDEO_LEN = self.params['ep_len']

        # Is this env multi binary, or self.discrete?
        #multi_bi = isinstance(self.env.action_space, gym.spaces.MultiBinary)
        is_city = True
        # Are the observations images?
        img = False

        self.params['agent_params']['is_city'] = is_city

        # Observation and action sizes

        #ob_dim = self.env.observation_space.shape if img else self.env.observation_space.shape[0]
        #ac_dim = self.env.action_space.n if multi_bi else self.env.action_space.shape[0]
        #ob_dim = self.env.observation_space.shape[0]
        #ac_dim = self.env.action_space.shape[0]

        self.params['agent_params']['n_drivers'] = self.params['n_drivers']
        self.params['agent_params']['ac_dim'] = self.params['n_drivers']
        self.params['agent_params']['ob_dim'] = (self.params['n_drivers'],(3+2*MAX_CAP+5+5*MAX_CAND_NUM))
        self.params['agent_params']['shared_exp'] = self.params['shared_exp']
        self.params['agent_params']['shared_exp_lambda'] = self.params['shared_exp_lambda']
        self.params['agent_params']['size_ac'] = self.params['size_ac']
        self.params['agent_params']['size_cr'] = self.params['size_cr']
        # simulation timestep, will be used for video saving
        #if 'model' in dir(self.env):
        #    self.fps = 1/self.env.model.opt.timestep
        #elif 'env_wrappers' in self.params:
        #    self.fps = 30 # This is not actually used when using the Monitor wrapper
        #elif 'video.frames_per_second' in self.env.env.metadata.keys():
        #    self.fps = self.env.env.metadata['video.frames_per_second']
        #else:
        #    self.fps = 10


        #############
        ## AGENT
        #############

        agent_class = self.params['agent_class']
        self.agent = agent_class(self.env, self.params['agent_params'])

    def run_training_loop(self, n_iter, collect_policy, eval_policy,
                          initial_expertdata=None, relabel_with_expert=False,
                          start_relabel_with_expert=1, expert_policy=None):
        """
        :param n_iter:  number of (dagger) iterations
        :param collect_policy:
        :param eval_policy:
        :param initial_expertdata:
        :param relabel_with_expert:  whether to perform dagger
        :param start_relabel_with_expert: iteration at which to start relabel with expert
        :param expert_policy:
        """

        # init vars at beginning of training
        self.total_envsteps = 0
        self.start_time = time.time()

        print_period = 1000 if isinstance(self.agent, DQNAgent) else 1

        for itr in range(n_iter):
            if itr % print_period == 0:
                print("\n\n********** Iteration %i ************"%itr)

            # decide if videos should be rendered/logged at this iteration
            if itr % self.params['video_log_freq'] == 0 and self.params['video_log_freq'] != -1:
                self.logvideo = True
            else:
                self.logvideo = False

            # decide if metrics should be logged
            if self.params['scalar_log_freq'] == -1:
                self.logmetrics = False
            elif itr % self.params['scalar_log_freq'] == 0:
                self.logmetrics = True
            else:
                self.logmetrics = False

            # collect trajectories, to be used for training
            if isinstance(self.agent, DQNAgent):
                # only perform an env step and add to replay buffer for DQN
                self.agent.step_env()
                envsteps_this_batch = 1
                train_video_paths = None
                paths = None
            else:
                use_batchsize = self.params['batch_size']
                #if itr==0:
                #    use_batchsize = self.params['batch_size_initial']
                paths, envsteps_this_batch, train_video_paths = (
                    self.collect_training_trajectories(
                        itr, initial_expertdata, collect_policy, use_batchsize)
                )

            self.total_envsteps += envsteps_this_batch

            # relabel the collected obs with actions from a provided expert policy
            if relabel_with_expert and itr>=start_relabel_with_expert:
                paths = self.do_relabel_with_expert(expert_policy, paths)

            # add collected data to replay buffer
            self.agent.add_to_replay_buffer(paths)

            # train agent (using sampled data from replay buffer)
            if itr % print_period == 0:
                print("\nTraining agent...")
            all_logs = self.train_agent()

            # log/save
            if self.logvideo or self.logmetrics:
                # perform logging
                #map0,map1,map2 = self.agent.critic.check_map()
                #plt.figure()
                #plt.imshow(map0, cmap='hot')
                #plt.show()
                #plt.savefig('/content/cs285_f2020/homework_fall2020/hw3/data_city2/heatmap0.png')
                #plt.imshow(map1, cmap='hot')
                #plt.figure()
                #plt.show()
                #plt.savefig('/content/cs285_f2020/homework_fall2020/hw3/data_city2/heatmap1.png')
                #plt.figure()
                #plt.imshow(map2, cmap='hot')
                #plt.show()
                #plt.savefig('/content/cs285_f2020/homework_fall2020/hw3/data_city2/heatmap2.png')
                
                
                print('\nBeginning logging procedure...')
                if isinstance(self.agent, DQNAgent):
                    self.perform_dqn_logging(all_logs)
                else:
                    self.perform_logging(itr, paths, eval_policy, train_video_paths, all_logs)

                if self.params['save_params']:
                    self.agent.save('{}/agent_itr_{}.pt'.format(self.params['logdir'], itr))
            else:
                   self.perform_logging(itr, paths, eval_policy, train_video_paths, all_logs, False)

    ####################################
    ####################################

    def collect_training_trajectories(self, itr, initial_expertdata, collect_policy, num_transitions_to_sample, save_expert_data_to_disk=False):
        if itr == 0:
            if initial_expertdata is not None:
                paths = pickle.load(open(self.params['expert_data'], 'rb'))
                return paths, 0, None
        if save_expert_data_to_disk:
            num_transitions_to_sample = self.params['batch_size_initial']

        # collect data to be used for training
        print("\nCollecting data to be used for training...")
        paths, envsteps_this_batch = utils.sample_trajectories(self.env, collect_policy, num_transitions_to_sample, self.params['ep_len'])

        # collect more rollouts with the same policy, to be saved as videos in tensorboard
        train_video_paths = None
        if self.logvideo:
            print('\nCollecting train rollouts to be used for saving videos...')
            train_video_paths = utils.sample_n_trajectories(self.env, collect_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)

        if save_expert_data_to_disk and itr == 0:
            with open('expert_data_{}.pkl'.format(self.params['env_name']), 'wb') as file:
                pickle.dump(paths, file)

        return paths, envsteps_this_batch, train_video_paths

    def train_agent(self):
        all_logs = []
        for train_step in range(self.params['num_agent_train_steps_per_iter']):
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = self.agent.sample(self.params['train_batch_size'])
            train_log = self.agent.train(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)
            all_logs.append(train_log)
        return all_logs

    ####################################
    ####################################
    def perform_dqn_logging(self, all_logs):
        last_log = all_logs[-1]

        episode_rewards = get_wrapper_by_name(self.env, "Monitor").get_episode_rewards()
        if len(episode_rewards) > 0:
            self.mean_episode_reward = np.mean(episode_rewards[-100:])
        if len(episode_rewards) > 100:
            self.best_mean_episode_reward = max(self.best_mean_episode_reward, self.mean_episode_reward)

        logs = OrderedDict()

        logs["Train_EnvstepsSoFar"] = self.agent.t
        print("Timestep %d" % (self.agent.t,))
        if self.mean_episode_reward > -5000:
            logs["Train_AverageReturn"] = np.mean(self.mean_episode_reward)
        print("mean reward (100 episodes) %f" % self.mean_episode_reward)
        if self.best_mean_episode_reward > -5000:
            logs["Train_BestReturn"] = np.mean(self.best_mean_episode_reward)
        print("best mean reward %f" % self.best_mean_episode_reward)

        if self.start_time is not None:
            time_since_start = (time.time() - self.start_time)
            print("running time %f" % time_since_start)
            logs["TimeSinceStart"] = time_since_start

        logs.update(last_log)

        sys.stdout.flush()

        for key, value in logs.items():
            print('{} : {}'.format(key, value))
            self.logger.log_scalar(value, key, self.agent.t)
        print('Done logging...\n\n')

        self.logger.flush()

    def perform_logging(self, itr, paths, eval_policy, train_video_paths, all_logs, is_eval=True):

        last_log = all_logs[-1]

        #######################

        # collect eval trajectories, for logging
        if is_eval:
            print("\nCollecting data for eval...")
            eval_paths, eval_envsteps_this_batch = utils.sample_trajectories(self.env, eval_policy, self.params['eval_batch_size'], self.params['ep_len'])

        # save eval rollouts as videos in tensorboard event file
        if self.logvideo and train_video_paths != None:
            print('\nCollecting video rollouts eval')
            eval_video_paths = utils.sample_n_trajectories(self.env, eval_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)

            #save train/eval videos
            print('\nSaving train rollouts as videos...')
            self.logger.log_paths_as_videos(train_video_paths, itr, fps=self.fps, max_videos_to_save=MAX_NVIDEO,
                                            video_title='train_rollouts')
            self.logger.log_paths_as_videos(eval_video_paths, itr, fps=self.fps,max_videos_to_save=MAX_NVIDEO,
                                             video_title='eval_rollouts')

        #######################

        # save eval metrics
        if True:
            # returns, for logging
            if is_eval:
                train_returns = [path["reward"].sum() for path in paths]
                eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]
                # episode lengths, for logging
                train_ep_lens = [len(path["reward"]) for path in paths]
                eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]
            else:
                train_returns = [path["reward"].sum() for path in paths]
                eval_returns = train_returns
                train_ep_lens = [len(path["reward"]) for path in paths]
                eval_ep_lens = train_ep_lens
            

            # decide what to log
            logs = OrderedDict()
            logs["Eval_AverageReturn"] = np.mean(eval_returns)
            logs["Eval_StdReturn"] = np.std(eval_returns)
            logs["Eval_MaxReturn"] = np.max(eval_returns)
            logs["Eval_MinReturn"] = np.min(eval_returns)
            logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

            logs["Train_AverageReturn"] = np.mean(train_returns)
            logs["Train_StdReturn"] = np.std(train_returns)
            logs["Train_MaxReturn"] = np.max(train_returns)
            logs["Train_MinReturn"] = np.min(train_returns)
            logs["Train_AverageEpLen"] = np.mean(train_ep_lens)
            logs["Impossible_Moves"] = self.env.num_impossible_move

            logs["Train_EnvstepsSoFar"] = self.total_envsteps
            logs["TimeSinceStart"] = time.time() - self.start_time
            logs.update(last_log)

            if itr == 0:
                self.initial_return = np.mean(train_returns)
            logs["Initial_DataCollection_AverageReturn"] = self.initial_return

            # perform the logging
            for key, value in logs.items():
                print('{} : {}'.format(key, value))
                self.logger.log_scalar(value, key, itr)
            print('Done logging...\n\n')

            self.logger.flush()
