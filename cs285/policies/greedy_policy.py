import numpy as np
from cs285.envs.city import MAX_CAND_NUM
import networkx as nx
import copy


class CentralGreedyPolicy:
    def __init__(self, city):
        self.matching_graph = nx.Graph()
        self.city = city

    def get_action(self):
        drivers = self.city.drivers
        actions = []
        for driver_ind, driver1 in enumerate(drivers):
            def dist_func(x, y):
                return self.city.travel_time(x, y, driver.driver_features)
            reward_list = []
            driver = copy.deepcopy(driver1)
            base_reward, _, _ = driver.take_action(0, dist_func)
            #print('calculate reward for ', driver_ind, 'base reward', base_reward)
            reward_list.append(base_reward)
            for i in range(MAX_CAND_NUM):
                driver = copy.deepcopy(driver1)  # here we make copies to preserve data
                new_reward,_,_ = driver.take_action(i+1, dist_func)
                reward_list.append(new_reward)
                #print('calculate reward for ', driver_ind, 'cand no.',i, 'reward:',new_reward,
                #  'candidte fee',driver.order_candidates[i].fee,'od', driver.order_candidates[i].ori, driver.order_candidates[i].dest,
                #  'id',driver.order_candidates[i].index)
            actions.append(reward_list.index(max(reward_list)))
        return actions
