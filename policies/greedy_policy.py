import numpy as np
from envs.city import Order
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
            driver = copy.deepcopy(driver1)  # here we make copies to preserve data
            def dist_func(x, y):
                return self.city.travel_time(x, y, driver.driver_features)
            baseline_reward, _, _ = driver.take_action(0, dist_func)
            new_reward, _, _ = driver.take_action(1, dist_func)
            print('calculate reward for ', driver_ind, 'new:',new_reward, 'base:',baseline_reward,
                  'candidte fee',driver.order_candidate.fee,'od', driver.order_candidate.ori, driver.order_candidate.dest,
                  'id',driver.order_candidate.index)
            if new_reward>baseline_reward and driver.capacity>=1:
                actions.append(1)
            else:
                actions.append(0)
        return actions
