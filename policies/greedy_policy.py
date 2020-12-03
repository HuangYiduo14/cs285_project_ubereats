import numpy as np
from envs.city import City, Order
import networkx as nx
import copy

class CentralGreedyPolicy:
    def __init__(self, city):
        self.matching_graph = nx.Graph()
        self.city = city

    def get_action(self):
        self.estimate_rewards()
        actions = self.bipartite_matching()
        return actions

    def estimate_rewards(self):
        drivers = self.city.drivers
        orders = self.city.order_buffer
        for driver_ind, driver1 in enumerate(drivers):
            driver = copy.deepcopy(driver1)
            def dist_func(x,y):
                return self.city.travel_time(x,y, driver.driver_features)

            baseline_reward,_ = driver.take_action([0, 0, 0, 0, 0, 0], dist_func)
            for order_key, order in enumerate(orders):
                # if we can add this order to this driver, then try
                if driver.capacity>0:
                    expected_reward, _ = driver.take_action([1, order.ori[0], order.ori[1], order.dest[0], order.dest[1], order.fee], dist_func)
                    driver.order_to_pick = order
                    if expected_reward >= baseline_reward: # only non-negative reward gains are considered
                        self.matching_graph.add_edge('D{0}'.format(driver_ind), 'O{0}'.format(order_key),
                                                     weight = expected_reward - baseline_reward)
        return self.matching_graph

    def bipartite_matching(self):
        actions = [None for i in range(self.city.n_drivers)]
        driver_node_key = {'D{0}'.format(i): i for i in range(self.city.n_drivers)}
        order_node_key = {'O{0}'.format(i): i for i in range(len(self.city.order_buffer))}
        matching = nx.max_weight_matching(self.matching_graph) # do max weight matching for this graph
        for node1, node2 in matching:
            if node1[0] == 'D':
                this_driver = driver_node_key[node1]
                this_order = order_node_key[node2]
            else:
                this_driver = driver_node_key[node2]
                this_order = order_node_key[node1]
            actions[this_driver] = (1, self.city.order_buffer[this_order])

        for driver_key, action in enumerate(actions):
            if action is None:
                actions[driver_key] = (0, Order(None, (0, 0), 0, 0, 0, -1))

        return actions
