import numpy as np
from envs.city import City, Order
import networkx as nx

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
        for driver_ind, driver in enumerate(drivers):
            _,_,_,baseline_reward,_,_ = self.city.vrp_orders(driver, list(driver.current_orders.values()))
            for order_key, order in enumerate(orders):
                _,_,_,expected_reward,_,_ = self.city.vrp_orders(driver, list(driver.current_orders.values()) + [order])
                if expected_reward>=baseline_reward and len(self.city.drivers[driver_ind].current_orders)+1 < self.city.drivers[driver_ind].max_orders:
                    self.matching_graph.add_edge('D{0}'.format(driver_ind), 'O{0}'.format(order_key), weight = expected_reward-baseline_reward)
        return self.matching_graph

    def bipartite_matching(self):
        actions = [None for i in range(self.city.n_drivers)]
        driver_node_key = {'D{0}'.format(i):i for i in range(self.city.n_drivers)}
        order_node_key = {'O{0}'.format(i):i for i in range(len(self.city.order_buffer))}
        matching = nx.max_weight_matching(self.matching_graph)
        for node1, node2 in matching:
            if node1[0] =='D':
                this_driver = driver_node_key[node1]
                this_order = order_node_key[node2]
            else:
                this_driver = driver_node_key[node2]
                this_order = order_node_key[node1]
            actions[this_driver] = self.city.order_buffer[this_order]
        for driver_key, action in enumerate(actions):
            if action is None:
                if len(self.city.drivers[driver_key].current_orders)==0:
                    actions[driver_key] = Order(None,(0,0),0,0,0,-1,is_repostion=True)
        return actions

