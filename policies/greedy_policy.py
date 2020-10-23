import numpy as np

class CentralGreedyPolicy:
    def __init__(self):
        self.matching_graph = None

    def get_action(self, drivers, orders):
        # TODO: given the current drivers and orders, calculate greedy assignment
        self.estimate_rewards(drivers,orders)
        action = self.bipartite_matching()
        return action

    def estimate_rewards(self, drivers, orders):
        # TODO: calculate all possible assignment rewards for each driver
        # self.matching_graph[driver i, order k] = reward for i to take k at this time step
        return self.matching_graph

    def bipartite_matching(self):
        # TODO: max matching from drivers to orders
        return action