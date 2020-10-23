import numpy as np

class CentralGreedyPolicy:
    def __init__(self):
        self.matching_graph = None

    def get_action(self, drivers, orders):
        # TODO: given the current drivers and orders, calculate greedy assignment
        self.estimate_rewards(drivers,orders)
        action = self.bipartite_matching()
        return action

    def l2_distance(a_x, a_y, b_x, b_y):
        a = np.array((a_x, a_y))
        b = np.array((b_x, b_y))
        return numpy.linalg.norm(a-b)
        
    def estimate_rewards(self, drivers, orders):
        # TODO: calculate all possible assignment rewards for each driver
        # self.matching_graph[driver i, order k] = reward for i to take k at this time step
        for driver in drivers:
            for order in orders: 
                dist = l2_distance(driver.x, driver.y, order.x, order.y)
                if dist <= self.driver_search_radius and dist <= self.order_search_radius: 
                    _ = driver.try_new_order()
        
        
        
        return self.matching_graph

    def bipartite_matching(self):
        # TODO: max matching from drivers to orders
        return action
