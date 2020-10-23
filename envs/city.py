import numpy as np

class Restaurant:
    def __init__(self, x, y, attractiveness):
        self.x = x
        self.y = y
        self.attractiveness = attractiveness


class Order:
    def __init__(self, o, d, end_time, fee, index):
        self.ori = o
        self.dest = d
        self.end_time = end_time
        self.fee = fee
        self.index = index


class Driver:
    def __init__(self, x, y, driver_features, order_radius=1, driver_radius=2):
        self.x = x
        self.y = y
        self.driver_features = driver_features
        self.current_orders = []  # we keep a 2D list of unfinished orders (order indices, travel times from each driver's location)
        self.current_orders.append([]) 
        self.is_idle = True
        self.trajectory = []  # we keep a record of VRP result trajectory

        self.order_search_radius = order_radius
        self.driver_search_radius = driver_radius

    def vrp_routing(self, orders):
   #def vrp_routing(self, order): ?
        # TODO: solve the vrp from current location to complete all Orders in self.current_orders
        # there could be a warm start vrp algorithm
        # (i.e. we add only one order to current trajectory and we keep most of the trajectory)
        return travel_time

    def try_new_order(self, order):
        # TODO: add this order to the current_order list to a temporary order list and do vrp routing
        # this function only calculate the vrp results, it is not the actual movement
        travel_time = vrp_routing(order)
        self.current_orders.append(order.index, travel_time)  
        return 0 

    def take_order(self, order):
        # TODO: update trajectory
        return 0

    def move_one_step(self):
        # TODO: move the vehicle in one time step according to its current_orders, vrp trajectory and current location
        # if some orders are completed
        # if we become idle
        return 0


class City:
    def __init__(self, size, n_drivers, n_restaurants, seed=1, even_demand_distr=True, demand_distr=None):
        self.seed = seed
        np.random.seed(seed)
        # initialize the city map
        self.time = 0
        self.width = size[0]  # we consider a rectangle city
        self.height = size[1]
        self.n_drivers = n_drivers
        self.n_restaurants = n_restaurants
        self.drivers = []


        # initialize drivers, restaurants and demand(home)
        for _ in range(n_drivers):
            type = np.random.choice([0, 1, 2])
            home_x = np.random.rand() * self.width
            home_y = np.random.rand() * self.height
            driver_features = {'type': type, 'home_address': [home_x, home_y], 'is_last_order': False}
            self.drivers.append(Driver(home_x, home_y, driver_features))


        self.restaurants = [
            Driver(np.random.rand() * self.width / 2., np.random.rand() * self.height / 2., np.random.rand()) for _ in
            range(n_restaurants)]  # we assume restaurant are in the center of the city
        if even_demand_distr:
            self.demand_distr = lambda x, y: 1
            # demand_distr is the pdf of demand
            # we can generate demand using the following procedure:
            # step 1. generate some active restaurant from a poisson process according to the 'attractiveness'
            # step 2. for each 'active' restaurant, generate the destination according to the demand_distr

        # initialize order buffer
        self.order_buffer = []

    def travel_time(self, x1, y1, x2, y2, driver_features):
        # TODO: calculation of travel time according to the drivers features given the city layout
        return travel_time

    def order_generate(self):
        # TODO: generate a list of new orders and add to self.order_buffer
        return orders

    def step(self, actions):
        # TODO: do one simulation step
        # given actions for each driver, update all drivers
        # generate reward
        # state: all variables in this city, however, a driver can only observe variables within certain radius

        # action: list of order index, 0 means no new order, e.g. [0,1,2,0] (car 0: no, car 1: order 1, car 2: order 2, car 3: no)


        # TODO: I think we can simplify the reward function, we can calculate the reward if the driver doesn't take new orders
        # TODO: If the driver takes one new order, then we generate reward
        # TODO: Therefore, we only generate reward when we take an new order
        self.time += 1
        return reward, observations

    def reset(self):
        # TODO: reset all drivers and orders
        return 0
