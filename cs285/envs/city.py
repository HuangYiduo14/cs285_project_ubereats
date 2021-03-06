import numpy as np
import pdb
from random import shuffle, choices
from cs285.infrastructure.tsp_utils import solve_tsp_dynamic_programming
import math
import itertools
import gym
from gym.utils import seeding
import copy
import random

BIG_NUM = 99999999
EPS = 1e-12
SEARCH_RADIUS = 3 # serach radius
DROP_PICK_RADIUS = EPS*10 # when a driver can pick/drop
MAX_ORDER_BUFFER = 300 # how many orders do we keep in buffer
MAX_CAND_NUM = 10 # number of candidate orders in the candidate set
MAX_CAP = 1 # max capacity of cars


class Restaurant:
    def __init__(self, x, y, attractiveness):
        self.x = x
        self.y = y
        self.attractiveness = attractiveness


class Order:
    def __init__(self, o, d, desired_delivery_time, pickup_ddl, fee, index):
        self.ori = o
        self.dest = d
        self.desired_delivery_time = desired_delivery_time
        self.desired_travel_time = desired_delivery_time - pickup_ddl
        self.fee = fee
        self.pickup_ddl = pickup_ddl
        self.is_picked = False
        self.picked_time = -1
        self.expected_drop_time = BIG_NUM
        self.index = index


class Driver:
    def __init__(self, x, y, driver_id, driver_features, max_capacity=4):
        self.x = x
        self.y = y
        self.time = 0
        self.driver_features = driver_features  # including speed
        self.id = driver_id

        self.max_capacity = max_capacity
        self.capacity = max_capacity

        self.trajectory = []
        self.traj_time = []

        self.last_drop_traj = []
        self.last_drop_traj_time = []

        self.next_is_drop = False  # whether the driver is on its way to drop or pick
        self.order_onboard = []  # list of Order
        self.order_drop_sequence = []  # list of index in order, order_drop_sequence[0] will be dropped first
        self.next_order_ind = BIG_NUM  # =order_drop_sequence[0]
        self.pend_order_dt = 0
        self.order_to_pick = None  # an Order object
        self.order_candidates = [Order([self.x, self.y], [self.x, self.y], BIG_NUM, BIG_NUM, 0,
                                       BIG_NUM) for _ in range(MAX_CAND_NUM)]  # lists of Order objects

    def shp_traj(self, location_list, dist_func, start_from_current=True):

        # our shortest_ham_path function will give a shortest path using the first node and the last node
        # therefore, we need to add the current location as the start point and a virtue point that is
        if len(location_list) == 0:
            return [], [], []

        if start_from_current:  # if start from current location, we add the current location
            nodes = [(self.x, self.y)] + location_list
            if len(nodes) == 2:
                return location_list, [0], [self.time + dist_func((self.x, self.y), location_list[0])]
        else:
            nodes = location_list
            if len(nodes) == 2:
                return location_list, [0, 1], [self.time + dist_func((self.x, self.y), location_list[0]),
                                               self.time + dist_func((self.x, self.y), location_list[0]) + dist_func(
                                                   location_list[0], location_list[1])]
            elif len(nodes) == 1:
                return location_list, [0], [self.time + dist_func((self.x, self.y), location_list[0])]
        n = len(nodes)
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist[i, j] = dist_func(nodes[i], nodes[j])
        dist[:, 0] = 0

        # shp_result = shortest_ham_path(n, dist)
        # import ipdb; ipdb.set_trace()
        shp_result = solve_tsp_dynamic_programming(dist)

        shp_result = shp_result[1]

        if start_from_current:
            ind_trajectory = shp_result[1:]
            ind_trajectory = [i - 1 for i in ind_trajectory]  # first point is the current location, ignored
            trajectory = [location_list[i] for i in ind_trajectory]
            travel_time = [dist[int(shp_result[i]), int(shp_result[i + 1])] for i in range(n - 1)]
            travel_time = np.cumsum(travel_time)
            arrival_time = travel_time + self.time
        else:
            ind_trajectory = shp_result
            trajectory = [location_list[i] for i in ind_trajectory]
            travel_time = [dist[int(shp_result[i]), int(shp_result[i + 1])] for i in range(n - 1)]
            travel_time.insert(0, 0)
            travel_time = np.cumsum(travel_time)
            arrival_time = travel_time + self.time + dist_func((self.x, self.y), location_list[0])

        # trajectory is the list of locations [[x0,y0],[x1,y1],...]
        # ind_trajectory is the index of the shortest path
        # arrival_time is the arrival time list [t0,t1,...]
        return trajectory, ind_trajectory, arrival_time

    def pickup_order(self, new_order):
        assert abs(self.x - new_order.ori[0]) + abs(self.y - new_order.ori[1]) < DROP_PICK_RADIUS
        if new_order.fee < EPS:
            return
        self.capacity -= 1
        assert self.capacity >= 0
        # TODO: delete print function
        #print('time:',self.time)
        #print('PICKUP: driver {0} picked order {1} at {2},{3} to {4}'.format(self.id, new_order.index, self.x, self.y, new_order.dest))
        new_order.is_picked = True
        new_order.picked_time = self.time
        self.next_is_drop = True
        self.order_onboard.append(new_order)
        self.order_to_pick = None
        self.pend_order_dt = 0
        # self.order_to_pick_fee = 0

    def drop_order(self):
        assert abs(self.x - self.order_onboard[self.next_order_ind].dest[0]) + \
               abs(self.y - self.order_onboard[self.next_order_ind].dest[1]) < DROP_PICK_RADIUS
        self.capacity += 1
        assert self.capacity <= self.max_capacity
        # TODO: delete print function
        #print('time:',self.time)
        #print('DROP: driver {0} droped order {1} at {2},{3}'.format(self.id, self.order_onboard[self.next_order_ind].index, self.x, self.y))
        order_dropped = self.order_onboard.pop(self.next_order_ind)
        self.order_drop_sequence = self.order_drop_sequence[1:]
        self.order_drop_sequence = [i if i < self.next_order_ind else i - 1 for i in self.order_drop_sequence]
        if len(self.order_drop_sequence) > 0:
            self.next_order_ind = self.order_drop_sequence[0]
        else:
            self.next_order_ind = BIG_NUM
            self.next_is_drop = False
        return order_dropped

    #def agent_reward(self, b_tn, c_tn, delta_tn, fee, is_lost_order, lost_order_fee, old_pend_order_dt):
        # TODO: change reward weight here in experiments
        # assert delta_tn>=-EPS
    #    if is_lost_order and self.order_to_pick.fee > EPS:
    #        return b_tn - .0001 * c_tn - 2. * (delta_tn - old_pend_order_dt) + 1. * (
    #                fee - lost_order_fee)  # if we give up the current pickup order, we will have extra penalty
    #    else:
    #        return b_tn - .0001 * c_tn - 2. * delta_tn + 1. * fee
 # changed here
    def agent_observe(self, dist_func):
        """
        :return: observation for this agent
            [available seats, current_x, current_y,
                 trajectory_x0, trajectory_y0,
                 trajectory_x1, trajectory_y1,
                 ...
                 trajectory_xm, trajectory_ym,
                order_to_pick: ori_x, ori_y, dest_x, dest_y, fee
                order_candidate_1: ori_x, ori_y, dest_x, dest_y, fee
                order_candidate_2: ori_x, ori_y, dest_x, dest_y, fee
                order_candidate_MAX_CAND_NUM: ori_x, ori_y, dest_x, dest_y, fee
            ]
        m = MAX_CAP
        """
        if self.next_is_drop:
            traj = self.trajectory
            traj_time = self.traj_time
            unpicked_fee = 0
        else:
            order_locations = [order.dest for order in self.order_onboard]
            traj, _, traj_time = self.shp_traj(order_locations, dist_func)
            if self.order_to_pick:
                unpicked_fee = self.order_to_pick.fee
            else:
                unpicked_fee = 0
        has_unpicked_order = not self.next_is_drop
        order_space_time_traj = [[traj[i][0], traj[i][1]] for i in range(len(traj))]
        # if we want to make a 1d vector:
        order_space_time_traj = list(itertools.chain(*order_space_time_traj))
        if len(order_space_time_traj) > 0:
            last_location = [traj[-1][0], traj[-1][1]]
        else:
            last_location = [self.x, self.y]
        # pad empty seats with virtual orders
        valid_traj_length = len(traj)
        for extra_order in range(MAX_CAP - valid_traj_length):
            order_space_time_traj += last_location

        if self.order_to_pick:
            order_to_pick_info = [
                self.order_to_pick.ori[0], self.order_to_pick.ori[1],
                self.order_to_pick.dest[0], self.order_to_pick.dest[1],
                self.order_to_pick.fee
            ]
        else:
            order_to_pick_info = [self.x, self.y, self.x, self.y, 0]

        order_candidates_info = []
        for i in range(MAX_CAND_NUM):
            if self.order_candidates[i]:
                order_candidates_info += [
                    self.order_candidates[i].ori[0], self.order_candidates[i].ori[1],
                    self.order_candidates[i].dest[0], self.order_candidates[i].dest[1],
                    self.order_candidates[i].fee
                ]
            else:
                order_candidates_info += [self.x, self.y, self.x, self.y, 0]

        return [self.capacity ,self.x, self.y] \
               + order_space_time_traj + order_to_pick_info + order_candidates_info

    def take_action(self, action, dist_func):
        """
        :param action:
        action = 0: we will not pick new orders
        action = 1: we will pick self.order_candidate[0]
        ...
        action = MAX_CAND_NUM: we will pick self.order_candidate[MAX_CAND_NUM-1]
        :return: reward for one driver and new space-time trajectory
        b_tn, c_tn and delta_tn: see DeepPool (7)
        """
        fee = 0
        is_lost_order = False
        b_tn = len(self.order_onboard)
        c_tn = 0
        delta_tn = 0
        lost_order = None
        lost_order_fee = 0
        old_pend_order_dt = self.pend_order_dt

        if action >= 1 and self.capacity >= 1:
            this_order = self.order_candidates[action - 1]
            x0, y0 = this_order.ori
            x1, y1 = this_order.dest
            fee = this_order.fee
            # print('DISPATCH: driver {0} goes to {1},{2} from {3},{4}'.format(self.id, x0, y0, self.x,self.y))
            order_locations = [order.dest for order in
                               self.order_onboard]  # noting here we only consider orders on board
            if abs(x0 - x1) + abs(y0 - y1) > EPS:  # case 1: we know which order to pick
                location_list = [(x0, y0)] + order_locations + [(x1, y1)]
            else:
                location_list = [(x0, y0)] + order_locations  # case 2: we only know the origin of the new order
            trajectory, ind_trajectory, arrival_time = self.shp_traj(location_list, dist_func, start_from_current=False)
            self.order_drop_sequence = [i - 1 for i in ind_trajectory if
                                        i >= 1]  # the first point is a pickup point so we ignore it
            self.trajectory = trajectory
            self.traj_time = arrival_time
            if len(self.order_drop_sequence) > 0:
                self.next_order_ind = self.order_drop_sequence[0]
            else:
                self.next_order_ind = -1
            if (not self.next_is_drop) and (self.order_to_pick) and (self.order_to_pick.fee > EPS):
                # if the car is going to pick an order when dispatched to new area, unpicked order will be lost
                is_lost_order = True
                lost_order = self.order_to_pick
                lost_order_fee = lost_order.fee
            # calculate the difference in time
            delta_tn = 0
            for i, o in enumerate(self.order_drop_sequence):  # i is the sequence while o is the index in order list
                dt = 0
                if o <= len(self.order_onboard) - 1:
                    dt = self.traj_time[i + 1] - self.order_onboard[
                        o].expected_drop_time  # the first traj time on the path is a pickup point
                    self.order_onboard[o].expected_drop_time = self.traj_time[i + 1]
                elif o == len(self.order_onboard):
                    if is_lost_order:
                        self.pend_order_dt = self.traj_time[i + 1] - self.time \
                                             - dist_func((self.x, self.y), this_order.ori) \
                                             - dist_func(this_order.ori, this_order.dest)
                    else:
                        self.pend_order_dt = 0
                    # print('dt',dt)
                    dt = self.pend_order_dt
                    this_order.expected_drop_time = self.traj_time[i + 1]
                else:
                    assert False
                delta_tn += dt
            self.next_is_drop = False
            self.order_to_pick = this_order
            
            c_tn = dist_func((self.x, self.y), (x0, y0))
        reward0 = -1 -b_tn*0.3

        #reward = self.agent_reward(b_tn, c_tn, delta_tn, fee, is_lost_order, lost_order_fee, old_pend_order_dt)
        #if action >= 1 and self.capacity<1:
        #   reward = -1
        # print('impossible movement')
        return reward0, is_lost_order, lost_order

    def add_to_candidate(self, new_order_candidates):
        self.order_candidates = new_order_candidates

    def move_one_step(self):
        # move the vehicle in one time step according to its current_orders, shp trajectory and current location
        if len(self.trajectory) == 0:
            next_x = self.x
            next_y = self.y
        else:
            next_x = self.trajectory[0][0]
            next_y = self.trajectory[0][1]
        speed0 = self.driver_features['speed']
        dx = next_x - self.x
        dy = next_y - self.y
        move_along_y = np.random.rand() > 0.5
        if move_along_y:
            speed = min(speed0, abs(dy))
            dxy = (0, speed * (dy > 0) - speed * (dy < 0))
        else:
            speed = min(speed0, abs(dx))
            dxy = (speed * (dx > 0) - speed * (dx < 0), 0)
        if abs(dx) < EPS:
            speed = min(speed0, abs(dy))
            dxy = (0, speed * (dy > 0) - speed * (dy < 0))
        if abs(dy) < EPS:
            speed = min(speed0, abs(dx))
            dxy = (speed * (dx > 0) - speed * (dx < 0), 0)
        # print('driver {0} from {1},{2}'.format(self.id, self.x, self.y))
        self.x += dxy[0]
        self.y += dxy[1]
        self.time += 1

        # print some informations
        # TODO: delete these lines
        # print('dx dy', dx, dy)
        #if abs(self.x - round(self.x))+ abs(self.y-round(self.y))<EPS:
            #print('time:',self.time)
            #print('dxy', dxy)
            #print('driver {0} moved to {1},{2}'.format(self.id, self.x, self.y))
            #print('new traj', self.trajectory)
            #print('new traj_time', self.traj_time)
            #print('new order sequence', self.order_drop_sequence)
            #print('orders on board', [order.index for order in self.order_onboard])
            #if self.order_to_pick:
                #print('order to pick', self.order_to_pick.ori, self.order_to_pick.dest)
        picked_order_val, drop_order_val = 0,0
        while True:
            # check all possible points along the trajectory that can be picked or droped
            if len(self.trajectory) > 0 and (
                    abs(self.x - self.trajectory[0][0]) + abs(self.y - self.trajectory[0][1]) < DROP_PICK_RADIUS):
                self.trajectory = self.trajectory[1:]
                self.traj_time = self.traj_time[1:]
                if not self.next_is_drop:
                    picked_order_val += self.order_to_pick.fee
                    self.pickup_order(self.order_to_pick)

                else:
                    order_dropped = self.drop_order()
                    drop_order_val += order_dropped.fee
            else:
                break
        return picked_order_val, drop_order_val


class City(gym.Env):
    def __init__(self, size, n_drivers, n_restaurants, seed=1, even_demand_distr=True, demand_distr=None,
                 time_horizon=100):
        self.seed0 = seed
        random.seed(self.seed0)
        np.random.seed(self.seed0)
        self.seed(self.seed0)
        # initialize the city map
        self.time = 0
        self.time_horizon = time_horizon
        self.width = size[0]  # we consider a rectangle city
        self.height = size[1]
        self.n_drivers = n_drivers
        self.n_restaurants = n_restaurants
        self.drivers = []
        self.total_orders = 0
        self.driver_map = np.zeros((self.width, self.height))
        self.request_map = np.zeros((self.width, self.height))
        self.demand_expectation = np.zeros(
            (self.width, self.height))  # for now, we assume the demand origin is stationary
        self.new_available_map = np.zeros((time_horizon, self.width, self.height))
        # driver map is the number of availabe drivers at each location
        self.capacity_profile = [MAX_CAP, MAX_CAP, MAX_CAP]
        self.speed_profile = [0.5, 0.5, 0.5]

        # information
        self.num_lost_orders = 0
        self.num_impossible_move = 0
        self.order_droped_val = 0
        self.order_picked_val = 0
        self.order_lost_val = 0
        self.order_pend_val = 0
        self.total_reward = 0

        # initialize drivers, restaurants and demand(home)
        self.state = []
        for i in range(n_drivers):
            def dist_func(x, y):
                return self.travel_time(x, y, self.drivers[i].driver_features)

            type = np.random.choice([0, 1, 2], p=[0.1, 0.8, 0.1])
            home_x = math.floor(np.random.rand() * self.width)
            home_y = math.floor(np.random.rand() * self.height)
            driver_features = {'type': type, 'home_address': [home_x, home_y], 'is_last_order': False,
                               'speed': self.speed_profile[type]}
            self.drivers.append(Driver(home_x, home_y, driver_id=i, driver_features=driver_features,
                                       max_capacity=self.capacity_profile[type]))
            self.driver_map[home_x, home_y] += 1
            self.state.append(self.drivers[i].agent_observe(dist_func))

        self.restaurants = [
            Restaurant(round(np.random.rand() * self.width /5 + self.width/2), round(np.random.rand() * self.height/5+self.height/2),
                       0.2* (np.random.rand() / 2. + 0.5)) for _ in range(n_restaurants-4)]
        self.restaurants.append(Restaurant(0,0,0.2 * (np.random.rand() / 2. + 0.5)))
        self.restaurants.append(Restaurant(0,9,0.2 * (np.random.rand() / 2. + 0.5)))
        self.restaurants.append(Restaurant(9,0,0.2 * (np.random.rand() / 2. + 0.5)))
        self.restaurants.append(Restaurant(9,9,0.2 * (np.random.rand() / 2. + 0.5)))
        for restaurant in self.restaurants:
            print('restaurant', restaurant.x, restaurant.y, restaurant.attractiveness)
            self.demand_expectation[restaurant.x, restaurant.y] += restaurant.attractiveness

        if even_demand_distr:
            self.demand_dest_distr = {(x, y): 1 / self.width / self.height for x in range(self.width) for y in
                                      range(self.height)}
        else:
            self.demand_dest_distr = demand_distr
            # demand_dest_distr is the pdf of demand (a dictionary)
        # initialize order buffer
        self.order_buffer = []
        self.all_orders = []

        # initialize action space


        obs_lb_one_driver = [0, 0, 0] + [0 for _ in range(2 * MAX_CAP)] + \
                            [0, 0, 0, 0, 0] + [0, 0, 0, 0, 0]*MAX_CAND_NUM
        obs_ub_one_driver = [MAX_CAP, 10, 10] + [10 for _ in range(2 * MAX_CAP)] + \
                            [10, 10, 10, 10, BIG_NUM] + [10, 10, 10, 10, BIG_NUM]*MAX_CAND_NUM
        self.observation_space = gym.spaces.Tuple([
            gym.spaces.Box(low=np.array(obs_lb_one_driver), high=np.array(obs_ub_one_driver)) for _ in
             range(self.n_drivers)])
        self.action_space = gym.spaces.Tuple([gym.spaces.Discrete(MAX_CAND_NUM+1) for _ in range(self.n_drivers)])

        self.state = np.array(self.state)

        """
        observation space for one driver:
            [available seats, current_x, current_y,
                 trajectory_x0, trajectory_y0,
                 trajectory_x1, trajectory_y1,
                 ...
                 trajectory_xm, trajectory_ym,  # kappa_t^i
                 order_to_pick: ori_x, ori_y, dest_x, dest_y, fee
                order_candidate: ori_x, ori_y, dest_x, dest_y, fee
            ]
        """
        # return self.state

    def reset(self):
        self.__init__((self.width, self.height), self.n_drivers, self.n_restaurants, self.seed0)
        return self.state

    def seed(self, seed=None):
        seeding.np_random(seed)

    def travel_time(self, xy1, xy2, driver_features):
        x1, y1 = xy1
        x2, y2 = xy2
        speed = self.speed_profile[driver_features['type']]
        travel_time = (abs(x2 - x1) + abs(y2 - y1)) / speed
        return math.ceil(travel_time)

    def travel_cost(self, xy1, xy2, driver_features):
        x1, y1 = xy1
        x2, y2 = xy2
        travel_cost = self.travel_time(xy1, xy2, driver_features)
        if driver_features['is_last_order']:
            home_add = driver_features['home_address']
            travel_cost += 0.1 * (self.travel_time((x2, y2), (home_add[0], home_add[1]), driver_features)
                                  - self.travel_time((x1, y1), (home_add[0], home_add[1]), driver_features))
        return travel_cost

    def fee(self, expected_delivery_time):
        fee = 2 * expected_delivery_time + 3
        return fee

    def late_penalty(self, late_time, fee):
        if late_time <= 0:
            return 0
        return min(fee, late_time * fee)

    def lost_order_penalty(self, fee):
        return 0.1 * fee

    def expected_delivery_time(self, restaurant_x, restaurant_y, home_x, home_y):
        return 2. * self.travel_time((restaurant_x, restaurant_y), (home_x, home_y), {'type': 1})

    def order_generate(self):
        # we can generate demand using the following procedure:
        # step 1. generate some active restaurant from a poisson process according to the 'attractiveness'
        # step 2. for each 'active' restaurant, generate the destination according to the demand_distr
        if len(self.order_buffer) > MAX_ORDER_BUFFER:
            return []
        active_restaurants = []
        new_orders = []
        for i in range(self.n_restaurants):
            if np.random.rand() < self.restaurants[i].attractiveness:
                active_restaurants.append(i)
        for i in active_restaurants:
            dict_keys = list(self.demand_dest_distr.keys())
            restaurant_x = self.restaurants[i].x
            restaurant_y = self.restaurants[i].y
            home = [restaurant_x, restaurant_y]
            while abs(home[0] - restaurant_x) + abs(home[1] - restaurant_y) < EPS:
                home_ind = np.random.choice(list(range(len(dict_keys))), p=list(self.demand_dest_distr.values()))
                home = dict_keys[home_ind]
            expected_delivery_time = self.expected_delivery_time(restaurant_x, restaurant_y, home[0], home[1])
            fee = self.fee(expected_delivery_time)

            if expected_delivery_time > 1:
                new_order = Order((restaurant_x, restaurant_y), (home[0], home[1]), self.time + expected_delivery_time,
                                  self.time + expected_delivery_time / 2, fee, self.total_orders)
                self.total_orders += 1
                new_orders.append(new_order)
        self.order_buffer = self.order_buffer + new_orders
        self.all_orders = self.all_orders + new_orders
        return new_orders

    def step(self, actions):
        # actions: list of actions for each driver
        # e.g. [0,1,2,1,...]
        self.order_generate()
        #print('time:', self.time, '=' * 30)
        rewards = []
        observations = []
        done = []
        info = {}
        # city_observation = {'drivers': self.drivers, 'order_buffer': self.order_buffer}
        order_candidate_selected = []
        restaurant_loc_list = [(r.x, r.y) for r in self.restaurants]
        restaurant_attractive_list = [r.attractiveness for r in self.restaurants]
        driver_seq = list(range(self.n_drivers))
        shuffle(driver_seq)
        for i in driver_seq:
            def dist_func(x, y):
                return self.travel_time(x, y, self.drivers[i].driver_features)

            ##############################################################
            # randomly assign new orders to each driver
            ##############################################################

            num_orders_found = 0
            candidate_orders = []
            for search_mult in range(1, 1 + dist_func([0, 0], [10, 10]) // SEARCH_RADIUS):
                # gradully increase the search radius
                if num_orders_found >= MAX_CAND_NUM:
                    break
                current_xy = [self.drivers[i].x, self.drivers[i].y]
                potential_orders = [order for order in self.order_buffer if
                                    search_mult * SEARCH_RADIUS >= dist_func(order.ori, current_xy) >= (
                                            search_mult - 1) * SEARCH_RADIUS]
                shuffle(potential_orders)
                #    pdb.set_trace()
                for cand_order in potential_orders:
                    if cand_order not in order_candidate_selected:
                        # to avoid conflicts, we will only assing one order to one driver
                        candidate_orders.append(cand_order)
                        order_candidate_selected.append(cand_order)
                        num_orders_found += 1
                        if num_orders_found >= MAX_CAND_NUM:
                            break
            # if no new order available, assign re-dispatch orders
            if num_orders_found < MAX_CAND_NUM:
                # print('Fake order assigned')
                while num_orders_found < MAX_CAND_NUM:
                    dispatched_loc = choices(restaurant_loc_list, restaurant_attractive_list)
                    dispatched_loc = dispatched_loc[0]
                    if abs(dispatched_loc[0] - self.drivers[i].x) + abs(
                            dispatched_loc[1] - self.drivers[i].y) > DROP_PICK_RADIUS:
                        candidate_orders.append(Order(dispatched_loc, dispatched_loc, self.time, BIG_NUM, 0, BIG_NUM))
                        num_orders_found += 1
            assert len(candidate_orders) == MAX_CAND_NUM
            self.drivers[i].add_to_candidate(candidate_orders)

            ###########################################################
            # take action
            ###########################################################

            # observe what happens before taking action
            # observations.append(self.drivers[i].agent_observe(dist_func))
            observations.append(self.drivers[i].agent_observe(dist_func))

            action_i = actions[i]
            this_reward0, is_lost_order, lost_order = self.drivers[i].take_action(action_i, dist_func)
            if is_lost_order:
                self.num_lost_orders += 1
            if action_i >= 1 and self.drivers[i].capacity < 1:
                self.num_impossible_move += 1
            if action_i >= 1 and self.drivers[i].capacity >= 1:
                order_i = self.drivers[i].order_to_pick
                if order_i.fee > EPS:
                    self.order_buffer.remove(order_i) # remove picked order from city buffer
                    self.order_pend_val += order_i.fee
                # print('driver {0} will take order {1},{2} -> {3},{4}'.format(i, order_i.ori[0], order_i.ori[1], order_i.dest[0], order_i.dest[1]))

                if is_lost_order and lost_order.fee > EPS:  # if order is given up
                    self.order_buffer.append(lost_order)  # return the order to city buffer
                    self.order_lost_val += lost_order.fee
            

            #################################################################
            # move one step and decide pickup/drop orders
            #################################################################
            picked_order_val, drop_order_val = self.drivers[i].move_one_step()
            self.order_picked_val += picked_order_val
            self.order_droped_val += drop_order_val

            # new reward structure
            rewards.append(this_reward0+drop_order_val)

            self.total_reward += this_reward0 + drop_order_val

            done.append(False)
        # if an other has not been picked up after ddl, delete it from buffer
        self.order_buffer = [order for order in self.order_buffer if order.pickup_ddl > self.time]
        self.time += 1
        self.state = observations
        if self.time % 100 == 0:
            print('impossible move:', self.num_impossible_move)
            print('lost orders:', self.num_lost_orders)
            print('total reward:',self.total_reward)
            print('orders left:',len(self.order_buffer))
        info = {'picked_fee':self.order_picked_val, 'dropped_fee':self.order_droped_val, 'pending_income':self.order_pend_val,
                'lost_fee':self.order_lost_val, 'num_lost_orders':self.num_lost_orders, 'num_impossible_move': self.num_impossible_move}
        return np.array(observations), np.array(rewards), done[0], info
