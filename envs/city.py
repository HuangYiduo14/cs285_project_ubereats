import numpy as np
import pdb
from infrastructure.tsp_utils import shortest_ham_path
import math
BIG_NUM = 99999999
EPS = 1e-12

class Restaurant:
    def __init__(self, x, y, attractiveness):
        self.x = x
        self.y = y
        self.attractiveness = attractiveness


class Order:
    def __init__(self, o, d, desired_delivery_time, pickup_ddl, fee, index):
        self.ori = o
        self.dest = d
        #self.desired_delivery_time = desired_delivery_time
        #self.desired_travel_time = desired_delivery_time - pickup_ddl
        self.fee = fee
        #self.pickup_ddl = pickup_ddl
        self.is_picked = False
        self.picked_time = -1
        self.expected_drop_time = BIG_NUM
        self.index = index


class Driver:
    def __init__(self, x, y, driver_id, driver_features, max_capacity=4):
        self.x = x
        self.y = y
        self.time = 0
        self.driver_features = driver_features
        self.id = driver_id

        self.max_capacity = max_capacity
        self.capacity = max_capacity

        self.trajectory = []
        self.traj_time = []

        self.next_is_drop = False
        self.order_onboard = []
        self.order_drop_sequence = []
        self.next_order_ind = -1
        self.order_to_pick = None
        self.order_to_pick_fee = 0


    def shp_traj(self, location_list, dist_func, start_from_current=True):
        if start_from_current:
            nodes = [(self.x,self.y)] + location_list
        else:
            nodes = location_list
        n = len(nodes) + 1
        dist = {(i,j):dist_func(nodes[i], nodes[j]) for i in range(n-1) for j in range(n-1)}
        dist.update({(i,i):0 for i in range(n-1)})
        dist.update({(i,n):0 for i in range(n)})
        shp_result = shortest_ham_path(n,dist)

        if start_from_current:
            ind_trajectory = shp_result[1][1:-1]
            ind_trajectory = [i-1 for i in ind_trajectory]
            trajectory = [location_list[i] for i in ind_trajectory]
            travel_time = [dist[int(shp_result[1][i]), int(shp_result[1][i+1])] for i in range(n-1)]
            travel_time = np.cumsum(travel_time)
            arrival_time = travel_time[:-1] + self.time
        else:
            ind_trajectory = shp_result[1][:-1]
            trajectory = [location_list[i] for i in ind_trajectory]
            travel_time = [dist[int(shp_result[1][i]), int(shp_result[1][i + 1])] for i in range(n - 1)]
            travel_time = np.cumsum(travel_time)
            arrival_time = travel_time[:-1] + self.time + dist_func((self.x,self.y), location_list[0])

        return trajectory, ind_trajectory, arrival_time

    def pickup_order(self, new_order):
        assert round(self.x) == round(new_order.ori[0])
        assert round(self.y) == round(new_order.ori[1])
        self.capacity -= 1
        assert self.capacity >= 0
        print('PICKUP: driver {0} picked order {1} at {2},{3}'.format(self.id, new_order.index,self.x,self.y))
        new_order.is_picked = True
        new_order.picked_time = self.time
        self.next_is_drop = True
        self.order_onboard.append(new_order)
        self.order_to_pick = None
        self.order_to_pick_fee = 0

    def drop_order(self):
        assert round(self.x) == round(self.order_onboard[self.next_order_ind].dest[0])
        assert round(self.x) == round(self.order_onboard[self.next_order_ind].dest[1])
        self.capacity += 1
        assert self.capacity <= self.max_capacity
        print('DROP: driver {0} droped order {1} at {2},{3}'.format(self.id, self.order_onboard[self.next_order_ind],self.x,self.y))
        order_dropped = self.order_onboard.pop(self.next_order_ind)
        self.order_drop_sequence = self.order_drop_sequence[1:]
        self.order_drop_sequence = [i if i < self.next_order_ind else i-1 for i in self.order_drop_sequence]
        self.next_order_ind = self.order_drop_sequence[0]
        return order_dropped

    def agent_reward(self, b_tn, c_tn, delta_tn, fee, lost_order):
        if lost_order and self.order_to_pick_fee > EPS:
            return b_tn - .1*c_tn - .85*delta_tn + fee - 1.1 * self.order_to_pick_fee
        else:
            return b_tn - .1 * c_tn - .85 * delta_tn + fee


    def take_action(self, action, dist_func, is_attempt=False):
        """
        :param action: [zeta, x0, y0, x1, y1, fee]
        zeta = 1 if we will pick (potential) new passengers
        x0, y0 = location of dispatching
        x1, y1 = if we will pick an order in (x0,y0), (x1,y1) will be its destination
        if we are not going to pick new users, x0 y0 and x1 y1 will be the next order drop location
        :return: reward for one driver and new space-time trajectory

        b_tn, c_tn and delta_tn: see DeepPool (7)
        """
        zeta,x0,y0,x1,y1,fee = action
        lost_order = False
        b_tn = len(self.order_onboard)
        c_tn = 0
        delta_tn = 0
        if zeta == 1:
            print('DISPATCH: driver {0} goes to {1},{2} from {3},{4}'.format(self.id, x0, y0, self.x,self.y))
            # case 1: we know which order to pick
            order_locations = [order.dest for order in self.order_onboard] # noting here we only consider orders on board
            if x0!=x1 and y0!=y1:
                location_list = [(x0, y0)] + order_locations + [(x1, y1)]
            else:
                location_list = [(x0, y0)] + order_locations
            trajectory, ind_trajectory, arrival_time = self.shp_traj(location_list, dist_func,
                                                                     start_from_current=False)
            self.order_drop_sequence = [i - 1 for i in ind_trajectory if i >= 1]
            self.trajectory = trajectory
            self.traj_time = arrival_time
            self.next_order_ind = self.order_drop_sequence[0]
            if not self.next_is_drop: # if the car is going to pick an order when dispatched to new area, unpicked order will be lost
                lost_order = True
            # calculate the difference in time
            delta_tn = 0
            for i, o in enumerate(self.order_drop_sequence): # i is the sequence while o is the index in order list
                if o <= len(self.order_onboard)-1:
                    dt = self.traj_time[i+1] - self.order_onboard[o].expected_drop_time # the first traj time on the path is a pickup point
                    self.order_onboard[o].expected_drop_time = self.traj_time[i+1]
                    delta_tn += dt
            self.next_is_drop = False
            c_tn = dist_func((self.x,self.y),(x0,y0))
        return self.agent_reward(b_tn,c_tn,delta_tn,fee,lost_order), lost_order


    def move_one_step(self):
        # move the vehicle in one time step according to its current_orders, vrp trajectory and current location
        if len(self.trajectory)==0:
            next_x = self.x
            next_y = self.y
        else:
            next_x = self.trajectory[0][0]
            next_y = self.trajectory[0][1]
        speed = self.driver_features['speed']
        dx = next_x - self.x
        dy = next_y - self.y
        move_along_y = np.random.rand() > 0.5
        if move_along_y:
            speed = min(speed,abs(dy))
            dxy = (0,  speed* (dy > 0) - speed * (dy < 0))
        else:
            speed = min(speed,abs(dx))
            dxy = (speed * (dx > 0) - speed * (dx < 0), 0)
        if dx < EPS:
            speed = min(speed, abs(dy))
            dxy = (0, speed* (dy > 0) - speed * (dy < 0))
        if dy < EPS:
            speed = min(speed, abs(dx))
            dxy = (speed* (dx > 0) - speed * (dx < 0), 0)
        self.x += dxy[0]
        self.y += dxy[1]
        self.time += 1
        print('driver {0} moved to {1},{2}'.format(self.id,self.x,self.y))
        print('new traj',self.trajectory)
        print('new traj_time',self.traj_time)
        print('new order sequence', self.order_drop_sequence)
        print('orders on board',[order.index for order in self.order_onboard])
        while True:
            if abs(self.x-self.trajectory[0][0])<EPS and abs(self.y-self.trajectory[0][1])<EPS:
                self.trajectory = self.trajectory[1:]
                self.traj_time = self.traj_time[1:]
                if self.next_is_drop:
                    self.drop_order()
                else:
                    self.pickup_order(self.order_to_pick)
            else:
                break
        return self.x, self.y

    def show_state(self):
        return [self.x, self.y, self.trajectory, self.traj_time, self.order_to_pick_fee]

class City:
    def __init__(self, size, n_drivers, n_restaurants, seed=1, even_demand_distr=True, demand_distr=None, time_horizon = 100):
        self.seed = seed
        np.random.seed(seed)
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
        self.demand_expectation = np.zeros((self.width,  self.height))  # for now, we assume the demand origin is stationary
        self.new_available_map = np.zeros((time_horizon,self.width, self.height))
        # driver map is the number of availabe drivers at each location
        self.capacity_profile = [7, 5, 3]
        self.speed_profile = [0.1, 0.2, 0.25]
        # initialize drivers, restaurants and demand(home)
        for i in range(n_drivers):
            type = np.random.choice([0, 1, 2], p=[0.1, 0.8, 0.1])
            home_x = round(np.random.rand() * self.width)
            home_y = round(np.random.rand() * self.height)
            driver_features = {'type': type, 'home_address': [home_x, home_y], 'is_last_order': False,
                               'speed': self.speed_profile[type]}
            self.drivers.append(Driver(home_x, home_y, driver_id = i, driver_features=driver_features,max_capacity=self.capacity_profile[type]))
            self.driver_map[home_x, home_y] += 1

        self.restaurants = [
            Restaurant(round(np.random.rand() * self.width / 2.), round(np.random.rand() * self.height / 2.),
                       np.random.rand() / 2. + 0.5) for _ in range(n_restaurants)]


        for restaurant in self.restaurants:
            self.demand_expectation[restaurant.x,restaurant.y] += restaurant.attractiveness
        if even_demand_distr:
            self.demand_dest_distr = {(x, y): 1 / self.width / self.height for x in range(self.width) for y in
                                 range(self.height)}
        else:
            self.demand_dest_distr = demand_distr
            # demand_dest_distr is the pdf of demand (a dictionary)
        # initialize order buffer
        self.order_buffer = []
        self.all_orders = []

        # for test only, we add one order at 0,0 and one car at 0,0 to see if the order can be picked immediately
        self.drivers[0].x = 0
        self.drivers[0].y = 0
        self.order_buffer = [Order((0, 0), (1, 1), 20, 10, 100, 0)]
        self.all_orders = [Order((0, 0), (1, 1), 20, 10, 100, 0)]
        self.total_orders = 1

    def travel_time(self, xy1, xy2, driver_features):
        x1, y1 = xy1
        x2, y2 = xy2
        speed = self.speed_profile[driver_features['type']]
        travel_time = (abs(x2 - x1) + abs(y2 - y1)) / speed
        return math.ceil(travel_time)

    def travel_cost(self, xy1, xy2, driver_features):
        x1, y1 = xy1
        x2, y2 = xy2
        travel_cost = self.travel_time(xy1,xy2, driver_features)
        if driver_features['is_last_order']:
            home_add = driver_features['home_address']
            travel_cost += 0.1 * (self.travel_time((x2, y2), (home_add[0], home_add[1]), driver_features)
                                  - self.travel_time((x1, y1), (home_add[0], home_add[1]), driver_features))
        return travel_cost

    def fee(self, expected_delivery_time):
        fee = 2 * expected_delivery_time + 3 + 10 * np.random.rand()
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
        active_restaurants = []
        new_orders = []
        for i in range(self.n_restaurants):
            if np.random.rand() < self.restaurants[i].attractiveness:
                active_restaurants.append(i)
        for i in active_restaurants:
            dict_keys = list(self.demand_dest_distr.keys())
            home_ind = np.random.choice(list(range(len(dict_keys))), p=list(self.demand_dest_distr.values()))
            home = dict_keys[home_ind]
            restaurant_x = self.restaurants[i].x
            restaurant_y = self.restaurants[i].y
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
        # action: list of actions for each driver
        # {driver ind i: (zeta_i, order ind o)}
        # for each vehicle, take an action
        rewards = []
        #city_observation = {'drivers': self.drivers, 'order_buffer': self.order_buffer}
        for i in range(self.n_drivers):
            zeta_i, order_i = actions[i]
            def dist_func(x,y):
                return self.travel_time(x,y, self.n_drivers[i].driver_features)
            if zeta_i==0:
                this_reward,lost_order = self.drivers[i].take_action([0, 0, 0, 0, 0, 0], dist_func)
            else:
                print('driver {0} take order {1},{2} -> {3},{4}'.format(i,order_i.ori[0], order_i.ori[1], order_i.dest[0], order_i.dest[1]))
                this_reward,lost_order = self.drivers[i].take_action([1, order_i.ori[0], order_i.ori[1], order_i.dest[0], order_i.dest[1], order_i.fee], dist_func)
                if lost_order and self.drivers[i].order_to_pick_fee>EPS: # if order is given up
                    self.order_buffer.append(self.drivers[i].order_to_pick) # return the order to buffer
                self.drivers[i].order_to_pick = order_i
                self.drivers[i].order_to_pick_fee = order_i.fee
            rewards.append(this_reward)
            self.drivers[i].move_one_step()
        # if an other has not been picked up, delete it from buffer
        self.order_buffer = [order for order in self.order_buffer if order.pickup_ddl > self.time]
        self.time += 1
        return rewards


if __name__ =='__main__':
    city_test = City((10, 10), n_drivers=1, n_restaurants=5)
