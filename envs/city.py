import numpy as np
import pdb


class Restaurant:
    def __init__(self, x, y, attractiveness):
        self.x = x
        self.y = y
        self.attractiveness = attractiveness


class Order:
    def __init__(self, o, d, desired_delivery_time, pickup_ddl, fee, index, is_repostion=False):
        self.ori = o
        self.dest = d
        self.desired_delivery_time = desired_delivery_time
        self.desired_travel_time = desired_delivery_time - pickup_ddl
        self.fee = fee
        self.pickup_ddl = pickup_ddl
        self.is_pickup = False
        self.is_reposition = is_repostion
        self.index = index


class Driver:
    def __init__(self, x, y, driver_features, order_radius=1, driver_radius=2, max_capacity=4, max_orders=4):
        self.x = x
        self.y = y
        self.driver_features = driver_features

        self.max_capacity = max_capacity
        self.capacity = max_capacity
        self.current_orders = dict()  # we keep a list of unfinished orders

        self.is_idle = True
        self.trajectory = []  # we keep a record of VRP result trajectory
        self.max_orders = max_orders
        # at time 0, we don't know the trajectory, so we have
        for i in range(max_orders + 1):
            self.trajectory.append(x)
            self.trajectory.append(y)
            self.trajectory.append(i)

        self.realized_reward = 0
        self.order_sequence = [-1]

        self.order_search_radius = order_radius
        self.driver_search_radius = driver_radius


    def move_one_step(self):
        # move the vehicle in one time step according to its current_orders, vrp trajectory and current location
        next_x = self.trajectory[3]
        next_y = self.trajectory[4]
        move_success = np.random.rand() < self.driver_features['speed']
        if move_success:
            dx = next_x - self.x
            dy = next_y - self.y
            #assert abs(dx)+abs(dy) > 0
            move_along_y = np.random.rand() > 0.5
            if move_along_y:
                dxy = (0, 1 * (dy > 0) - 1 * (dy < 0))
            else:
                dxy = (1 * (dx > 0) - 1 * (dx < 0), 0)
            if dx == 0:
                dxy = (0, 1 * (dy > 0) - 1 * (dy < 0))
            if dy == 0:
                dxy = (1 * (dx > 0) - 1 * (dx < 0), 0)
            self.x += dxy[0]
            self.y += dxy[1]
            self.trajectory[0] = self.x
            self.trajectory[1] = self.y
            self.trajectory[2] += 1
        else:
            self.trajectory[2] += 1

        # update new expected time
        # if overdue orders has not been picked up, delete them from buffer
        return -1


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
        self.total_orders = 0
        self.Inf = 9999999

        self.speed_profile = [0.1, 0.2, 0.25]
        # initialize drivers, restaurants and demand(home)
        for _ in range(n_drivers):
            type = np.random.choice([0, 1, 2], p=[0.1, 0.8, 0.1])
            home_x = round(np.random.rand() * self.width)
            home_y = round(np.random.rand() * self.height)
            driver_features = {'type': type, 'home_address': [home_x, home_y], 'is_last_order': False,
                               'speed': self.speed_profile[type]}
            self.drivers.append(Driver(home_x, home_y, driver_features))

        self.restaurants = [
            Restaurant(round(np.random.rand() * self.width / 2.), round(np.random.rand() * self.height / 2.),
                       np.random.rand() / 2. + 0.5)
            for _ in
            range(n_restaurants)]  # we assume restaurant are in the center of the city
        if even_demand_distr:
            self.demand_distr = {(x, y): 1 / self.width / self.height for x in range(self.width) for y in
                                 range(self.height)}
            # demand_distr is the pdf of demand (a dictionary)

        # initialize order buffer
        self.order_buffer = []
        self.all_orders = []

        # for test only, we add one order at 0,0 and one car at 0,0 to see if the order can be picked immediately
        self.drivers[0].x = 0
        self.drivers[0].y = 0
        self.order_buffer = [Order((0, 0), (1, 1), 20, 10, 100, 0)]
        self.all_orders = [Order((0, 0), (1, 1), 20, 10, 100, 0)]
        self.total_orders = 1

    def travel_time(self, x1, y1, x2, y2, driver_features):
        speed = self.speed_profile[driver_features['type']]
        travel_time = (abs(x2 - x1) + abs(y2 - y1)) / speed
        return round(travel_time)

    def travel_cost(self, x1, y1, x2, y2, driver_features):
        travel_cost = self.travel_time(x1, y1, x2, y2, driver_features)
        if driver_features['is_last_order']:
            home_add = driver_features['home_address']
            travel_cost += 0.1 * (self.travel_time(x2, y2, home_add[0], home_add[1], driver_features)
                                  - self.travel_time(x1, y1, home_add[0], home_add[1], driver_features))
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
        return 2. * self.travel_time(restaurant_x, restaurant_y, home_x, home_y, {'type': 1})

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
            dict_keys = list(self.demand_distr.keys())
            home_ind = np.random.choice(list(range(len(dict_keys))), p=list(self.demand_distr.values()))
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

    def observe(self, driver: Driver):
        obs = []
        obs += [driver.x, driver.y]
        obs += driver.trajectory
        obs += driver.driver_features['home_address']
        obs += [driver.driver_features['type'], driver.driver_features['is_last_order']]
        return obs

    def step(self, actions):
        # action: list of actions for each driver
        # for each vehicle, take an action
        rewards = []
        observations = []
        drop_order = []
        pick_order = []
        #city_observation = {'drivers': self.drivers, 'order_buffer': self.order_buffer}
        for i in range(self.n_drivers):
            this_reward, order_deleted_index, order_picked_index = self.driver_take_action(self.drivers[i], actions[i])
            rewards.append(this_reward)
            observations.append(self.observe(self.drivers[i]))
            drop_order.append(order_deleted_index)
            pick_order.append(order_picked_index)


        # if an other has not been picked up, delete it from buffer
        self.order_buffer = [order for order in self.order_buffer if order.pickup_ddl > self.time]
        self.time += 1

        return rewards, observations, drop_order, pick_order

    def vrp_orders(self, driver, orders):
        """
        Do vrp for driver from the current location to find a path to satisfy all orders.
        If some order can be picked or deleted at this time, i.e. the driver is at its pickup/drop point, we calculate
        the corresponding reward and called this actual_reward.
        For other orders that will be picked along this path, we calculate the expected reward.
        Only the actual_reward is deterministic because the travel time is stochastic for each movement.

        :param driver: Driver, we call (driver.x, driver.y) the current location
        :param orders: list of Order
        :return:
        new_trajectory: trajectory plan [current x, current y, current time, next x, next y, expected time,....]
        Notice that if current x and y is a pickup/drop point, we will not include this point in the trajectory,
        instead, we add the order index into list order_picked_index or order_deleted_index

        order_sequence: orders to pickup/drop on the trajectory, -1 means no pickup/drop. [-1, order 1, order 2, order 1,...]
        each element in this list corresponds to 3 elements in trajectory [x,y,t]

        actual_reward: if current x and y is a drop point or some unpicked orders are lost, reward will be incurred

        expected_reward: total reward gained from the planned trajectory

        order_picked_index, order_deleted_index: orders that are picked up/ lost at the current loctaion
        """
        num_orders = len(orders)
        is_picked = [order.is_pickup for order in orders]
        cap_available = driver.capacity
        assert num_orders <= driver.max_orders
        x = driver.x
        y = driver.y
        t = self.time
        # update drivers orders and penalize lost orders at this time
        order_deleted_index = []
        order_picked_index = []
        expected_reward = 0
        lost_order_penalty = 0
        for order in orders:
            if order.pickup_ddl < t and (not order.is_pickup):
                lost_order_penalty += self.lost_order_penalty(order.fee)
                order_deleted_index.append(order.index)
        targets = dict() # targets is the location of all possible next points
        for key, order in enumerate(orders):
            if order.is_pickup:
                targets[key] = order.dest
            else:
                targets[key] = order.ori
        targets = {key: loc for key, loc in targets.items() if (orders[key].pickup_ddl >= t or orders[key].is_pickup)}
        new_trajectory = [x, y, t]
        order_sequence = [-1]
        actual_reward = -lost_order_penalty

        # greedily find next location until the targets dictionary is empty
        while len(targets) > 0:
            min_trip_cost = self.Inf
            next_order = -1
            for order_key, target in targets.items():
                this_trip_cost = self.travel_cost(x, y, target[0], target[1], driver.driver_features)
                if this_trip_cost < min_trip_cost:
                    if (not is_picked[order_key]) and (cap_available == 0):
                        continue
                    min_trip_cost = this_trip_cost
                    next_order = order_key
            t += self.travel_time(x, y, targets[next_order][0], targets[next_order][1], driver.driver_features)
            x = targets[next_order][0]
            y = targets[next_order][1]
            expected_reward -= self.travel_cost(x, y, targets[next_order][0], targets[next_order][1],
                                                driver.driver_features)
            if t != self.time:
                # if the next point is the same as the current point, we will not include it in the trajectory,
                # instead, we will pickup/drop the order and calculate the actual reward
                new_trajectory += [x, y, t]
                order_sequence.append(orders[next_order].index)
            if is_picked[next_order]:
                # if is already picked up
                # calculate fee and late penalty
                late_time = (t - orders[next_order].desired_delivery_time) / orders[next_order].desired_travel_time
                loc = targets.pop(next_order) # delete this order from targets if dropped
                cap_available += 1
                expected_reward = expected_reward + orders[next_order].fee - self.late_penalty(late_time,
                                                                                               orders[next_order].fee)
                if t == self.time:
                    # if this order has the same location as driver.x , driver.y and can be dropped,
                    # then drop it and calculate actual reward and update order_deleted_index
                    order_deleted_index.append(orders[next_order].index)
                    actual_reward = actual_reward + orders[next_order].fee - self.late_penalty(late_time,
                                                                                               orders[next_order].fee)

            else:
                is_picked[next_order] = True
                targets[next_order] = orders[next_order].dest
                cap_available -= 1
                if t == self.time:
                    # if this order has the same location as driver.x , driver.y and can be dropped,
                    # then drop it and calculate actual reward and update order_picked_index
                    order_picked_index.append(orders[next_order].index)

            # overdue orders that are not picked up are expected to be lost
            for key, _ in targets.items():
                if orders[key].pickup_ddl < t and (not orders[key].is_pickup):
                    expected_reward -= self.lost_order_penalty(orders[key].fee)
            targets = {key: loc for key, loc in targets.items() if
                       ((orders[key].pickup_ddl >= t) or orders[key].is_pickup)}
        # the format of trajectory is [x0,y0,t0,x1,y1,t1,....,xm,ym,tm], if we only have k-1 orders, k<m,
        # then we fill the trajectory with the last order completed like
        # [x0,y0,t0,...,xk,yk,tk,xk,yk,tk,....,xk,yk,tk] so that the length of the trajectory is fixed
        for _ in range((3 * (driver.max_orders * 2 + 1) - len(new_trajectory)) // 3):
            new_trajectory += [x, y, t]
            order_sequence.append(-1)
        return new_trajectory, order_sequence, actual_reward, expected_reward, order_picked_index, order_deleted_index

    def driver_take_action(self, driver: Driver, action: Order):
        """
        driver take an action, calculate reward and update current_orders list and trajectory,
        then move the driver one step according to its planned trajectory
        :param driver: Driver
        :param action: Order to take for this driver
        :return:
        reward: actual reward from this step
        """
        order_droped_index = []
        order_picked_index = []
        if (action is None) or (not action.is_reposition): # if this is not a reposition order
            if action is not None: # if this order is not None (None means stay on the current trajectory)
                driver.current_orders[action.index] = action
                # update the order buffer since this order is taken
                # TODO: use a dict {order.index: order} to represent the order_buffer so that deleting can be done in O(1)
                self.order_buffer = [order for order in self.order_buffer if action.index != order.index]
            new_order_list = list(driver.current_orders.values())
            # calculate the vrp trajectory for one driver
            new_trajectory, order_sequence, actual_reward, _, order_picked_index, order_deleted_index = self.vrp_orders(
                driver, new_order_list)
            # pickup/drop/delete orders
            for order_ind in order_picked_index:
                driver.current_orders[order_ind].is_pickup = True
                print('picked order', order_ind)
                driver.capacity -= 1
                assert driver.capacity >= 0
            order_droped_index = []
            for order_ind in order_deleted_index:
                if driver.current_orders[order_ind].is_pickup:
                    print('drop order', order_ind)
                    driver.current_orders.pop(order_ind)
                    driver.capacity += 1
                    assert driver.capacity <= driver.max_capacity
                    order_droped_index.append(order_ind)
                else:
                    print('pickup overdue', order_ind)
                    driver.current_orders.pop(order_ind)
            driver.trajectory = new_trajectory
            driver.order_sequence = order_sequence
        else: # if this is a repostion order (only happens when there is no current orders)
            assert len(driver.current_orders) == 0
            driver.current_orders = dict()
            # update trajectory according to the reposition order, new trajecory is
            # [current x, current y, current time, reposition x, reposition y, estimated arrival time, reposition x, reposition y, estimated arrival time,...]
            travel_time = self.travel_time(driver.x, driver.y, action.dest[0], action.dest[1], driver.driver_features)
            driver.trajectory = [driver.x, driver.y, self.time, action.dest[0], action.dest[1], self.time + travel_time]
            for _ in range((3 * (driver.max_orders + 1) - 6) // 3):
                driver.trajectory += [action.dest[0], action.dest[1], self.time + travel_time]
            actual_reward = 0
        # move one step
        reward = actual_reward + driver.move_one_step()
        return reward, order_droped_index, order_picked_index
