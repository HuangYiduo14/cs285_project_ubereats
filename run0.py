from envs.city import *
from policies.greedy_policy import CentralGreedyPolicy

city_test = City((10,10), n_drivers=5, n_restaurants=5)

city_log = []
for t in range(5):
    city_test.order_generate()
    greedy_policy = CentralGreedyPolicy(city_test)
    actions = greedy_policy.get_action()
    rewards, observations, city_observation_t = city_test.step(actions)
    city_log.append(city_observation_t)
    print('time',t,'===='*100)
    #print('buffer size', len(city_test.order_buffer))
    for i in range(5):
        print(i,'cap',city_test.drivers[i].capacity)
        print(rewards[i], city_test.drivers[i].x, city_test.drivers[i].y, city_test.drivers[i].order_sequence, city_test.drivers[i].trajectory)
        print({idx:[order.is_pickup, order.ori, order.dest, order.desired_delivery_time, order.pickup_ddl] for idx, order in city_test.drivers[i].current_orders.items()})