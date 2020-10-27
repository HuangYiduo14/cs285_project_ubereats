from envs.city import *
from policies.greedy_policy import CentralGreedyPolicy
import matplotlib.pyplot as plt

city_test = City((10,10), n_drivers=5, n_restaurants=5)

#city_log = []
#reward_record = []
total_reward = 0
last_x = 0
last_y = 0
i = 3
for t in range(200):
    print(t)
    city_test.order_generate()
    greedy_policy = CentralGreedyPolicy(city_test)
    actions = greedy_policy.get_action()
    rewards, observations, drop_order, pick_order = city_test.step(actions)
    #city_log.append(city_observation_t)
    """
    if t == 0:
        last_x = city_test.drivers[i].x
        last_y = city_test.drivers[i].y
    plt.scatter([city_test.drivers[i].x], [city_test.drivers[i].y],color='black')
    plt.arrow(last_x, last_y, city_test.drivers[i].x-last_x,city_test.drivers[i].y-last_y )

    if last_x != city_test.drivers[i].x or last_y != city_test.drivers[i].y:
        plt.text(city_test.drivers[i].x, city_test.drivers[i].y, '{0},'.format(t), color='black')
    last_x = city_test.drivers[i].x
    last_y = city_test.drivers[i].y

    for k, order_ind in enumerate(pick_order[i]):
        plt.text(city_test.drivers[i].x+.1*k, city_test.drivers[i].y +.1, '{0},'.format(order_ind), color='blue')
    for k, order_ind in enumerate(drop_order[i]):
        plt.text(city_test.drivers[i].x+.1*k, city_test.drivers[i].y -.1, '{0}'.format(order_ind), color='red')
    """

    #print('time',t,'===='*100)
    #print('buffer size', len(city_test.order_buffer))
    #for i in range(5):
    #    print(i,'cap',city_test.drivers[i].capacity)
    #    print(rewards[i], city_test.drivers[i].x, city_test.drivers[i].y, city_test.drivers[i].order_sequence, city_test.drivers[i].trajectory)
    #    print({idx:[order.is_pickup, order.ori, order.dest, order.desired_delivery_time, order.pickup_ddl] for idx, order in city_test.drivers[i].current_orders.items()})
    total_reward += sum(rewards)
    #reward_record.append(total_reward)
#plt.savefig('traj_{0}.png'.format(i))
