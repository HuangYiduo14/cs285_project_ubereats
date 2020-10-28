from envs.city import *
from policies.greedy_policy import CentralGreedyPolicy
import matplotlib.pyplot as plt


############################# plot figure a and b ##########################################
reward_record = []
total_reward = 0

city_test = City((10, 10), n_drivers=5, n_restaurants=5)
last_x = 0
last_y = 0
i = 1
plt.figure()

num_revisited_dict_black = {(x,y):0 for x in range(10) for y in range(10)}
num_revisited_dict_red = {(x,y):0 for x in range(10) for y in range(10)}
num_revisited_dict_blue = {(x,y):0 for x in range(10) for y in range(10)}
for t in range(100):
    print(t)
    city_test.order_generate()
    greedy_policy = CentralGreedyPolicy(city_test)
    actions = greedy_policy.get_action()
    rewards, observations, drop_order, pick_order = city_test.step(actions)

    total_reward += sum(rewards)
    if t == 0:
        last_x = city_test.drivers[i].x
        last_y = city_test.drivers[i].y
    plt.scatter([city_test.drivers[i].x], [city_test.drivers[i].y],color='black')
    plt.arrow(last_x, last_y, city_test.drivers[i].x-last_x,city_test.drivers[i].y-last_y )

    if last_x != city_test.drivers[i].x or last_y != city_test.drivers[i].y:
        offset = num_revisited_dict_black[(city_test.drivers[i].x, city_test.drivers[i].y)]
        plt.text(city_test.drivers[i].x+.3*offset, city_test.drivers[i].y, '{0},'.format(t), color='black')
        num_revisited_dict_black[(city_test.drivers[i].x, city_test.drivers[i].y)] += 1


    last_x = city_test.drivers[i].x
    last_y = city_test.drivers[i].y

    for k, order_ind in enumerate(pick_order[i]):
        plt.text(city_test.drivers[i].x+.2*k + .2*num_revisited_dict_blue[(city_test.drivers[i].x, city_test.drivers[i].y)], city_test.drivers[i].y +.2, '{0},'.format(order_ind), color='blue')
    num_revisited_dict_blue[(city_test.drivers[i].x, city_test.drivers[i].y)] += len(pick_order[i])
    for k, order_ind in enumerate(drop_order[i]):
        plt.text(city_test.drivers[i].x+.2*k + .2*num_revisited_dict_red[(city_test.drivers[i].x, city_test.drivers[i].y)], city_test.drivers[i].y -.2, '{0},'.format(order_ind), color='red')
    num_revisited_dict_red[(city_test.drivers[i].x, city_test.drivers[i].y)] += len(drop_order[i])
    #print('time',t,'===='*100)
    #print('buffer size', len(city_test.order_buffer))
    #for i in range(5):
    #    print(i,'cap',city_test.drivers[i].capacity)
    #    print(rewards[i], city_test.drivers[i].x, city_test.drivers[i].y, city_test.drivers[i].order_sequence, city_test.drivers[i].trajectory)
    #    print({idx:[order.is_pickup, order.ori, order.dest, order.desired_delivery_time, order.pickup_ddl] for idx, order in city_test.drivers[i].current_orders.items()})
    reward_record.append(total_reward)

plt.savefig('traj_{0}.png'.format(i))

plt.scatter([0],[0],color='black')
plt.ylim((-1,1))
plt.text(0, 0,'time when this dirver visit this loaction [t1,t2,...]',color='black')
plt.text(0, 0.2,'order picked at this location [order id 1, order id 2,...]',color='blue')
plt.text(0,-0.2,'order dropped at this location [order id 1, order id 2,...]',color='red')



plt.figure()
plt.plot(reward_record)
plt.xlabel('time')
plt.ylabel('total reward')
plt.grid()
plt.savefig('reward over time')



############################# plot figure c ##########################################
avg_reward_record = []
for n_drivers in [1,3,5,10,20,30,50,100]:
    print('drivers',n_drivers)
    total_reward = 0
    city_test = City((10, 10), n_drivers=n_drivers, n_restaurants=5)
    for t in range(200):
        print(t, n_drivers)
        city_test.order_generate()
        greedy_policy = CentralGreedyPolicy(city_test)
        actions = greedy_policy.get_action()
        rewards, observations, drop_order, pick_order = city_test.step(actions)
        # city_log.append(city_observation_t)
        total_reward += sum(rewards)
    avg_reward_record.append(total_reward/n_drivers)
plt.figure()
plt.plot([1,3,5,10,20,30,50,100], avg_reward_record)
plt.scatter([0],[0])
plt.xlabel('number of drivers')
plt.ylabel('average reward for each driver')
plt.grid()
plt.savefig('avg reward vs n drivers.png')
