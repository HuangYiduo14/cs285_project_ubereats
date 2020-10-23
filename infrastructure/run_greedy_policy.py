from envs.city import *
from policies.greedy_policy import CentralGreedyPolicy

city_test = City((10,10), n_drivers=5, n_restaurants=5)
greedy_policy = CentralGreedyPolicy()

for t in range(100):
    city_test.order_generate()
    actions = greedy_policy.get_action(city_test.drivers, city_test.order_buffer)
    city_test.step(actions)
