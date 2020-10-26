from envs.city import *
from policies.greedy_policy import CentralGreedyPolicy

city_test = City((10,10), n_drivers=5, n_restaurants=5)


city_log = []
for t in range(100):
    city_test.order_generate()
    greedy_policy = CentralGreedyPolicy(city_test)
    actions = greedy_policy.get_action()
    rewards, observations, city_observation_t = city_test.step(actions)
    city_log.append(city_observation_t)