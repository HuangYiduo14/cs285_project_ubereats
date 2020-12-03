from envs.city import *
from policies.greedy_policy import CentralGreedyPolicy




city_log = []
for t in range(100):
    city_test.order_generate()
    greedy_policy = CentralGreedyPolicy(city_test)
    actions = greedy_policy.get_action()
    rewards, observations, city_observation_t = city_test.step(actions)
    city_log.append(city_observation_t)