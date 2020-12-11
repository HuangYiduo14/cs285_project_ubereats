from cs285.envs.city import *
from cs285.policies.greedy_policy import CentralGreedyPolicy

city_test = City((10, 10), n_drivers=1, n_restaurants=5)
for t in range(100):
    city_test.order_generate()
    greedy_policy = CentralGreedyPolicy(city_test)
    actions = greedy_policy.get_action()
    rewards= city_test.step(actions)
