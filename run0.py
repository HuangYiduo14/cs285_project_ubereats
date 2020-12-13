from cs285.envs.city import *
from cs285.policies.greedy_policy import CentralGreedyPolicy

city_test = City((10, 10), n_drivers=1, n_restaurants=6, seed=10)

re_history = []
fee_history = []
fee_v_history = []

for t in range(1000):
    greedy_policy = CentralGreedyPolicy(city_test)
    actions = greedy_policy.get_action()
    obs, res, _, _ = city_test.step(actions)
    re_history.append(sum(res))
    fee_history.append(city_test.order_picked_val - city_test.order_droped_val)
    fee_v_history.append(city_test.order_pend_val - city_test.order_lost_val - city_test.order_picked_val)

import matplotlib.pyplot as plt
plt.plot(fee_history,label='onboard fee')
plt.plot(fee_v_history,label = 'pending fee')

print('average reward',np.mean(re_history))