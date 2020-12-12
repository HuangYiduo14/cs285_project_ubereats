
#from pulp import *
#import numpy as np
#import pandas as pd
#from more_itertools import iterate, take
BIG_NUM = 1e5


from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np


def solve_tsp_dynamic_programming(
    distance_matrix: np.ndarray,
    maxsize: Optional[int] = None
) -> Tuple[List, float]:
    """
    Copy from https://github.com/fillipe-gsm/python-tsp.git
    Solve TSP to optimality with dynamic programming.
    Parameters
    ----------
    distance_matrix
        Distance matrix of shape (n x n) with the (i, j) entry indicating the
        distance from node i to j. It does not need to be symmetric
    maxsize
        Parameter passed to ``lru_cache`` decorator. Used to define the maximum
        size for the recursion tree. Defaults to `None`, which essentially
        means "take as much space as needed".
    Returns
    -------
    permutation
        A permutation of nodes from 0 to n that produces the least total
        distance
    distance
        The total distance the optimal permutation produces
    Notes
    -----
    Algorithm: cost of the optimal path
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Consider a TSP instance with 3 nodes: {0, 1, 2}. Let dist(0, {1, 2}) be the
    distance from 0, visiting all nodes in {1, 2} and going back to 0. This can
    be computed recursively as:
        dist(0, {1, 2}) = min(
            c_{0, 1} + dist(1, {2}),
            c_{0, 2} + dist(2, {1}),
        )
    wherein c_{0, 1} is the cost from going from 0 to 1 in the distance matrix.
    The inner dist(1, {2}) is computed as:
        dist(1, {2}) = min(
            c_{1, 2} + dist(2, {}),
        )
    and similarly for dist(2, {1}). The stopping point in the recursion is:
        dist(2, {}) = c_{2, 0}.
    This process can be generalized as:
        dist(ni, N) =   min   ( c_{ni, nj} + dist(nj, N - {nj}) )
                      nj in N
    and
        dist(ni, {}) = c_{ni, 0}
    With starting point as dist(0, {1, 2, ..., tsp_size}). The notation
    N - {nj} is the difference operator, meaning set N without node nj.
    Algorithm: compute the optimal path
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The previous process returns the distance of the optimal path. To find the
    actual path, we need to store in a memory the following key/values:
        memo[(ni, N)] = nj_min
    with nj_min the node in N that provided the smallest value of dist(ni, N).
    Then, the process goes backwards starting from
    memo[(0, {1, 2, ..., tsp_size})].
    In the previous example, suppose memo[(0, {1, 2})] = 1.
    Then, look for memo[(1, {2})] = 2.
    Then, since the next step would be memo[2, {}], stop there. The optimal
    path would be 0 -> 1 -> 2 -> 0.
    Reference
    ---------
    https://en.wikipedia.org/wiki/Held%E2%80%93Karp_algorithm#cite_note-5
    """
    # Get initial set {1, 2, ..., tsp_size} as a frozenset because since
    # @lru_cache requires a hashable type
    distance_matrix[:, 0] = 0
    N = frozenset(range(1, distance_matrix.shape[0]))
    memo: Dict[Tuple, int] = {}

    # Step 1: get minimum distance
    @lru_cache(maxsize=maxsize)
    def dist(ni: int, N: frozenset) -> float:
        if not N:
            return distance_matrix[ni, 0]

        # Store the costs in the form (nj, dist(nj, N))
        costs = [
            (nj, distance_matrix[ni, nj] + dist(nj, N.difference({nj})))
            for nj in N
        ]
        nmin, min_cost = min(costs, key=lambda x: x[1])
        memo[(ni, N)] = nmin
        return min_cost

    best_distance = dist(0, N)

    # Step 2: get path with the minimum distance
    ni = 0  # start at the origin
    solution = [0]
    while N:
        ni = memo[(ni, N)]
        solution.append(ni)
        N = N.difference({ni})

    assert solution[0] == 0
    return best_distance, solution

'''
def shortest_ham_path(n, dist):
    """
    Solve shortest Hamiltonian path as a variant of TSP
    the path must start from node 0 and end in n-1

    For example, input [0,1,2,3,4,5], output [0,2,3,4,5,1] will be the shortest path start from 0 while all vertices are visited exactly once.

    Adapted from tsp package
    """
    #n = len(nodes)
    dist[:, 0] = 0
    a = pd.DataFrame(
        [(i, j, dist[i, j]) for i in range(n) for j in range(n) if i != j],
        columns=["NodeI", "NodeJ", "Dist"],
    )
    m = LpProblem()
    m.setSolver(PULP_CBC_CMD(msg=False))
    a["VarIJ"] = [LpVariable("x%d" % i, cat=LpBinary) for i in a.index]
    a["VarJI"] = a.sort_values(["NodeJ", "NodeI"]).VarIJ.values
    u = [0] + [LpVariable("y%d" % i, lowBound=0) for i in range(n - 1)]
    m += lpDot(a.Dist, a.VarIJ)
    for i, v in a.groupby("NodeI"):
        m += lpSum(v.VarIJ) == 1  # inflow
        m += lpSum(v.VarJI) == 1  # outflow
    for _, (i, j, _, vij, vji) in a.query("NodeI!=0 & NodeJ!=0").iterrows():
        m += u[i] + 1 - (n - 1) * (1 - vij) + (n - 3) * vji <= u[j] # variant of MTZ constraints
    for _, (_, j, _, v0j, vj0) in a.query("NodeI==0").iterrows():
        m += 1 + (1 - v0j) + (n - 3) * vj0 <= u[j]
    for _, (i, _, _, vi0, v0i) in a.query("NodeJ==0").iterrows():
        m += u[i] <= (n - 1) - (1 - vi0) - (n - 3) * v0i
    sol = m.solve()

    a["ValIJ"] = a.VarIJ.apply(value)
    dc = dict(a[a.ValIJ > 0.5][["NodeI", "NodeJ"]].values)
    traj = list(take(n, iterate(lambda k: dc[k], 0)))
    assert traj[0] == 0
    return value(m.objective), traj
'''
