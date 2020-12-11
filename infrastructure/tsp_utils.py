
from pulp import *
import numpy as np
import pandas as pd
from more_itertools import iterate, take
BIG_NUM = 1e5
def shortest_ham_path(n, dist=None):
    """
    Solve shortest Hamiltonian path as a variant of TSP
    the path must start from node 0 and end in n-1

    For example, input [0,1,2,3,4,5], output [0,2,3,4,1,5] will be the shortest path start from 0 and end at 5
    while all vertices are visited exactly once.

    Adapted from tsp package
    """
    #n = len(nodes)
    for i in range(1,n-1):
        dist[i,0] = BIG_NUM
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
    return value(m.objective), list(take(n, iterate(lambda k: dc[k], 0)))
