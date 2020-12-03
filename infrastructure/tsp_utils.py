
from pulp import *
import numpy as np
import pandas as pd
from more_itertools import iterate, take

def shortest_ham_path(n, dist=None):
    """
    Solve shortest Hamiltonian path as a variant of TSP
    adapted from tsp package
    """
    #n = len(nodes)
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
        if i==0:
            m += lpSum(v.VarIJ) == 1  # inflow
            m += lpSum(v.VarJI) == 0 # outflow
        elif i==n-1:
            m += lpSum(v.VarIJ) == 0 # inflow
            m += lpSum(v.VarJI) == 1  # outflow
        else:
            m += lpSum(v.VarIJ) == 1 # inflow
            m += lpSum(v.VarJI) == 1 # outflow
    for _, (i, j, _, vij, vji) in a.query("NodeI!=0 & NodeJ!=0").iterrows():
        m += u[i] + 1 - (n - 1) * (1 - vij) + (n - 3) * vji <= u[j]
    for _, (_, j, _, v0j, vj0) in a.query("NodeI==0").iterrows():
        m += 1 + (1 - v0j) + (n - 3) * vj0 <= u[j]
    for _, (i, _, _, vi0, v0i) in a.query("NodeJ==0").iterrows():
        m += u[i] <= (n - 1) - (1 - vi0) - (n - 3) * v0i
    m.solve()
    a["ValIJ"] = a.VarIJ.apply(value)
    dc = dict(a[a.ValIJ > 0.5][["NodeI", "NodeJ"]].values)
    return value(m.objective), list(take(n, iterate(lambda k: dc[k], 0)))