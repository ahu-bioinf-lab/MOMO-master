import platgo as pg
import numpy as np

"""
Hypervolume
Reference 
E. Zitzler and L. Thiele, Multiobjective evolutionary algorithms: A
comparative case study and the strength Pareto approach, IEEE
Transactions on Evolutionary Computation, 1999, 3(4): 257-271.
"""


def cal_hv(pop: pg.Population, refer_array: np.ndarray) -> float:
    """
    :param:
    :pop: Population
    :pf: the pareto front information
    """
    objv = pop.objv
    N, M = objv.shape
    # Normalize the population according to the reference point set
    fmin = np.min(np.vstack((np.min(objv, axis=0), np.zeros([1, M]))), axis=0)
    famx = np.max(refer_array, axis=0)
    objv = (objv - fmin) / np.tile((famx - fmin) * 1.1, (N, 1))
    objv = objv[np.max(objv, axis=1) <= 1]
    #  The reference point is set to (1,1,...)
    ref_point = np.ones(M)
    if objv.size == 0:
        score = 0
    elif M < 4:
        # Calculate the exact HV value
        pl = np.unique(objv, axis=0)
        s = [[1, pl]]
        for k in range(M - 1):
            s_ = []
            for i in range(len(s)):
                stemp = Slice(s[i][1], k, ref_point)
                for j in range(len(stemp)):
                    temp = [[stemp[j][0] * s[i][0], np.array(stemp[j][1])]]
                    s_ = Add(temp, s_)
            s = s_
        score = 0
        for i in range(len(s)):
            p = Head(s[i][1])
            score = score + s[i][0] * np.abs(p[-1] - ref_point[-1])
    #     # (score)
    else:
        # Estimate the HV value by Monte Carlo estimation
        sample_num = 1000000
        max_value = ref_point
        min_value = np.min(objv, axis=0)
        samples = np.random.uniform(np.tile(min_value, (sample_num, 1)), np.tile(max_value, (sample_num, 1)))
        for i in range(len(objv)):
            domi = np.ones(len(samples), dtype=bool)
            m = 0
            while m <= M-1 and np.any(domi):
                b = objv[i][m] <= samples[:, m]
                domi = domi & b
                m += 1
            samples = samples[~domi]
        score = np.prod(max_value - min_value) * (1 - len(samples)/sample_num)

    return score


def Slice(pl: np.ndarray, k: int, ref_point: np.ndarray) -> list:

    p = Head(pl)
    pl = Tail(pl)
    ql = np.array([])
    s = []
    while len(pl):
        ql = Insert(p, k + 1, ql)
        p_ = Head(pl)
        if ql.ndim == 1:
            list_ = [[np.abs(p[k] - p_[k]), np.array([ql])]]
        else:
            list_ = [[np.abs(p[k] - p_[k]), ql]]
        s = Add(list_, s)
        p = p_
        pl = Tail(pl)
    ql = Insert(p, k + 1, ql)
    if ql.ndim == 1:
        list_ = [[np.abs(p[k] - ref_point[k]), [ql]]]
    else:
        list_ = [[np.abs(p[k] - ref_point[k]), ql]]
    s = Add(list_, s)
    return s


def Insert(p: np.ndarray, k: int, pl: np.ndarray) -> np.ndarray:

    flag1 = 0
    flag2 = 0
    ql = np.array([])
    hp = Head(pl)
    while len(pl) and hp[k] < p[k]:
        if len(ql) == 0:
            ql = hp
        else:
            ql = np.vstack((ql, hp))
        pl = Tail(pl)
        hp = Head(pl)
    if len(ql) == 0:
        ql = p
    else:
        ql = np.vstack((ql, p))
    m = max(p.shape)
    while len(pl):
        q = Head(pl)
        for i in range(k, m):
            if p[i] < q[i]:
                flag1 = 1
            elif p[i] > q[i]:
                flag2 = 1
        if not (flag1 == 1 and flag2 == 0):
            if len(ql) == 0:
                ql = Head(pl)
            else:
                ql = np.vstack((ql, Head(pl)))
        pl = Tail(pl)
    return ql


def Head(pl: np.ndarray) -> np.ndarray:
    # 取第一行所有元素
    if pl.ndim == 1:
        p = pl
    else:
        p = pl[0]
    return p


def Tail(pl: np.ndarray) -> np.ndarray:
    # 取除去第一行的所有元素
    if pl.ndim == 1 or min(pl.shape) == 1:
        ql = np.array([])
    else:
        ql = pl[1:]
    return ql


def Add(list_: list, s: list) -> list:

    n = len(s)
    m = 0
    for k in range(n):
        if np.all(list_[0][1]) == np.all(s[k][1]) and len(list_[0][1]) == len(s[k][1]):
            s[k][0] = s[k][0] + list_[0][0]
            m = 1
            break
    if m == 0:
        if n == 0:
            s = list_
        else:
            s.append(list_[0])
    s_ = s
    return s_


if __name__ == "__main__":
    decs = np.random.random((100, 3))
    pop = pg.Population(decs=decs)
    pop.objv = np.array([[1.3,1.2,1.1],[1.1,1.2,1.3],[1.2,1.2,1.2]])
    pareto = np.array([[1, 1, 1]])
    score = cal_hv(pop, pareto)
    print(score)


