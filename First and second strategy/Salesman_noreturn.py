# Python3 program to implement traveling salesman
# problem using naive approach.
from sys import maxsize
from itertools import permutations


# implementation of traveling Salesman Problem
def TSP_enumeration(graph):
    s = 0
    V = len(graph)
    # store all vertex apart from source vertex
    vertex = []
    for i in range(V):
        if i != s:
            vertex.append(i)

    # store minimum weight Hamiltonian Cycle
    min_path = maxsize
    next_permutation = permutations(vertex)
    path = []
    for i in next_permutation:
        P = [0]
        # store current Path weight(cost)
        current_pathweight = 0

        # compute current path weight
        k = s
        for j in i:
            current_pathweight += graph[k][j]
            k = j
            P.append(j)
        current_pathweight += graph[k][s]

        # update minimum
        if min_path > current_pathweight:
            min_path = current_pathweight
            path = P
    # path.remove(V-1)
    aoao = []
    index = path.index(0)
    if path[index + 1] == V - 1:
        for i in list(range(0, V)):
            aoao.append(path[(index - i) % V])
    else:
        for i in list(range(0, len(path))):
            aoao.append(path[(index + i) % V])
    aoao.remove(V - 1)
    return [round(min_path - graph[1][-1], 1), aoao]


import numpy as np
import matplotlib.pyplot as plt


class ACO_TSP(object):

    def __init__(self, raw_Adj, num_ant=31, rho=0.1, Q=1):
        self.num = len(raw_Adj)  # 位置个数
        self.num_ant = num_ant  # 蚁群个数
        self.raw_Adj = raw_Adj
        self.matrix_distance = self.matrix_dis()
        self.eta = 1 / self.matrix_distance  # 启发函数矩阵

        self.Tau_info = np.ones((self.num, self.num))  # 初始信息素浓度矩阵
        self.path = np.array([0] * self.num * self.num_ant).reshape(self.num_ant, self.num)
        self.rho = rho  # 信息素的挥发程度
        self.Q = Q  # 常系数

    def matrix_dis(self):
        res = np.array(self.raw_Adj)

        return res

    def comp_fit(self, one_path):
        res = 0
        for i in range(self.num - 1):
            res += self.matrix_distance[one_path[i], one_path[i + 1]]
        res += self.matrix_distance[one_path[-1], one_path[0]]
        return res

    def initial_path(self):
        self.path = np.array([0] * self.num * self.num_ant).reshape(self.num_ant, self.num)

    def rand_chrom(self):
        for i in range(self.num_ant):
            self.path[i, 0] = np.random.randint(self.num)

    def update_info(self, fit):
        # fit 表示蚂蚁一条路径的长度
        delta = sum([self.Q / fit[i] for i in range(self.num_ant)])
        Delta_Tau = np.zeros((self.num, self.num))
        for i in range(self.num_ant):
            for j in range(self.num - 1):
                Delta_Tau[self.path[i, j], self.path[i, j + 1]] += self.Q / fit[i]
            Delta_Tau[self.path[i, 0], self.path[i, -1]] += self.Q / fit[i];
        self.Tau_info = (1 - self.rho) * self.Tau_info + Delta_Tau

    def out_path(self, one_path):
        res = str(one_path[0] + 1) + '-->'
        for i in range(1, self.num):
            res += str(one_path[i] + 1) + '-->'
        res += str(one_path[0] + 1) + '\n'
        print(res)


def TSP_ACO(raw_Adj):
    num_ant = 20
    alpha = 1  # 信息素重要程度
    beta = 8  # 启发函数
    rho = 0.1  # 信息素挥发
    Q = 1  # 常系数

    iter_0 = 0
    Max_iter = 300
    n = len(raw_Adj)  # 城市个数

    # 蚁群算法类
    Path_short = ACO_TSP(raw_Adj, num_ant=num_ant, rho=rho, Q=Q)

    # 城市集合，因为python下标从0开始，直接去range(n)
    city_index = np.array(range(n))

    best_path = []
    best_fit = []  # 存储每一步的最优路径

    while iter_0 <= Max_iter:
        Path_short.initial_path()  # 清空蚂蚁路径
        Path_short.rand_chrom()  # 随机安排蚂蚁的初始位置

        # 更新每一个蚂蚁的行走路径
        for i in range(num_ant):
            for j in range(1, n):
                pathed = Path_short.path[i, :j]  # 蚂蚁i已经经过的路径
                # 蚂蚁i下个可访问城市
                allow_city = city_index[~np.isin(city_index, pathed)]

                # 利用轮盘赌法根据概率选择下一个城市
                prob = np.zeros(len(allow_city))
                for k in range(len(allow_city)):
                    prob[k] = (Path_short.Tau_info[pathed[-1], allow_city[k]]) ** (alpha) \
                              * ((Path_short.eta[pathed[-1], allow_city[k]]) ** beta)
                prob = prob / sum(prob)
                cumsum = np.cumsum(prob)
                pick = np.random.rand()
                for r in range(len(prob)):
                    if cumsum[r] >= pick:
                        Path_short.path[i, j] = allow_city[r]
                        break
        # 计算每个蚂蚁经过的路径距离
        fit = np.zeros(num_ant)
        for i in range(num_ant):
            fit[i] = Path_short.comp_fit(Path_short.path[i, :])

        # 存储当前迭代所有蚂蚁的最优路径
        min_index = np.argmin(fit)
        if iter_0 == 0:
            best_path.append(Path_short.path[min_index, :])
            best_fit.append(fit[min_index])
        else:
            if fit[min_index] < best_fit[-1]:
                best_path.append(Path_short.path[min_index, :])
                best_fit.append(fit[min_index])
            else:
                best_path.append(best_path[-1])
                best_fit.append(best_fit[-1])

        # 更新信息素
        Path_short.update_info(fit)
        '''
        if iter_0%100 == 0:
            print('第'+str(iter_0)+'步后的最短的路程: '+str( best_fit[-1]))
            print('第'+str(iter_0)+'步后的最优路径:')
            Path_short.out_path(best_path[-1])# 显示每一步的最优路径
        '''
        iter_0 += 1

    Path_short.best_path = best_path
    Path_short.best_fit = best_fit
    a = best_path[-1].tolist()
    # convert
    path = a
    V = len(path)
    aoao = []
    index = path.index(0)
    if path[index + 1] == V - 1:
        for i in list(range(0, V)):
            aoao.append(path[(index - i) % V])
    else:
        for i in list(range(0, len(path))):
            aoao.append(path[(index + i) % V])
    aoao.remove(V - 1)
    # a.remove(len(data)-1)
    return [round(best_fit[-1] - raw_Adj[1][-1], 1), aoao]


from python_tsp.exact import solve_tsp_dynamic_programming
def python_TSP(Adj):
    uuuu=np.array(Adj)
    permutation, distance = solve_tsp_dynamic_programming(uuuu)
    path=permutation
    V=len(path)
    aoao=[]
    index = path.index(0)
    if path[index+1]==V-1:
        for i in list(range(0,V)):
            aoao.append(path[(index-i)%V])
    else:
        for i in list(range(0,len(path))):
            aoao.append(path[(index+i)%V])
    aoao.remove(V-1)
    #permutation.remove(len(Adj)-1)
    return [round(distance-Adj[1][-1],1),aoao]