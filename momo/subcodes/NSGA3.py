# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 14:07:42 2022

@author: 86136
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
#from utils import uniformpoint,funfun,cal,GO,envselect,IGD
import copy
import random

from scipy.special import comb
from itertools import combinations
import numpy as np
import copy
import math
from property import *
from nonDominationSort import *

"""
选择算子 
"""
def select1(pool, pops, fits, ranks, distances):
    # 一对一锦标赛选择
    # pool: 新生成的种群大小
    nPop, nChr = pops.shape
    nF = fits.shape[1]
    newPops = np.zeros((pool, nChr))
    newFits = np.zeros((pool, nF))

    indices = np.arange(nPop).tolist()
    i = 0
    while i < pool:
        idx1, idx2 = random.sample(indices, 2)  # 随机挑选两个个体
        idx = compare(idx1, idx2, ranks, distances)
        newPops[i] = pops[idx]
        newFits[i] = fits[idx]
        i += 1
    return newPops, newFits

def compare(idx1, idx2, ranks, distances):
    # return: 更优的 idx
    if ranks[idx1] < ranks[idx2]:
        idx = idx1
    elif ranks[idx1] > ranks[idx2]:
        idx = idx2
    else:
        if distances[idx1] <= distances[idx2]:
            idx = idx2
        else:
            idx = idx1
    return idx
"""交叉算子 
混合线性交叉 
"""

def crossover(pops, pc,d, lb, rb):
    # 拷贝父代种群，以防止改变父代种群结构  ,d=0.25参数
    chrPops = pops.copy()
    nPop = chrPops.shape[0]
    for i in range(0, nPop):
        if np.random.rand() < pc:
            mother = chrPops[np.random.randint(nPop)]
            alpha1=np.random.rand()#生成2个随机数#,[alpha1,alpha2]=np.random.rand(2)
            r1=(-d)+(1+2*d)*alpha1
            #r2=(-d)+(1+2*d)*alpha2
            chrPops[i] = chrPops[i]+r1*(mother-chrPops[i])#混合线性交叉
            chrPops[i][chrPops[i]<lb] = lb
            chrPops[i][chrPops[i]>rb] = rb
    return chrPops

"""变异算子 
单点
"""
def mutate(pops, pm, nChr,m):
    nPop = pops.shape[0]
    for i in range(nPop):
        if np.random.rand() < pm:
            zz=np.random.rand(m)
            zzz = 4*zz-2#-2到2
            #pos = np.random.randint(0,nChr,1)
            pos = np.random.randint(0, nChr, m)#变异2个位置
            pops[i][pos] = zzz
    return pops

"""
种群或个体的适应度 
"""
def fitness_qed(mol,fp_0):
    # 计算种群或者个体的适应度，嵌入需解码为SMILES
    # 如果是1维需要转为2维
    nPop = len(mol)
    fits = np.array([ff_qed(mol[i], fp_0) for i in range(nPop)])
    return fits

def ff_qed(mol, fp_0):
    #pen_logP = penalized_logP(seq)#需改为种群
    qed = -QED(mol)
    sim = -tanimoto_similarity(mol, fp_0)
    return qed, sim#pen_logP,


def fitness_plogp(mol,fp_0):
    # 计算种群或者个体的适应度，嵌入需解码为SMILES
    # 如果是1维需要转为2维
    nPop = len(mol)
    fits = np.array([ff_plogp(mol[i], fp_0) for i in range(nPop)])
    return fits

def ff_plogp(mol, fp_0):
    pen_logP = -penalized_logP(mol)#需改为种群
    #qed = QED(seq)
    sim = -tanimoto_similarity(mol, fp_0)
    return pen_logP, sim#pen_logP,

##多目标QED,SIM,Plogp
def fitness_qedlogp(seq, mol,fp_0):
    # 计算种群或者个体的适应度，嵌入需解码为SMILES
    # 如果是1维需要转为2维
    nPop = len(mol)
    fits = np.array([ff_qedlogp(seq[i], mol[i], fp_0) for i in range(nPop)])

    return fits

def ff_qedlogp(seq, mol, fp_0):
    #pen_logP = penalized_logP(seq)#需改为种群
    qed = -QED(mol)
    pen_logP = -penalized_logP(mol)
    sim = -tanimoto_similarity(mol, fp_0)
    return qed, pen_logP, sim#pen_logP,

def fitness_qedjnkgsksa(seqs,mols):#,fp_0
    # 计算种群或者个体的适应度，嵌入需解码为SMILES
    # 如果是1维需要转为2维
    nPop = len(seqs)
    fits = np.array([ff_qedjnkgsksa(seqs[i], mols[i]) for i in range(nPop)])
    return fits

def ff_qedjnkgsksa(seq,mol):#fp_0
    #pen_logP = penalized_logP(seq)#需改为种群
    qed = -QED(mol)  # 需改为种群
    jnk3 = -jnk(seq)
    gskb = -gsk(seq)
    sa_nom = -normalize_sa(seq)
    #sim = tanimoto_similarity(mol, fp_0)
    return qed, sa_nom, gskb, jnk3 #sim  # pen_logP
#多目标qed,drd2,sim
def fitness_qeddrd(seqs,mols,fp_0):
    # 计算种群或者个体的适应度，嵌入需解码为SMILES
    # 如果是1维需要转为2维
    nPop = len(seqs)
    fits = np.array([ff_qeddrd(seqs[i], mols[i], fp_0) for i in range(nPop)])
    return fits

def ff_qeddrd(seq,mol,fp_0):
    #pen_logP = penalized_logP(seq)#需改为种群
    qed = -QED(mol)  # 需改为种群
    drd = -drd2(seq)
    #sa_nom = normalize_sa(seq)
    sim = -tanimoto_similarity(mol, fp_0)
    return qed, sim, drd  # pen_logP

#代码参考知乎https://zhuanlan.zhihu.com/p/146618031
def NDsort(mixpop,N,M):
    nsort = N#排序个数
    N,M = mixpop.shape[0],mixpop.shape[1]
    Loc1=np.lexsort(mixpop[:,::-1].T)#loc1为新矩阵元素在旧矩阵中的位置，从第一列依次进行排序
    mixpop2=mixpop[Loc1]
    Loc2=Loc1.argsort()#loc2为旧矩阵元素在新矩阵中的位置
    frontno=np.ones(N)*(np.inf)#初始化所有等级为np.inf
    #frontno[0]=1#第一个元素一定是非支配的
    maxfno=0#最高等级初始化为0
    while (np.sum(frontno < np.inf) < min(nsort,N)):#被赋予等级的个体数目不超过要排序的个体数目
        maxfno=maxfno+1
        for i in range(N):
            if (frontno[i] == np.inf):
                dominated = 0
                for j in range(i):
                    if (frontno[j] == maxfno):
                        m=0
                        flag=0
                        while (m<M and mixpop2[i,m]>=mixpop2[j,m]):
                            if(mixpop2[i,m]==mixpop2[j,m]):#相同的个体不构成支配关系
                                flag=flag+1
                            m=m+1
                        if (m>=M and flag < M):
                            dominated = 1
                            break
                if dominated == 0:
                    frontno[i] = maxfno
    frontno=frontno[Loc2]
    return frontno,maxfno

def uniformpoint(N,M):
    H1=1
    while (comb(H1+M-1,M-1)<=N):
        H1=H1+1
    H1=H1-1
    W=np.array(list(combinations(range(H1+M-1),M-1)))-np.tile(np.array(list(range(M-1))),(int(comb(H1+M-1,M-1)),1))
    W=(np.hstack((W,H1+np.zeros((W.shape[0],1))))-np.hstack((np.zeros((W.shape[0],1)),W)))/H1
    if H1<M:
        H2=0
        while(comb(H1+M-1,M-1)+comb(H2+M-1,M-1) <= N):
            H2=H2+1
        H2=H2-1
        if H2>0:
            W2=np.array(list(combinations(range(H2+M-1),M-1)))-np.tile(np.array(list(range(M-1))),(int(comb(H2+M-1,M-1)),1))
            W2=(np.hstack((W2,H2+np.zeros((W2.shape[0],1))))-np.hstack((np.zeros((W2.shape[0],1)),W2)))/H2
            W2=W2/2+1/(2*M)
            W=np.vstack((W,W2))#按列合并
    W[W<1e-6]=1e-6
    N=W.shape[0]
    return W,N

# 求两个向量矩阵的余弦值,x的列数等于y的列数
def pdist(x, y):
    x0 = x.shape[0]
    y0 = y.shape[0]
    xmy = np.dot(x, y.T)  # x乘以y
    xm = np.array(np.sqrt(np.sum(x ** 2, 1))).reshape(x0, 1)
    ym = np.array(np.sqrt(np.sum(y ** 2, 1))).reshape(1, y0)
    xmmym = np.dot(xm, ym)
    cos = xmy / xmmym
    return cos


def lastselection(popfun1, popfun2, K, Z, Zmin):
    # 选择最后一个front的解
    popfun = copy.deepcopy(np.vstack((popfun1, popfun2))) - np.tile(Zmin, (popfun1.shape[0] + popfun2.shape[0], 1))
    N, M = popfun.shape[0], popfun.shape[1]
    N1 = popfun1.shape[0]
    N2 = popfun2.shape[0]
    NZ = Z.shape[0]

    # 正则化
    extreme = np.zeros(M)
    w = np.zeros((M, M)) + 1e-6 + np.eye(M)
    for i in range(M):
        extreme[i] = np.argmin(np.max(popfun / (np.tile(w[i, :], (N, 1))), 1))

    # 计算截距
    extreme = extreme.astype(int)  # python中数据类型转换一定要用astype
    # temp = np.mat(popfun[extreme,:]).I
    temp = np.linalg.pinv(np.mat(popfun[extreme, :]))
    hyprtplane = np.array(np.dot(temp, np.ones((M, 1))))
    a = 1 / hyprtplane
    if np.sum(a == math.nan) != 0:
        a = np.max(popfun, 0)
    np.array(a).reshape(M, 1)  # 一维数组转二维数组
    # a = a.T - Zmin
    a = a.T
    popfun = popfun / (np.tile(a, (N, 1)))

    ##联系每一个解和对应向量
    # 计算每一个解最近的参考线的距离
    cos = pdist(popfun, Z)
    distance = np.tile(np.array(np.sqrt(np.sum(popfun ** 2, 1))).reshape(N, 1), (1, NZ)) * np.sqrt(1 - cos ** 2)
    # 联系每一个解和对应的向量
    d = np.min(distance.T, 0)
    pi = np.argmin(distance.T, 0)

    # 计算z关联的个数
    rho = np.zeros(NZ)
    for i in range(NZ):
        rho[i] = np.sum(pi[:N1] == i)

    # 选出剩余的K个
    choose = np.zeros(N2)
    choose = choose.astype(bool)
    zchoose = np.ones(NZ)
    zchoose = zchoose.astype(bool)
    while np.sum(choose) < K:
        # 选择最不拥挤的参考点
        temp = np.ravel(np.array(np.where(zchoose == True)))
        jmin = np.ravel(np.array(np.where(rho[temp] == np.min(rho[temp]))))
        j = temp[jmin[np.random.randint(jmin.shape[0])]]
        #        I = np.ravel(np.array(np.where(choose == False)))
        #        I = np.ravel(np.array(np.where(pi[(I+N1)] == j)))
        I = np.ravel(np.array(np.where(pi[N1:] == j)))
        I = I[choose[I] == False]
        if (I.shape[0] != 0):
            if (rho[j] == 0):
                s = np.argmin(d[N1 + I])
            else:
                s = np.random.randint(I.shape[0])
            choose[I[s]] = True
            rho[j] = rho[j] + 1
        else:
            zchoose[j] = False
    return choose


def envselect(mixpop, mixpopfun, N, Z, Zmin, M, D):
    # 非支配排序
    #mixpopfun = cal(mixpop, name, M, D)
    frontno, maxfno = NDsort(mixpopfun, N, M)
    Next = frontno < maxfno
    # 选择最后一个front的解
    Last = np.ravel(np.array(np.where(frontno == maxfno)))
    choose = lastselection(mixpopfun[Next, :], mixpopfun[Last, :], N - np.sum(Next), Z, Zmin)
    Next[Last[choose]] = True
    # 生成下一代
    pop = copy.deepcopy(mixpop[Next, :])
    popfun = copy.deepcopy(mixpopfun[Next, :])
    return pop, popfun#, Next


def GO(pop, t1, t2, pc, pm):
    pop1 = copy.deepcopy(pop[0:int(pop.shape[0] / 2), :])
    pop2 = copy.deepcopy(pop[(int(pop.shape[0] / 2)):(int(pop.shape[0] / 2) * 2), :])
    N, D = pop1.shape[0], pop1.shape[1]
    # 模拟二进制交叉
    beta = np.zeros((N, D))
    mu = np.random.random_sample([N, D])
    beta[mu <= 0.5] = (2 * mu[mu <= 0.5]) ** (1 / (t1 + 1))
    beta[mu > 0.5] = (2 - 2 * mu[mu > 0.5]) ** (-1 / (t1 + 1))
    beta = beta * ((-1) ** (np.random.randint(2, size=(N, D))))
    beta[np.random.random_sample([N, D]) < 0.5] = 1
    beta[np.tile(np.random.random_sample([N, 1]) > pc, (1, D))] = 1
    off = np.vstack(((pop1 + pop2) / 2 + beta * (pop1 - pop2) / 2, (pop1 + pop2) / 2 - beta * (pop1 - pop2) / 2))
    # 多项式变异
    low = np.zeros((2 * N, D))
    up = np.ones((2 * N, D))
    site = np.random.random_sample([2 * N, D]) < pm / D
    mu = np.random.random_sample([2 * N, D])
    temp = site & (mu <= 0.5)
    off[off < low] = low[off < low]
    off[off > up] = up[off > up]
    off[temp] = off[temp] + (up[temp] - low[temp]) * ((2 * mu[temp] + (1 - 2 * mu[temp]) * (
                (1 - (off[temp] - low[temp]) / (up[temp] - low[temp])) ** (t2 + 1))) ** (1 / (t2 + 1)) - 1)
    temp = site & (mu > 0.5)
    off[temp] = off[temp] + (up[temp] - low[temp]) * (1 - (2 * (1 - mu[temp]) + 2 * (mu[temp] - 0.5) * (
                (1 - (up[temp] - off[temp]) / (up[temp] - low[temp])) ** (t2 + 1))) ** (1 / (t2 + 1)))

    return off

'''
def funfun(M, N, name):
    # 种群初始化
    D = M + 4  # 定义自变量个数为目标个数加4
    low = np.zeros((1, D))
    up = np.ones((1, D))
    pop = np.tile(low, (N, 1)) + (np.tile(up, (N, 1)) - np.tile(low, (N, 1))) * np.random.rand(N, D)

    # 计算PF
    if name == 'DTLZ1':
        # g=np.transpose(np.mat(100*(D-M+1+np.sum(((pop[:,(M-1):]-0.5)**2-np.cos(20*np.pi*(pop[:,(M-1):]-0.5))),1))))
        g = np.array(100 * (
                    D - M + 1 + np.sum(((pop[:, (M - 1):] - 0.5) ** 2 - np.cos(20 * np.pi * (pop[:, (M - 1):] - 0.5))),
                                       1))).reshape(N, 1)
        popfun = np.multiply(0.5 * np.tile(1 + g, (1, M)),
                             (np.fliplr((np.hstack((np.ones((g.shape[0], 1)), pop[:, :(M - 1)]))).cumprod(1))))
        popfun = np.multiply(popfun, (np.hstack((np.ones((g.shape[0], 1)), 1 - np.fliplr(pop[:, :(M - 1)])))))
        P, nouse = uniformpoint(N, M)
        P = P / 2
    elif name == 'DTLZ2':
        # g=np.transpose(np.mat(np.sum((pop[:,(M-1):]-0.5)**2,1)))
        g = np.array(np.sum((pop[:, (M - 1):] - 0.5) ** 2, 1)).reshape(N, 1)
        popfun = np.multiply(np.tile(1 + g, (1, M)), (
            np.fliplr((np.hstack((np.ones((g.shape[0], 1)), np.cos(pop[:, :(M - 1)] * (np.pi / 2))))).cumprod(1))))
        popfun = np.multiply(popfun, (
            np.hstack((np.ones((g.shape[0], 1)), 1 - np.sin(np.fliplr(pop[:, :(M - 1)]) * (np.pi / 2))))))
        P, nouse = uniformpoint(N, M)
        # P = P/np.tile(np.transpose(np.mat(np.sqrt(np.sum(P**2,1)))),(1,M))
        P = P / np.tile(np.array(np.sqrt(np.sum(P ** 2, 1))).reshape(P.shape[0], 1), (1, M))
    elif name == 'DTLZ3':
        g = np.array(100 * (
                    D - M + 1 + np.sum(((pop[:, (M - 1):] - 0.5) ** 2 - np.cos(20 * np.pi * (pop[:, (M - 1):] - 0.5))),
                                       1))).reshape(N, 1)
        popfun = np.multiply(np.tile(1 + g, (1, M)), (
            np.fliplr((np.hstack((np.ones((g.shape[0], 1)), np.cos(pop[:, :(M - 1)] * (np.pi / 2))))).cumprod(1))))
        popfun = np.multiply(popfun, (
            np.hstack((np.ones((g.shape[0], 1)), 1 - np.sin(np.fliplr(pop[:, :(M - 1)]) * (np.pi / 2))))))
        P, nouse = uniformpoint(N, M)
        # P = P/np.tile(np.transpose(np.mat(np.sqrt(np.sum(P**2,1)))),(1,M))
        P = P / np.tile(np.array(np.sqrt(np.sum(P ** 2, 1))).reshape(P.shape[0], 1), (1, M))

    return pop, popfun, P, D
'''
'''
def cal(pop, name, M, D):
    N = pop.shape[0]
    if name == 'DTLZ1':
        g = np.array(100 * (
                    D - M + 1 + np.sum(((pop[:, (M - 1):] - 0.5) ** 2 - np.cos(20 * np.pi * (pop[:, (M - 1):] - 0.5))),
                                       1))).reshape(N, 1)
        popfun = np.multiply(0.5 * np.tile(1 + g, (1, M)),
                             (np.fliplr((np.hstack((np.ones((g.shape[0], 1)), pop[:, :(M - 1)]))).cumprod(1))))
        popfun = np.multiply(popfun, (np.hstack((np.ones((g.shape[0], 1)), 1 - np.fliplr(pop[:, :(M - 1)])))))

    elif name == 'DTLZ2':
        g = np.array(np.sum((pop[:, (M - 1):] - 0.5) ** 2, 1)).reshape(N, 1)
        popfun = np.multiply(np.tile(1 + g, (1, M)), (
            np.fliplr((np.hstack((np.ones((g.shape[0], 1)), np.cos(pop[:, :(M - 1)] * (np.pi / 2))))).cumprod(1))))
        popfun = np.multiply(popfun, (
            np.hstack((np.ones((g.shape[0], 1)), 1 - np.sin(np.fliplr(pop[:, :(M - 1)]) * (np.pi / 2))))))

    elif name == 'DTLZ3':
        g = np.array(100 * (
                    D - M + 1 + np.sum(((pop[:, (M - 1):] - 0.5) ** 2 - np.cos(20 * np.pi * (pop[:, (M - 1):] - 0.5))),
                                       1))).reshape(N, 1)
        popfun = np.multiply(np.tile(1 + g, (1, M)), (
            np.fliplr((np.hstack((np.ones((g.shape[0], 1)), np.cos(pop[:, :(M - 1)] * (np.pi / 2))))).cumprod(1))))
        popfun = np.multiply(popfun, (
            np.hstack((np.ones((g.shape[0], 1)), 1 - np.sin(np.fliplr(pop[:, :(M - 1)]) * (np.pi / 2))))))
    return popfun
'''

def EuclideanDistances(A, B):
    BT = B.transpose()
    # vecProd = A * BT
    vecProd = np.dot(A, BT)
    # print(vecProd)
    SqA = A ** 2
    # print(SqA)
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))
    # print(sumSqAEx)

    SqB = B ** 2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))
    SqED = sumSqBEx + sumSqAEx - 2 * vecProd
    SqED[SqED < 0] = 0.0
    ED = np.sqrt(SqED)
    return ED


def IGD(popfun, PF):
    distance = np.min(EuclideanDistances(PF, popfun), 1)
    score = np.mean(distance)
    return score

'''
#参数设置
N_GENERATIONS = 40                                 # 迭代次数
POP_SIZE = 100                                      # 种群大小
name = 'DTLZ1'                                      # 测试函数选择，目前可供选择DTLZ1,DTLZ2,DTLZ3
M = 3                                               # 目标个数
t1 = 20                                             # 交叉参数t1
t2 = 20                                             # 变异参数t2
pc = 1                                              # 交叉概率
pm = 1                                              # 变异概率
#画图部分
if(M<=3):
    fig = plt.figure()
    ax = Axes3D(fig)

###################################################################################################################################################################
#产生一致性的参考点和随机初始化种群
Z,N = uniformpoint(POP_SIZE,M)#生成一致性的参考解##
pop,popfun,PF,D = funfun(M,N,name)#生成初始种群及其适应度值，真实的PF,自变量个数
popfun = cal(pop,name,M,D)#计算适应度函数值
Zmin = np.array(np.min(popfun,0)).reshape(1,M)#求理想点
#ax.scatter(Z[:,0],Z[:,1],Z[:,2],c='r')
#ax.scatter(PF[:,0],PF[:,1],PF[:,2],c='b')

#迭代过程
for i in range(N_GENERATIONS):
    print("第{name}次迭代".format(name=i))
    matingpool=random.sample(range(N),N)
    off = GO(pop[matingpool,:],t1,t2,pc,pm)#遗传算子,模拟二进制交叉和多项式变异
    offfun = cal(off,name,M,D)#计算适应度函数
    mixpop = copy.deepcopy(np.vstack((pop, off)))
    Zmin = np.array(np.min(np.vstack((Zmin,offfun)),0)).reshape(1,M)#更新理想点
    mixpopfun=cal(mixpop,name,M,D)
    pop, popfun = envselect(mixpop,mixpopfun,N,Z,Zmin,name,M,D)
    #popfun = cal(pop,name,M,D)
    if(M<=3):
        ax.cla()
        type1 = ax.scatter(popfun[:,0],popfun[:,1],popfun[:,2],c='g')
        plt.pause(0.00001)

# 绘制PF
if(M<=3):
    type2 = ax.scatter(PF[:,0],PF[:,1],PF[:,2],c='r',marker = 'x',s=200)
    plt.legend((type1, type2), (u'Non-dominated solution', u'PF'))
else:
    fig1 = plt.figure()
    plt.xlim([0,M])
    for i in range(pop.shape[0]):
        plt.plot(np.array(pop[i,:]))
plt.show()
'''



'''
smi = 'CCCCCC1CCC(CCCCCCCCNCc2ccc([O-])c[nH+]2)CC1'
mol = Chem.MolFromSmiles(smi)
print(Descriptors.MolWt(mol))
'''
