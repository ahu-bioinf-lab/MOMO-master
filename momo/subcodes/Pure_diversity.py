#import platgo as pg

'''
function Score = PD(Parameter)
# <metric> <max>
# Pure diversity

    [PopObj,PF,~,~,~,~] = deal(Parameter{:});
    N = size(PopObj,1);
    C = false(N);
    C(logical(eye(size(C)))) = true;
    D = pdist2(PopObj,PopObj,'minkowski',0.1);
    D(logical(eye(size(D)))) = inf;
    Score = 0;
    for k = 1 : N-1
        while true
            [d,J] = min(D,[],2);
            [~,i] = max(d);
            if D(J(i),i) ~= -inf
                D(J(i),i) = inf;
            end
            if D(i,J(i)) ~= -inf
                D(i,J(i)) = inf;
            end
            P = any(C(i,:),1);
            while ~P(J(i))
                newP = any(C(P,:),1);
                if P == newP
                    break;
                else
                    P = newP;
                end
            end
            if ~P(J(i))
                break;
            end
        end
        C(i,J(i)) = true;
        C(J(i),i) = true;
        D(i,:)    = -inf;
        Score     = Score + d(i);
    end
end
'''
import numpy as np

def PD_cal(PopObj):
    # <metric> <max>
    # Pure diversity
    #PopObj, PF, _, _, _, _ = Parameter
    N = PopObj.shape[0]
    C = np.eye(N, dtype=bool)
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            #计算参数为0.1的闵可夫斯基距离
            D[i, j] = np.power(np.sum(np.power(np.abs(PopObj[i, :] - PopObj[j, :]), 0.1)), 1/0.1)
            D[j, i] = D[i, j]
    np.fill_diagonal(D, np.inf)
    Score = 0

    for k in range(N-1):
        while True:
            #每行的最小值，结果存储在向量d向量
            d = np.min(D, axis=1)
            #d中最大的索引（最大的行）
            i = np.argmax(d)
            #J每行最小值的索引
            J = np.argmin(D, axis=1)
            #所有行中最小值最大的位置
            if D[J[i], i] != -np.inf:
                D[J[i], i] = np.inf
            if D[i, J[i]] != -np.inf:
                D[i, J[i]] = np.inf
            #c中i行是否有true
            #P = np.any(C[i, :], axis=0)
            P = C[i,:]
            while not P[J[i]]:
                newP = np.any(C[P, :], axis=0)
                if np.array_equal(P, newP):
                    break
                else:
                    P = newP
            if not P[J[i]]:
                break
        C[i, J[i]] = True
        C[J[i], i] = True
        D[i, :] = -np.inf
        Score += d[i]

    return Score
'''
if __name__ == "__main__":
    decs = np.random.random((100, 3))
    #pop = pg.Population(decs=decs)
    #pop.objv = np.array([[1.3,1.2,1.1],[1.1,1.2,1.3],[1.2,1.2,1.2]])
    #fits = np.array([[1.3, 1.2, 1.1], [1.1, 1.2, 1.3], [1.2, 1.2, 1.2]])
    fits = np.array([[0.5, 1.2, 1.1], [0.3, 1.2, 1.3], [0.7, 1.2, 1.2]])
    pareto = np.array([[0, 0, 0]])
    #score = cal_hv(pop, pareto)
    score = PD_cal(fits)
    print(score)
'''

