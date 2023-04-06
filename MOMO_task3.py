import argparse
import os
from tensorboardX import SummaryWriter
# import tensorflow as tf
import time

from momo.subcodes.models import CDDDModel

from momo.subcodes.NSGA3 import *
from momo.subcodes.property import *

from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)
from momo.subcodes.nonDominationSort import *

# Suppress warnings
# tf.logging.set_verbosity(tf.logging.ERROR)
# RDLogger.logger().setLevel(RDLogger.CRITICAL)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=123456789,
                        help='Random seed.')

    # parser.add_argument('seq', type=str, help='Sequence to optimize.')
    parser.add_argument("--smile_path", default="qedplogp_test.csv")
    parser.add_argument("--seq", default='opti')

    parser.add_argument("--opt", default='qeddrd',
                        choices=['qed', 'logP', 'qedplogp', 'qeddrd', 'qedsajnk', 'qedsagsk', 'qedsadrd2'])
    args = parser.parse_args()
    if args.opt == 'qed':
        ff = ff_qed
        fitness = fitness_qed
    elif args.opt == 'logP':
        ff = ff_plogp
        fitness = fitness_plogp
    elif args.opt == 'qedplogP':
        ff = ff_qedlogp
        fitness = fitness_qedlogp
    elif args.opt == 'qeddrd':
        ff = ff_qeddrd
        fitness = fitness_qeddrd
        tres = [-0.8, -0.3, -0.4]
        col = ['SMILES', 'mol_id', 'qed', 'sim', 'drd']
    args = parser.parse_args()
    # device = 'cuda'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:', device)
    if args.seed is not None:
        np.random.seed(args.seed)
        # tf.set_random_seed(args.seed)
        torch.manual_seed(args.seed)

    ######加载预训练的JTVAE
    model = CDDDModel()
    # canonicalize
    data = pd.read_csv('./momo/data/qeddrd_test.csv').values
    orismi_all = pd.read_csv('./momo/data/oripops_qeddrd/QMO_qeddrd_mol200_optsmiles.csv').values
    smi_pro_tuple = []  # 保留所有分子最后一代，一个分子npop个子代
    #r_time = []  # 保留所有分子restart次数
    improvement = []
    # 参数设置
    pern = 500  # 扰动隐向量的个数，扰动多个然后排序选npop个
    sita = 0.5  # 扰动程度的参数
    nPop = 100  # 种群个体数
    pc = 1  # 交叉率
    pm = 0.5  # 变异率
    nChr = 512  # 染色体长度
    lb = -1  # 下界
    rb = 1  # 上界
    d = 0.25  # 线性交叉参数
    nIter = 100  # 迭代次数
    restart = 1
    m = 5  # 变异个数
    #restart = 5  # 最大重启次数
    SR = 0  # 记录优化成功的分子数0.9，0.4
    t1 = time.time()  # 开始时间
    D = nChr
    M = 5
    mm1 = 0
    mm2 = 200
    for i in range(mm1,mm2):
        nn = i
        aa = orismi_all[:, 1] == i
        index = np.where(aa == 1)
        orismi = orismi_all[index]  # 提取指定分子的初始种群
        if orismi.shape[0] != 0:
            #一个分子SMILES序列
            smiles = data[i][0]
            print('ori_smiles:', smiles)
            # 一个分子SMILES序列
            mol_0 = Chem.MolFromSmiles(smiles)
            seq = Chem.MolToSmiles(mol_0)
            #print(seq)
            run_str = 'optiza'
            results_dir = os.path.join('results', args.seq)
            os.makedirs(results_dir, exist_ok=True)
            writer = SummaryWriter(os.path.join('runs', args.seq, run_str))
        #调试
            t2 = time.time()#编码前时间
            z_0 = model.encode(smiles)  # 对分子序列进行编码
            fp_0 = morgan_fingerprint(Chem.MolFromSmiles(seq))  # 分子序列的摩根指纹，用于计算相似性
            fits_0 = ff(seq, mol_0, fp_0)
            print(fits_0)
            rec_mol, rec_smiles = model.decode(z_0)  ###解码潜在表示，分子和SMILES
            print('rec_smiles:', rec_smiles)
            # 评价适应度
            fits_rec = fitness(rec_smiles, rec_mol, fp_0)

            pops_smi = np.zeros((z_0.shape[0] * orismi.shape[0], z_0.shape[1]))
            nonid = []
            mol = []
            for i in range(orismi.shape[0]):
                try:
                    pops_smi[i] = model.encode(orismi[i][0])
                    mol1 = Chem.MolFromSmiles(orismi[i][0])
                    if mol1 is not None:
                        mol.append(mol1)
                    else:
                        nonid.append(i)
                except:
                    nonid.append(i)
            pops_smi_val = np.delete(pops_smi, nonid, 0)
            orismi_val = np.delete(orismi, nonid, 0)
            fits_pops = fitness(orismi_val, mol, fp_0)  # 适应度计算
            fits_pops[np.isnan(fits_pops)] = 0

            ########### 隐向量的扰动数量、控制相似性的系数，越大生成的相似性越低，容易产生空集
            r = 1  # 重启
            while r <= restart:
                dispop = np.zeros((z_0.shape[0] * pern, z_0.shape[1]))
                gauss = np.random.normal(0, 1, (pern, z_0.shape[1]))
                dispop[:pern] = gauss * sita + z_0.numpy()
                dispop = torch.from_numpy(dispop)
                dismol, dissmiles = model.decode(dispop)
                disfits = fitness(dissmiles, dismol, fp_0)  # 适应度计算
                disfits[np.isnan(disfits)] = 0

                # 产生一致性的参考点和随机初始化种群
                Z, N = uniformpoint(nPop, M)  # 生成一致性的参考解##
                Zmin = np.array(np.min(fits_pops, 0)).reshape(1, M)  # 求理想点
                mixpop = copy.deepcopy(np.vstack((dispop, pops_smi_val)))
                mixpopfun = copy.deepcopy(np.vstack((disfits, fits_pops)))
                pops, fits = envselect(mixpop, mixpopfun, N, Z, Zmin, M, D)

                iter = 1
                while iter <= nIter:
                    # 进度条
                    print("【进度】【{0:20s}】【正在进行{1}代...】【共{2}代】". \
                          format('▋' * int(iter / nIter * 20), iter, nIter), end='\r')
                    chrpops = crossover(pops, pc, d, lb, rb)#混合线性交叉
                    chrpops = mutate(chrpops, pm, nChr,m) # 变异产生子种群
                    # 解码子代种群分子
                    chrpops = torch.from_numpy(chrpops)
                    chrmol, chrsmiles = model.decode(chrpops)  ###解码潜在表示，分子和SMILES
                    chrfits = fitness(chrsmiles, chrmol, fp_0)
                    chrfits[np.isnan(chrfits)] = 0
                    # 从原始种群和子种群中筛选nPop个
                    Zmin = np.array(np.min(np.vstack((Zmin, chrfits)), 0)).reshape(1, M)  # 更新理想点
                    mixpop = copy.deepcopy(np.vstack((pops, chrpops)))
                    mixpopfun = copy.deepcopy(np.vstack((fits, chrfits)))
                    pops, fits = envselect(mixpop, mixpopfun, N, Z, Zmin, M, D)
                    iter = iter + 1

                # 解码最终种群分子
                pops = torch.from_numpy(pops)
                endmol, endsmiles = model.decode(pops)
                endsmiles = np.array(endsmiles)
                endfits = fits
            #保存
                rr = []
                unique_smiles = []  # 储存当前分子唯一的SMILES,可放在重启前
                for i in range(len(endsmiles)):
                    if endsmiles[i] not in unique_smiles:   # and endfits[i][2] >= 0.2 and endfits[i][0] >= 0.85 and endfits[i][1] >= 0:
                        unique_smiles.append(endsmiles[i])
                        tuple = (endsmiles[i], nn, -endfits[i][0], -endfits[i][1], -endfits[i][2])  # , endfits[i][0]-fits_0[0]
                        smi_pro_tuple.append(tuple)
                        if (endfits[:][i] <= tres).all()==1:#(fits_0[0]  + 0.1)
                            rr.append(1)
                if 1 in rr:
                    r = restart+1
                    SR = SR + 1
                else:
                    r = r+1
                print('restart_run:', r)
            print('save-mol:', nn)
            t3 = time.time()  # 解码后时间
            df_to_save_A_to_B = pd.DataFrame(smi_pro_tuple, columns=col)  # , 'qed_imp'
            df_to_save_A_to_B.to_csv('MOMO_nsga3_qeddrd_mol200.csv', index=False)
            result = [nn - mm1 + 1, SR]
            print('reslut-all,SR:', result)
            np.savetxt('MOMO_nsga3_qeddrd_mol200_result.txt', result, fmt='%s')
            time_one = t3 - t2  # 扰动个数
            print(time_one)
    t4 = time.time()  # 解码进化总时间
    time_all = t4 - t1
    print(time_all)

