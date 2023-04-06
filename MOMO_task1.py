import argparse
import os
from tensorboardX import SummaryWriter
# import tensorflow as tf
import time

from momo.subcodes.models import CDDDModel

from momo.subcodes.NSGA2 import *
from momo.subcodes.property import *
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)

# Suppress warnings
# tf.logging.set_verbosity(tf.logging.ERROR)
# RDLogger.logger().setLevel(RDLogger.CRITICAL)


if __name__ == "__main__":
    # 20
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=None,
                        help='Random seed.')

    # parser.add_argument('seq', type=str, help='Sequence to optimize.')
    parser.add_argument("--smile_path", default="qed_test.csv")
    parser.add_argument("--seq", default='opti')
    parser.add_argument("--opt", default='qed',
                        choices=['qed', 'logP', 'drd', 'jnk', 'gsk'])
    args = parser.parse_args()

    if args.opt == 'qed':
        ff = ff_qed
        fitness = fitness_qed
        tre_pre = 0.9
        tre_sim = 0.4
        col = ['SMILES', 'mol_id', 'qed', 'sim']
    elif args.opt == 'logP':
        ff = ff_plogp
        tre_sim = 0.4
        fitness = fitness_plogp
    elif args.opt == 'drd':
        ff = ff_drd
        fitness = fitness_drd
        tre_pre = 0.5
        tre_sim = 0.3
        col = ['SMILES', 'mol_id', 'drd', 'sim']
    elif args.opt == 'gsk':
        ff = ff_qeddrd
        fitness = fitness_qeddrd
        tre_pre = 0.5
        tre_sim = 0.3
        col = ['SMILES', 'mol_id', 'gsk', 'sim']
    elif args.opt == 'jnk':
        ff = ff_qedjnksa
        fitness = fitness_qedjnksa
        tre_pre = 0.5
        tre_sim = 0.3
        col = ['SMILES', 'mol_id', 'jnk3', 'sim']


    # device = 'cuda'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:', device)
    if args.seed is not None:
        np.random.seed(args.seed)
        # tf.set_random_seed(args.seed)
        torch.manual_seed(args.seed)


    ######加载预训练的CDDD
    model = CDDDModel()
    # canonicalize
    data = pd.read_csv('./momo/data/qed_test.csv').values
    orismi_all = pd.read_csv('./momo/data/oripops_qed/QMO_qed_mol800_optsmiles.csv').values

    smi_pro_tuple = []#保留所有分子最后一代，一个分子npop个子代
    # 参数设置
    pern = 500     # 扰动隐向量的个数，扰动多个然后排序选npop个
    sita = 0.5      #扰动程度的参数
    nPop = 100      #种群个体数
    pc = 1          #交叉率
    pm = 0.5        #变异率
    nChr = 512       #染色体长度
    lb = -1         #下界
    rb = 1          #上界
    d = 0.25         #线性交叉参数
    nIter = 50      #迭代次数
    m = 5           # 变异个数
    restart = 5
    SR = 0          # 记录优化成功的分子数0.9，0.4
    t1 = time.time()# 开始时间
    mm1 = 0
    mm2 = 800
    qed = []
    for i in range(mm1,mm2):
        nn = i
        aa = orismi_all[:, 1] == i
        index = np.where(aa == 1)
        orismi = orismi_all[index]  # 提取指定分子的初始种群
        if orismi.shape[0] != 0:
            #一个分子SMILES序列
            smiles = data[i][0]
            print(smiles)
            # 一个分子SMILES序列
            mol_0 = Chem.MolFromSmiles(smiles)
            seq = Chem.MolToSmiles(mol_0)
            #print(seq)
            run_str = 'optiza'
            results_dir = os.path.join('results', args.seq)
            os.makedirs(results_dir, exist_ok=True)
            writer = SummaryWriter(os.path.join('runs', args.seq, run_str))
            t2 = time.time()  # 编码前时间
            z_0 = model.encode(smiles) # 对分子序列进行编码
            fp_0 = morgan_fingerprint(Chem.MolFromSmiles(seq))  # 分子序列的摩根指纹，用于计算相似性
            fits_0 = ff(seq, mol_0, fp_0)
            print(fits_0)
            pops_smi = np.zeros((z_0.shape[0] * orismi.shape[0], z_0.shape[1]))
            nonid = []
            mol = []
            ii = 0
            while ii < orismi.shape[0]:
                qmo = 0
                if orismi[ii][2] >= 0.4 and orismi[ii][3] >= 0.9:#sim,qed
                    tuple = (orismi[ii][0], nn, orismi[ii][3], orismi[ii][2])  #qed,sim
                    print('success:', tuple)
                    smi_pro_tuple.append(tuple)
                    SR = SR + 1
                    ii = orismi.shape[0] + 1
                    qmo += 1
                else:
                    try:
                        pops_smi[ii] = model.encode(orismi[ii][0])
                        mol1 = Chem.MolFromSmiles(orismi[ii][0])
                        if mol1 is not None:
                            mol.append(mol1)
                        else:
                            nonid.append(ii)
                        ii = ii + 1
                    except:
                        nonid.append(ii)
                        ii = ii + 1
            print(qmo)
            if qmo == 0:
                pops_smi_val = np.delete(pops_smi, nonid, 0)
                orismi_val = np.delete(orismi, nonid, 0)
                fits_pops = fitness(orismi_val, mol, fp_0)  # 适应度计算
                fits_pops = np.nan_to_num(fits_pops)

                r = 1#重启

                while r <= restart:
                ########### 隐向量的扰动数量、控制相似性的系数，越大生成的相似性越低
                    ####### 对隐向量加入高斯噪声进行扰动
                    dispop = np.zeros((z_0.shape[0] * pern, z_0.shape[1]))
                    gauss = np.random.normal(0, 1, (pern, z_0.shape[1]))
                    dispop[:pern] = gauss * sita + z_0.numpy()
                    dispop = torch.from_numpy(dispop)
                    dismol, dissmiles = model.decode(dispop)
                    disfits = fitness(dissmiles, dismol, fp_0)  # 适应度计算
                    disfits[np.isnan(disfits)] = 0
                    ranks = nonDominationSort(dispop, disfits)  # 非支配排序，对潜向量还是分子？？
                    distances = crowdingDistanceSort(dispop, disfits, ranks)  # 拥挤度
                    dispop, disfits = select1(nPop, dispop, disfits, ranks, distances)
                    pops, fits = optSelect(dispop, disfits, pops_smi_val, fits_pops)
                    iter = 1
                    while iter <= nIter:
                        # 进度条
                        print("【进度】【{0:20s}】【正在进行{1}代...】【共{2}代】". \
                              format('▋' * int(iter / nIter * 20), iter, nIter), end='\r')
                        #交叉变异
                        chrpops = crossover(pops, pc, d, lb, rb)#混合线性交叉
                        chrpops = mutate(chrpops, pm, nChr,m) # 变异产生子种群
                        # 解码子代种群分子
                        chrpops = torch.from_numpy(chrpops)
                        chrmol, chrsmiles = model.decode(chrpops)  ###解码潜在表示，分子和SMILES
                        chrfits = fitness(chrsmiles, chrmol, fp_0)
                        chrfits = np.nan_to_num(chrfits)
                        # 从原始种群和子种群中筛选nPop个
                        pops, fits = optSelect(pops, fits, chrpops, chrfits)
                        iter_num = []
                        for i in range(pops.shape[0]):
                            iter_num.append((fits[:][i] >= [tre_pre, tre_sim]).all())
                        if 1 in iter_num:
                            iter = nIter + 1
                        else:
                            iter = iter + 1
                        print(iter)
                    # 解码最终种群分子
                    #[endsmiles, nonid] = decode_from_jtvae(pops, opts, model)
                    pops = torch.from_numpy(pops)
                    endmol, endsmiles = model.decode(pops)
                    endsmiles = np.array(endsmiles)
                    endfits = fits
                    rr = []#储存endfits是否满足>=0.9,>=0.4，True=1，False=0
                    unique_smiles = []  # 储存当前分子唯一的SMILES,可放在重启前
                    for i in range(len(endsmiles)):
                        rr.append((endfits[:][i] >= [tre_pre, tre_sim]).all())
                        if endsmiles[i] not in unique_smiles and endfits[i][0] >= tre_pre:# and endfits[i][1] >= 0.4 and endfits[i][0] >= 0.9
                            unique_smiles.append(endsmiles[i])
                            tuple = (endsmiles[i], nn, endfits[i][0], endfits[i][1])
                            smi_pro_tuple.append(tuple)
                    #print(rr)
                    if 1 in rr:
                        r = restart+1
                        SR = SR + 1
                    else:
                        r = r+1
                    print('restart_run:',r)
            df_to_save_A_to_B = pd.DataFrame(smi_pro_tuple, columns=['SMILES','mol_id', 'qed', 'sim'])
            df_to_save_A_to_B.to_csv('MOMO_qed_mol800.csv', index=False)
            result = [nn - mm1 + 1, SR]
            print('result-all,SR:', result)
            np.savetxt('MOMO_qed_mol800.txt', result, fmt='%s')

            t3 = time.time()  # 一个分子重启动进化后时间
            time_onemol = t3 - t2#扰动个数
            print(time_onemol)
    t4 = time.time()  # 解码进化总时间
    time_all = t4 - t1
    print(time_all)
    # 从原始种群和子种群中筛选nPop个
