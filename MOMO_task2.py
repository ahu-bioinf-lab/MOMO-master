import argparse
import os
from tensorboardX import SummaryWriter
# import tensorflow as tf
import time


import torch
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
    parser.add_argument("--smile_path", default="qed_testval.csv")
    parser.add_argument("--seq", default='opti')

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
    data = pd.read_csv('./momo/data/plogp_testval.csv').values
    orismi_all = pd.read_csv('./momo/data/oripops_plogp/QMO_plogp_mol50_optsmiles.csv').values
    # fit_mol = []
    smi_pro_tuple = []  # 保留所有分子最后一代，一个分子npop个子代
    smi_sr_4 = []
    #r_time = []  # 保留所有分子restart次数
    improvement = []
    improvement6 = []
    SR=0
    SR6=0
    # 参数设置
    pern = 500  #
    sita = 0.5  # The parameter of the degree of disturbance
    nPop = 100  # Population number
    pc = 1  # Crossover rate
    pm = 0.5  # Variation rate
    nChr = 512
    lb = -1
    rb = 1
    d = 0.25  # Linear crossover parameter
    nIter = 100  # Number of iterations
    m = 5  # Number of variations
    D = nChr
    M = 2
    #restart = 5  # 最大重启次数
    t1 = time.time()  # 开始时间
    mm1 = 0
    mm2 = 50
    for i in range(mm1,mm2):
        nn = i
        aa = orismi_all[:, 1] == i
        index = np.where(aa == 1)
        orismi = orismi_all[index]#提取指定分子的初始种群
        if orismi.shape[0]!=0:
            smiles = data[i][0]
            print('ori_smiles:', smiles)
            mol_0 = Chem.MolFromSmiles(smiles)
            seq = Chem.MolToSmiles(mol_0)
            run_str = 'optiza'
            results_dir = os.path.join('results', args.seq)
            os.makedirs(results_dir, exist_ok=True)
            writer = SummaryWriter(os.path.join('runs', args.seq, run_str))
            t2 = time.time()
            z_0 = model.encode(smiles)
            fp_0 = morgan_fingerprint(Chem.MolFromSmiles(seq))
            fits_0 = ff_plogp(mol_0, fp_0)
            print(fits_0)
            rec_mol, rec_smiles = model.decode(z_0)
            print('rec_smiles:', rec_smiles)
            fits_rec = fitness_plogp(rec_mol, fp_0)
            ########### disturb original population
            dispop = np.zeros((z_0.shape[0] * pern, z_0.shape[1]))
            gauss = np.random.normal(0, 1, (pern, z_0.shape[1]))
            dispop[:pern] = gauss * sita + z_0.numpy()
            dispop = torch.from_numpy(dispop)
            dismol, dissmiles = model.decode(dispop)
            disfits = fitness_plogp(dismol, fp_0)  # 适应度计算
            disfits[np.isnan(disfits)] = 20
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
            fits_pops = fitness_plogp(mol, fp_0)  # 适应度计算
            fits_pops[np.isnan(fits_pops)] = 20

            # Generate consistent reference points and randomly initialize populations
            Z, N = uniformpoint(nPop, M)
            Zmin = np.array(np.min(fits_pops, 0)).reshape(1, M)
            mixpop = copy.deepcopy(np.vstack((dispop, pops_smi_val)))
            mixpopfun = copy.deepcopy(np.vstack((disfits, fits_pops)))
            pops, fits = envselect(mixpop, mixpopfun, N, Z, Zmin, M, D)

            iter = 1
            while iter <= nIter:
                # 进度条
                print("【进度】【{0:20s}】【正在进行{1}代...】【共{2}代】". \
                      format('▋' * int(iter / nIter * 20), iter, nIter), end='\r')
                #交叉变异
                #print(fits)
                chrpops = crossover(pops, pc, d, lb, rb)#混合线性交叉
                chrpops = mutate(chrpops, pm, nChr,m) # 变异产生子种群
                # 解码子代种群分子
                chrpops = torch.from_numpy(chrpops)
                chrmol, chrsmiles = model.decode(chrpops)  ###解码潜在表示，分子和SMILES
                chrfits = fitness_plogp(chrmol, fp_0)
                chrfits[np.isnan(chrfits)] = 20
                # 从原始种群和子种群中筛选nPop个
                Zmin = np.array(np.min(np.vstack((Zmin, chrfits)), 0)).reshape(1, M)  # 更新理想点
                mixpop = copy.deepcopy(np.vstack((pops, chrpops)))
                mixpopfun = copy.deepcopy(np.vstack((fits, chrfits)))

                pops, fits = envselect(mixpop, mixpopfun, N, Z, Zmin, M, D)
                iter += 1
                print(iter)
            # 解码最终种群分子
            print('enditer')
            pops = torch.from_numpy(pops)
            endmol, endsmiles = model.decode(pops)
            endsmiles = np.array(endsmiles)
            endfits = fits
            #保存
            t3 = time.time()  # 解码后时间
            imp_plogp = 0
            imp_plogp6 = 0
            # rr = []
            unique_smiles = []  # 储存当前分子唯一的SMILES,可放在重启前
            for i in range(len(endsmiles)):
                # rr.append((endfits[:][i] <= [fits_0[0] - 3, -0.4]).all())
                if endsmiles[i] not in unique_smiles:
                    unique_smiles.append(endsmiles[i])
                    improve = fits_0[0] - endfits[i][0]
                    tuple = (endsmiles[i], nn, -endfits[i][0], -endfits[i][1], improve)
                    smi_pro_tuple.append(tuple)
                    if -endfits[i][1] >= 0.4:
                        if improve >= imp_plogp:
                            imp_plogp = improve
                            smi_sr_4.append(tuple)
                    if -endfits[i][1] >= 0.6:
                        if improve >= imp_plogp6:
                            imp_plogp6 = improve
            print('save')
            if imp_plogp != 0:
                SR = SR + 1
                improvement.append(imp_plogp)
            if imp_plogp6 != 0:
                SR6 = SR6 + 1
                improvement6.append(imp_plogp6)
            df_to_save_A_to_B = pd.DataFrame(smi_pro_tuple, columns=['SMILES', 'mol_id', 'plogp', 'sim', 'improve'])
            df_to_save_A_to_B.to_csv('MOMO-plogp-mol50_nsga3.csv', index=False)
            df_to_save = pd.DataFrame(smi_sr_4, columns=['SMILES', 'mol_id', 'plogp', 'sim', 'improve'])
            df_to_save.to_csv('MOMO-plogp-mol50_nsga3_SR.csv', index=False)
            result = [SR, np.mean(improvement), SR6, np.mean(improvement6)]
            np.savetxt('MOMO-plogp-mol50_nsga3.txt', result, fmt='%s')
            time_one = t3 - t2  # 扰动个数
            print(time_one)
    t4 = time.time()  # 解码进化总时间
    time_all = t4 - t1
    print(time_all)



