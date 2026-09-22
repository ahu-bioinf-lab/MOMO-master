# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 14:07:42 2022

@author: 86136
"""
from property import *
from nonDominationSort import *
import calc_no as dock
import torch
from tqdm import tqdm
from rdkit.Chem import Descriptors
"""
种群初始化 
"""
def initPops(seqs): 
    pops = model.encode(seqs)  
    return pops 
"""
选择算子 
"""
def select1(pool, pops, fits, ranks, distances, smiles):
    # 一对一锦标赛选择 
    # pool: 新生成的种群大小 
    nPop, nChr = pops.shape 
    nF = fits.shape[1] 
    newPops = np.zeros((pool, nChr)) 
    newFits = np.zeros((pool, nF))  
    newsmiles = [0]*pool
    indices = np.arange(nPop).tolist()
    i = 0 
    while i < pool: 
        idx1, idx2 = random.sample(indices, 2)  # 随机挑选两个个体 
        idx = compare(idx1, idx2, ranks, distances) 
        newPops[i] = pops[idx] 
        newFits[i] = fits[idx]
        newsmiles[i] = smiles[idx]
        i += 1 
    return newPops, newFits,  newsmiles

def select1_single(pool, pops, fits,fits_s, smiles):
    # 一对一锦标赛选择
    # pool: 新生成的种群大小
    nPop, nChr = pops.shape
    nF = fits.shape[1]
    newPops = np.zeros((pool, nChr))
    newFits = np.zeros((pool, nF))
    newsmiles = [0]*pool
    newFits_single = [0] * pool
    indices = np.arange(nPop).tolist()
    i = 0
    while i < pool:
        idx1, idx2 = random.sample(indices, 2)  # 随机挑选两个个体
        idx = compare_single(idx1, idx2, fits_s)
        newPops[i] = pops[idx]
        newFits[i] = fits[idx]
        newsmiles[i] = smiles[idx]
        newFits_single[i] = fits_s[idx]
        i += 1
    return newPops, newFits,newFits_single, newsmiles
def compare_single(idx1, idx2, fits_single):
    # return: 更优的 idx
    if fits_single[idx1] < fits_single[idx2]:
        idx = idx2
    elif fits_single[idx1] > fits_single[idx2]:
        idx = idx1
    else:
        idx = idx1
    return idx


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

def crossover_2(pops, pc,d, lb, rb):
    # 拷贝父代种群，以防止改变父代种群结构  ,d=0.25参数
    nPop = pops.shape[0]
    chrPops = np.zeros((nPop * 2, pops.shape[1]))
    for i in range(0, nPop):
        if np.random.rand() < pc:
            mother1 = pops[np.random.randint(nPop)]
            mother2 = pops[np.random.randint(nPop)]
            [alpha1, alpha2] = np.random.rand(2)  # 生成2个随机数#,[alpha1,alpha2]=np.random.rand(2)
            r1 = (-d) + (1 + 2 * d) * alpha1
            r2 = (-d) + (1 + 2 * d) * alpha2
            chrPops[2 * i] = pops[i]+r1*(mother1-pops[i])  # 混合线性交叉
            chrPops[2 * i + 1] = pops[i]+r2*(mother2-pops[i])  # 混合线性交叉
        chrPops[chrPops < lb] = lb
        chrPops[chrPops > rb] = rb
    return chrPops

'''#离散片段交叉
def crossover(pops, pc, nChr, lb, rb):
    # 拷贝父代种群，以防止改变父代种群结构  ,d=0.25参数
    chrPops = pops.copy()
    nPop = chrPops.shape[0]
    for i in range(0, nPop):
        if np.random.rand() < pc:
            mother = chrPops[np.random.randint(nPop)]
            pos = np.random.randint(0, nChr-1, 1)
            chrPops[i][pos[0]:] = mother[pos[0]:]
            chrPops[i][chrPops[i]<lb] = lb
            chrPops[i][chrPops[i]>rb] = rb
    return chrPops

#模拟二进制交叉
def crossover_SBX(pops, pc, etaC, lb, rb):
    # 拷贝父代种群，以防止改变父代种群结构
    """
        :param pc: the probabilities of doing crossover
        :param etaC: the distribution index of simulated binary crossover，设20
        lb：下界
        rb：上界
    """
    chrPops = pops.copy()
    nPop = chrPops.shape[0]
    #for i in range(0, nPop, 2):
    #    if np.random.rand() < pc:
    #        SBX(chrPops[i], chrPops[i+1], etaC, lb, rb)  # 交叉
    #return chrPops
    for i in range(0, nPop):
        if np.random.rand() < pc:
            mother = chrPops[np.random.randint(nPop)]
            SBX(chrPops[i], mother, etaC, lb, rb)  # 交叉
    return chrPops

def SBX(chr1, chr2, etaC, lb, rb):

    # 模拟二进制交叉
    pos1, pos2 = np.sort(np.random.randint(0,len(chr1),2)) #随机产生两个位置
    pos2 += 1
    u = np.random.rand()
    if u <= 0.5:
        gamma = (2*u) ** (1/(etaC+1))
    else:
        gamma = (1/(2*(1-u))) ** (1/(etaC+1))
    x1 = chr1[pos1:pos2]
    x2 = chr2[pos1:pos2]
    chr1[pos1:pos2], chr2[pos1:pos2] = 0.5*((1+gamma)*x1+(1-gamma)*x2), \
        0.5*((1-gamma)*x1+(1+gamma)*x2)
    # 检查是否符合约束
    chr1[chr1<lb] = lb
    chr1[chr1>rb] = rb
    chr2[chr2<lb] = lb
    chr2[chr2<rb] = rb
'''
"""变异算子 
单点
"""
def mutate(pops, pm, nChr,m):
    nPop = pops.shape[0] 
    for i in range(nPop):
        if np.random.rand() < pm:
            zz=np.random.rand(m)
            #zzz = 4*zz-2#-2到2
            #pos = np.random.randint(0,nChr,1)
            pos = np.random.randint(0, nChr, m)#变异2个位置
            pops[i][pos] = zz
    return pops

'''
#多项式变异
def mutate_mutpol(pops, pm, etaM, lb, rb):
    """
            :param pm: the probabilities of doing mutate
            :param etaM: the distribution index of mutate，设20
            lb：下界
            rb：上界
    """
    nPop = pops.shape[0]
    for i in range(nPop):
        if np.random.rand() < pm:
            polyMutation(pops[i], etaM, lb, rb)
    return pops

def polyMutation(chr, etaM, lb, rb):
    # 多项式变异
    pos1, pos2 = np.sort(np.random.randint(0,len(chr),2))
    pos2 += 1
    u = np.random.rand()
    if u < 0.5:
        delta = (2*u) ** (1/(etaM+1)) - 1
    else:
        delta = 1-(2*(1-u)) ** (1/(etaM+1))
    chr[pos1:pos2] += delta
    chr[chr<lb] = lb
    chr[chr>rb] = rb
'''
"""扰动产生新的子代 
多点
"""
'''
def Disturb(pops, nChr,m,lb, rb):
    
    nPop = pops.shape[0]
    dis_pop = np.zeros((nPop, nChr))
    gauss = np.random.normal(0, 1, (nPop, nChr))
    dis_pop[:nPop] = gauss*0.5 + pops
    
    nPop = pops.shape[0]
    for i in range(nPop):
        pos = np.random.randint(0, nChr, m)  # 变异m个位置
        gauss = np.random.normal(0, 1, m)
        pops[i][pos] = pops[i][pos]+gauss#*0.5
        pops[i][pops[i] < lb] = lb
        pops[i][pops[i] > rb] = rb
    return pops
'''
"""
种群或个体的适应度 
"""
def fitness_qed(seq, mol,fp_0):
    # 计算种群或者个体的适应度，嵌入需解码为SMILES
    # 如果是1维需要转为2维 
    nPop = len(mol)
    fits = np.array([ff_qed(seq[i], mol[i], fp_0) for i in range(nPop)])
    return fits 

def ff_qed(seq, mol, fp_0):
    #pen_logP = penalized_logP(seq)#需改为种群
    qed = QED(mol)
    sim = tanimoto_similarity(mol, fp_0)
    return qed, sim#pen_logP,



def fitness_plogp(seq, mol,fp_0):
    # 计算种群或者个体的适应度，嵌入需解码为SMILES
    # 如果是1维需要转为2维
    nPop = len(mol)
    fits = np.array([ff_plogp(seq[i], mol[i], fp_0) for i in range(nPop)])
    return fits

def ff_plogp(seq, mol, fp_0):
    pen_logP = penalized_logP(mol)#需改为种群
    #qed = QED(seq)
    sim = tanimoto_similarity(mol, fp_0)
    return pen_logP, sim#pen_logP,

##gsk & sim
def fitness_gsk(seq, mol,fp_0):
    # 计算种群或者个体的适应度，嵌入需解码为SMILES
    # 如果是1维需要转为2维
    nPop = len(mol)
    fits = np.array([ff_gsk(seq[i], mol[i], fp_0) for i in range(nPop)])
    return fits

def ff_gsk(seq, mol, fp_0):
    #pen_logP = penalized_logP(seq)#需改为种群
    gskb = gsk(seq)
    sim = tanimoto_similarity(mol, fp_0)
    return gskb, sim#pen_logP,

##drd2 & sim
def fitness_drd(seq, mol,fp_0):
    # 计算种群或者个体的适应度，嵌入需解码为SMILES
    # 如果是1维需要转为2维
    nPop = len(mol)
    fits = np.array([ff_drd(seq[i], mol[i], fp_0) for i in range(nPop)])
    return fits

def ff_drd(seq, mol, fp_0):
    #pen_logP = penalized_logP(seq)#需改为种群
    drd = drd2(seq)
    sim = tanimoto_similarity(mol, fp_0)
    return drd, sim#pen_logP,

#docking
def fitness_4lde(seq, mol,fp_0):
    # 计算种群或者个体的适应度，嵌入需解码为SMILES
    # 如果是1维需要转为2维
    nPop = len(mol)

    fits = np.array([ff_4lde(seq[i], mol[i], fp_0) for i in range(nPop)])
    return fits

def ff_4lde(seq, mol, fp_0):
    #pen_logP = penalized_logP(seq)#需改为种群
    qed = QED(mol)
    sim = tanimoto_similarity(mol, fp_0)
    lde4 = -cal_4lde(seq)
    return qed, sim, lde4#pen_logP,

def cal_4lde(seq):
    mol = Chem.MolFromSmiles(seq)
    if mol is None:
        return 10 ** 4
    else:
        lde4 = dock.perform_calc_single(seq, '4lde', docking_program='qvina')
        return lde4

#docking+LE
def fitness_le_1syh(seq, mol,fp_0):
    # 计算种群或者个体的适应度，嵌入需解码为SMILES
    # 如果是1维需要转为2维
    nPop = len(mol)
    fits = np.array([ff_le_1syh(seq[i], mol[i], fp_0) for i in range(nPop)])
    return fits

def ff_le_1syh(seq, mol, fp_0):
    sim = tanimoto_similarity(mol, fp_0)
    syh1 = -cal_4lde(seq)
    print('1syh',syh1)
    try:
        HA = mol.GetNumHeavyAtoms()
    except:
        HA = 100
    print('HA', HA)  # 30-60
    if HA < 5:
        HA = 100
    LE = syh1 / HA
    print('LE', LE)
    return sim, LE, syh1, HA#pen_logP,

def cal_1syh(seq):
    mol = Chem.MolFromSmiles(seq)
    if mol is None:
        return 10 ** 4
    else:
        syh1 = dock.perform_calc_single(seq, '1syh', docking_program='qvina')
        return syh1

##多目标QED,SIM,SA
def fitness_qedsa(seq, mol,fp_0):
    # 计算种群或者个体的适应度，嵌入需解码为SMILES
    # 如果是1维需要转为2维
    nPop = len(mol)
    fits = np.array([ff_qedsa(seq[i], mol[i], fp_0) for i in range(nPop)])
    return fits

def ff_qedsa(seq, mol, fp_0):
    #pen_logP = penalized_logP(seq)#需改为种群
    qed = QED(mol)
    sa = 10-cal_SA(mol)
    sim = tanimoto_similarity(mol, fp_0)
    return qed, sa, sim#pen_logP,

##多目标QED,SIM,Plogp
def fitness_qedlogp(seq, mol,fp_0):
    # 计算种群或者个体的适应度，嵌入需解码为SMILES
    # 如果是1维需要转为2维
    nPop = len(mol)
    fits = np.array([ff_qedlogp(seq[i], mol[i], fp_0) for i in range(nPop)])

    return fits

def ff_qedlogp(seq, mol, fp_0):
    #pen_logP = penalized_logP(seq)#需改为种群
    qed = QED(mol)
    pen_logP = penalized_logP(mol)
    sim = tanimoto_similarity(mol, fp_0)
    return qed, pen_logP, sim#pen_logP,


#多目标qed,drd2,sim
def fitness_qeddrd(seqs,mols,fp_0):
    # 计算种群或者个体的适应度，嵌入需解码为SMILES
    # 如果是1维需要转为2维
    nPop = len(seqs)
    fits = np.array([ff_qeddrd(seqs[i], mols[i], fp_0) for i in range(nPop)])
    return fits

def ff_qeddrd(seq,mol,fp_0):
    #pen_logP = penalized_logP(seq)#需改为种群
    qed = QED(mol)  # 需改为种群
    drd = drd2(seq)
    #sa_nom = normalize_sa(seq)
    sim = tanimoto_similarity(mol, fp_0)
    return qed, sim, drd  # pen_logP

#多目标qed,jnk3,sa_nom,sim
def fitness_qedjnksa(seqs,mols,fp_0):
    # 计算种群或者个体的适应度，嵌入需解码为SMILES
    # 如果是1维需要转为2维
    nPop = len(seqs)
    fits = np.array([ff_qedjnksa(seqs[i], mols[i], fp_0) for i in range(nPop)])
    return fits

def ff_qedjnksa(seq,mol,fp_0):
    #pen_logP = penalized_logP(seq)#需改为种群
    #print('seq:', seq)
    qed = QED(mol)  # 需改为种群
    jnk3 = jnk(seq)
    sa_nom = normalize_sa(seq)
    sim = tanimoto_similarity(mol, fp_0)
    return qed, jnk3, sa_nom, sim  # pen_logP

#多目标qed,gskb,sa_nom,sim
def fitness_qedgsksa(seqs,mols,fp_0):
    # 计算种群或者个体的适应度，嵌入需解码为SMILES
    # 如果是1维需要转为2维
    nPop = len(seqs)
    fits = np.array([ff_qedgsksa(seqs[i], mols[i], fp_0) for i in range(nPop)])
    return fits

def ff_qedgsksa(seq,mol,fp_0):
    #pen_logP = penalized_logP(seq)#需改为种群
    qed = QED(mol)  # 需改为种群
    gskb = gsk(seq)
    sa_nom = normalize_sa(seq)
    sim = tanimoto_similarity(mol, fp_0)
    return qed, sim, gskb, sa_nom  # pen_logP

#多目标qed,drd2,sa_nom,sim
def fitness_qeddrdsa(seqs,mols,fp_0):
    # 计算种群或者个体的适应度，嵌入需解码为SMILES
    # 如果是1维需要转为2维
    nPop = len(seqs)
    fits = np.array([ff_qeddrdsa(seqs[i], mols[i], fp_0) for i in range(nPop)])
    return fits

def ff_qeddrdsa(seq,mol,fp_0):
    #pen_logP = penalized_logP(seq)#需改为种群
    qed = QED(mol)  # 需改为种群
    drd = drd2(seq)
    sa_nom = normalize_sa(seq)
    sim = tanimoto_similarity(mol, fp_0)
    return qed, drd, sa_nom, sim  # pen_logP


"""
种群的合并和优选 
"""
'''
def optSelect(pops, fits, chrPops, chrFits):
    """种群合并与优选 
    Return: 
        newPops, newFits 
    """
    nPop, nChr = pops.shape 
    nF = fits.shape[1] 
    newPops = np.zeros((nPop, nChr)) 
    newFits = np.zeros((nPop, nF)) 
    # 合并父代种群和子代种群构成一个新种群 
    MergePops = np.concatenate((pops,chrPops), axis=0) 
    MergeFits = np.concatenate((fits,chrFits), axis=0) 
    MergeRanks = nonDominationSort(MergePops, MergeFits) 
    MergeDistances = crowdingDistanceSort(MergePops, MergeFits, MergeRanks) 

    indices = np.arange(MergePops.shape[0]) 
    r = 0 
    i = 0 
    rIndices = indices[MergeRanks==r]  # 当前等级为r的索引 
    while i + len(rIndices)  <= nPop:
        newPops[i:i+len(rIndices)] = MergePops[rIndices] 
        newFits[i:i+len(rIndices)] = MergeFits[rIndices] 
        r += 1  # 当前等级+1 
        i += len(rIndices) 
        rIndices = indices[MergeRanks==r]  # 当前等级为r的索引 
    
    if i < nPop: 
        rDistances = MergeDistances[rIndices]   # 当前等级个体的拥挤度 
        rSortedIdx = np.argsort(rDistances)[::-1]  # 按照距离排序 由大到小 
        surIndices = rIndices[rSortedIdx[:(nPop-i)]]  
        newPops[i:] = MergePops[surIndices] 
        newFits[i:] = MergeFits[surIndices] 
    return (newPops, newFits)
'''
#种群的CV需更新
def optSelect_uni(pops, fits, chrPops, chrFits, nPop, smiles, chrsmiles):
    """种群合并与优选
    Return:
        newPops, newFits
    """
    nChr = pops.shape[1]
    nF = fits.shape[1]
    newPops = np.zeros((nPop, nChr))
    newFits = np.zeros((nPop, nF))
    newsmiles = [0] * nPop
    # 合并父代种群和子代种群构成一个新种群
    MergePops = np.concatenate((pops, chrPops), axis=0)
    MergeFits = np.concatenate((fits, chrFits), axis=0)
    Mergesmiles = np.concatenate((smiles, chrsmiles), axis=0)
    #首先去除重复
    MergeFits_uni, indices = np.unique(MergeFits, axis=0, return_index=True)
    MergePops_uni = MergePops[indices]
    Mergesmiles_uni = Mergesmiles[indices]
    #print('MergeFits_uni',MergeFits_uni)
    #print('MergeFits_uni_cal', MergeFits_uni[:, 0:2])
    MergeRanks = nonDominationSort(MergePops_uni, MergeFits_uni[:, 0:2])
    MergeDistances = crowdingDistanceSort(MergePops_uni, MergeFits_uni[:, 0:2], MergeRanks)

    indices = np.arange(MergePops_uni.shape[0])
    r = 0
    i = 0
    rIndices = indices[MergeRanks == r]  # 当前等级为r的索引
    while i + len(rIndices) < nPop:
        newPops[i:i + len(rIndices)] = MergePops_uni[rIndices]
        newFits[i:i + len(rIndices)] = MergeFits_uni[rIndices]
        newsmiles[i:i + len(rIndices)] = Mergesmiles_uni[rIndices]
        r += 1  # 当前等级+1
        i += len(rIndices)
        rIndices = indices[MergeRanks == r]  # 当前等级为r的索引
        # 若加到最后一个等级仍不足种群数，随机采分子
        # 不够的分子采样补全
        if r == max(MergeRanks) + 1:
            IID = indices.tolist()
            for j in range(i, nPop):
                idx1, idx2 = random.sample(IID, 2)  # 随机挑选两个个体
                idx = compare(idx1, idx2, MergeRanks, MergeDistances)
                newPops[j] = MergePops_uni[idx]
                newFits[j] = MergeFits_uni[idx]
                newsmiles[j] = Mergesmiles_uni[idx]
                j += 1
                i += 1

    if i < nPop:
        rDistances = MergeDistances[rIndices]  # 当前等级个体的拥挤度
        rSortedIdx = np.argsort(rDistances)[::-1]  # 按照距离排序 由大到小
        surIndices = rIndices[rSortedIdx[:(nPop - i)]]
        newPops[i:] = MergePops_uni[surIndices]
        newFits[i:] = MergeFits_uni[surIndices]
        newsmiles[i:] = Mergesmiles_uni[surIndices]
    return (newPops, newFits, newsmiles)


def optSelect_single(pops, fits,fits_s, chrPops, chrFits,chrFits_s, nPop, smiles, chrsmiles):
    # 合并父代种群和子代种群构成一个新种群
    MergePops = np.concatenate((pops, chrPops), axis=0)
    MergeFits = np.concatenate((fits, chrFits), axis=0)
    MergeFits_s = np.concatenate((fits_s, chrFits_s), axis=0)
    Mergesmiles = np.concatenate((smiles, chrsmiles), axis=0)
    # 首先去除重复
    MergeFits_uni, indices = np.unique(MergeFits, axis=0, return_index=True)
    MergeFits_s_uni = MergeFits_s[indices]
    MergePops_uni = MergePops[indices]
    Mergesmiles_uni = Mergesmiles[indices]

    #从大到小排序
    MergeFits_s_rev = [fit*(-1) for fit in MergeFits_s_uni]
    sort_f = np.argsort(MergeFits_s_rev)

    pops_sort = np.array([MergePops_uni[s] for s in sort_f][:nPop])
    fits_sort = np.array([MergeFits_uni[s] for s in sort_f][:nPop])
    fits_single_sort = [MergeFits_s_uni[s] for s in sort_f][:nPop]
    smiles_sort = [Mergesmiles_uni[s] for s in sort_f][:nPop]

    return pops_sort, fits_sort, fits_single_sort, smiles_sort
'''
def optSelect_id(pops, fits, chrPops, chrFits):
    """种群合并与优选
    Return:
        newPops, newFits
    """
    nPop, nChr = pops.shape
    nF = fits.shape[1]
    newPops = np.zeros((nPop, nChr))
    newFits = np.zeros((nPop, nF))
    # 合并父代种群和子代种群构成一个新种群
    MergePops = np.concatenate((pops, chrPops), axis=0)
    MergeFits = np.concatenate((fits, chrFits), axis=0)
    MergeRanks = nonDominationSort(MergePops, MergeFits)
    MergeDistances = crowdingDistanceSort(MergePops, MergeFits, MergeRanks)
    optse_id = []
    indices = np.arange(MergePops.shape[0])
    r = 0
    i = 0
    rIndices = indices[MergeRanks == r]  # 当前等级为r的索引
    while i + len(rIndices) <= nPop:
        newPops[i:i + len(rIndices)] = MergePops[rIndices]
        newFits[i:i + len(rIndices)] = MergeFits[rIndices]
        for j in rIndices:
            optse_id.append(j)
        r += 1  # 当前等级+1
        i += len(rIndices)
        rIndices = indices[MergeRanks == r]  # 当前等级为r的索引

    if i < nPop:
        rDistances = MergeDistances[rIndices]  # 当前等级个体的拥挤度
        rSortedIdx = np.argsort(rDistances)[::-1]  # 按照距离排序 由大到小
        surIndices = rIndices[rSortedIdx[:(nPop - i)]]
        newPops[i:] = MergePops[surIndices]
        newFits[i:] = MergeFits[surIndices]
        for s in surIndices:
            optse_id.append(s)
    return newPops, newFits, optse_id

#接收概率，按照NSGA2排序，分子仍有一定概率不被接收
def optSelect_ap(pops, fits, chrPops, chrFits, ap):
    """种群合并与优选
    Return:
        newPops, newFits
    """
    nPop, nChr = pops.shape
    nF = fits.shape[1]
    newPops = np.zeros((nPop, nChr))
    newFits = np.zeros((nPop, nF))
    # 合并父代种群和子代种群构成一个新种群
    MergePops = np.concatenate((pops, chrPops), axis=0)
    MergeFits = np.concatenate((fits, chrFits), axis=0)
    MergeRanks = nonDominationSort(MergePops, MergeFits)
    MergeDistances = crowdingDistanceSort(MergePops, MergeFits, MergeRanks)

    indices = np.arange(MergePops.shape[0])
    r = 0
    i = 0
    rIndices = indices[MergeRanks == r]  # 当前等级为r的索引
    apIndices = rIndices[np.random.rand(len(rIndices)) <= ap]  # 当前等级中通过概率筛选的索引

    while i + len(apIndices) <= nPop:
        newPops[i:i + len(apIndices)] = MergePops[apIndices]#添加满足概率的当前等级的解
        newFits[i:i + len(apIndices)] = MergeFits[apIndices]
        r += 1  # 当前等级+1
        i += len(apIndices)
        rIndices = indices[MergeRanks == r]  # 当前等级为r的索引
        apIndices = rIndices[np.random.rand(len(rIndices)) <= ap]  # 当前等级中通过概率筛选的索引

    if i < nPop:
        rDistances = MergeDistances[apIndices]  # 当前等级个体的拥挤度
        rSortedIdx = np.argsort(rDistances)[::-1]  # 按照距离排序 由大到小
        surIndices = rIndices[rSortedIdx[:(nPop - i)]]
        newPops[i:] = MergePops[surIndices]
        newFits[i:] = MergeFits[surIndices]
    return (newPops, newFits)
'''

'''
smi = 'CCCCCC1CCC(CCCCCCCCNCc2ccc([O-])c[nH+]2)CC1'
mol = Chem.MolFromSmiles(smi)
print(Descriptors.MolWt(mol))
'''
