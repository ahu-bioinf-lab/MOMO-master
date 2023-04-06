# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 14:07:42 2022

@author: 86136
"""
from property import *
from nonDominationSort import *
import torch
from tqdm import tqdm
from numpy import dot
from numpy.linalg import norm
from rdkit.Chem import Descriptors
"""
种群初始化 
"""
#def initPops(seqs):
#    pops = model.encode(seqs)
#    return pops
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
    chrPops = pops
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

def crossover_z0(z_0, archive1_emb, pc,d, lb, rb):
    # 拷贝父代种群，以防止改变父代种群结构  ,d=0.25参数
    chrPops = archive1_emb
    nPop = chrPops.shape[0]
    for i in range(0, nPop):
        if np.random.rand() < pc:
            mother = z_0
            alpha1=np.random.rand()#生成2个随机数#,[alpha1,alpha2]=np.random.rand(2)
            r1=(-d)+(1+2*d)*alpha1
            #r2=(-d)+(1+2*d)*alpha2
            chrPops[i] = chrPops[i]+r1*(mother-chrPops[i])#混合线性交叉
            chrPops[i][chrPops[i]<lb] = lb
            chrPops[i][chrPops[i]>rb] = rb
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
'''
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

"""变异算子 
单点
"""
def mutate(pops, pm, nChr,m):
    nPop = pops.shape[0] 
    for i in range(nPop):
        if np.random.rand() < pm:
            zz=np.random.rand(m)
            zz = torch.tensor(zz).to(torch.float32)
            #zzz = 4*zz-2#-2到2
            #pos = np.random.randint(0,nChr,1)
            pos = np.random.randint(0, nChr, m)#变异2个位置
            pops[i][pos] = zz
    return pops


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

"""扰动产生新的子代 
多点
"""
def Disturb(pops, nChr,m,lb, rb):
    '''
    nPop = pops.shape[0]
    dis_pop = np.zeros((nPop, nChr))
    gauss = np.random.normal(0, 1, (nPop, nChr))
    dis_pop[:nPop] = gauss*0.5 + pops
    '''
    nPop = pops.shape[0]
    for i in range(nPop):
        pos = np.random.randint(0, nChr, m)  # 变异m个位置
        gauss = np.random.normal(0, 1, m)
        pops[i][pos] = pops[i][pos]+gauss#*0.5
        pops[i][pops[i] < lb] = lb
        pops[i][pops[i] > rb] = rb
    return pops

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
    sim = tanimoto_similarity(seq, fp_0)
    return drd, sim#pen_logP,



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
    return qed, gskb, sa_nom, sim  # pen_logP

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

#计算余弦相似度
def cos(z_0,Z):
    sim_cos=[]
    zz = z_0.numpy().reshape(512)
    for i in range(Z.shape[0]):
        z_1 = Z[i].numpy().reshape(512)
        dot_sim = dot(zz, z_1) / (norm(zz) * norm(z_1))
        sim_cos.append(dot_sim)
    return sim_cos  # pen_logP
"""
种群的合并和优选 
"""
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


def optSelect_unique(pops, fits, chrPops, chrFits):
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
    while i + len(rIndices) <= nPop:
        #循环每个位置，判断是否已经存在
        for clock in rIndices:
            if newFits.__contains__(clock) == 0:
                newPops[i] = MergePops[clock]
                newFits[i] = MergeFits[clock]
                i += 1
        r += 1  # 当前等级+1
        rIndices = indices[MergeRanks == r]  # 当前等级为r的索引

    while i < nPop:
        rDistances = MergeDistances[rIndices]  # 当前等级个体的拥挤度
        rSortedIdx = np.argsort(rDistances)[::-1]  # 按照距离排序 由大到小
        for clock in rSortedIdx:
            if newFits.__contains__(clock) == 0:
                newPops[i] = MergePops[clock]
                newFits[i] = MergeFits[clock]
                i += 1
        r += 1  # 当前等级+1
        rIndices = indices[MergeRanks == r]  # 当前等级为r的索引
        if r==max(MergeRanks):
            rDistances = MergeDistances[rIndices]  # 当前等级个体的拥挤度
            rSortedIdx = np.argsort(rDistances)[::-1]  # 按照距离排序 由大到小
            surIndices = rIndices[rSortedIdx[:(nPop - i)]]
            newPops[i:] = MergePops[surIndices]
            newFits[i:] = MergeFits[surIndices]
            break
    return (newPops, newFits)


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

#无效不保存，返回无效id
def decode_from_jtvae(mols,opts, model):
    returned_smiles = []
    nonid = []
    tree_dims = int(opts.latent_size / 2)
    for i in tqdm(range(mols.shape[0])):
        try:
            tree_vec = np.expand_dims(mols[i, 0:tree_dims], 0)
            mol_vec = np.expand_dims(mols[i, tree_dims:], 0)
            tree_vec = torch.autograd.Variable(torch.from_numpy(tree_vec).float())
            mol_vec = torch.autograd.Variable(torch.from_numpy(mol_vec).float())
            smi = model.decode(tree_vec, mol_vec, prob_decode=False)
            if not smi:
                nonid.append(i)
            returned_smiles.append((i, smi))
        except:
           returned_smiles.append((i, 'None'))
           nonid.append(i)
    return returned_smiles, nonid



def decode_from_jtvae_limit(mols,opts, model):
    returned_smiles = []
    nonid = []
    chrmol = []
    tree_dims = int(opts.latent_size / 2)
    for i in tqdm(range(mols.shape[0])):
        try:
            tree_vec = np.expand_dims(mols[i, 0:tree_dims], 0)
            mol_vec = np.expand_dims(mols[i, tree_dims:], 0)
            tree_vec = torch.autograd.Variable(torch.from_numpy(tree_vec).float())
            mol_vec = torch.autograd.Variable(torch.from_numpy(mol_vec).float())
            smi = model.decode(tree_vec, mol_vec, prob_decode=False)
            if not smi:
                nonid.append(i)
            returned_smiles.append((i, smi))
            mol1 = Chem.MolFromSmiles(smi)
            if mol1 is not None:
                '''
                if (Descriptors.MolWt(mol1)>600) | (len(smi)>100):
                    nonid.append(i)
                else:
                    chrmol.append(mol1)
                '''
                chrmol.append(mol1)
            else:
                nonid.append(i)
        except:
           returned_smiles.append((i, 'None'))
           nonid.append(i)
    return returned_smiles, chrmol , nonid

'''
smi = 'CCCCCC1CCC(CCCCCCCCNCc2ccc([O-])c[nH+]2)CC1'
mol = Chem.MolFromSmiles(smi)
print(Descriptors.MolWt(mol))
'''
