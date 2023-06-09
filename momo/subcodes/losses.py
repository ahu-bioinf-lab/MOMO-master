from functools import lru_cache

import numpy as np
import pandas as pd
import rdkit
from moses.metrics import QED as QED_
from moses.metrics import SA, logP
from rdkit.Chem import AllChem
from tdc import Oracle
from sklearn.preprocessing import MinMaxScaler

def penalized_logP(mol):
    """Penalized logP.

    Computed as logP(mol) - SA(mol) as in JT-VAE.

    Args:
        mol (rdkit.Chem.rdchem.Mol): Molecule on which to compute the metric.

    Returns:
        float: Penalized logP or NaN if mol is None.
    """
    try:
        return logP(mol) - SA(mol)
    except:
        return np.nan

def loss_pen_logP_imp(mol, smiles, pen_logP_0, threshold=None, penalty=-1):
    """Penalized logP improvement loss.

    Args:
        mol (iterable or rdkit.Chem.rdchem.Mol): Batch of molecules on which to
            compute the loss.
        pen_logP_0 (float): Penalized logP of original molecule.
        threshold (float, optional): Threshold for constrained optimization or
            None for unconstrained. Note: for this metric **greater** value is
            better.
        penalty (float, optional): Value given to invalid molecules.

    Returns:
        array: Loss values.
    """
    if isinstance(mol, rdkit.Chem.rdchem.Mol):
        mol = [mol]
    improvement = np.array([penalized_logP(m) for m in mol]) - pen_logP_0
    # replace NaNs with penalty score
    improvement = np.nan_to_num(improvement, nan=penalty)
    #print('ori_improvement:', improvement)
    #print(improvement.shape[0])

    if improvement.shape[0]!=1:
        improvement = improvement.reshape(-1, 1)
        #print('reshape improvement:', improvement)
        improvement = MinMaxScaler().fit_transform(improvement)
        improvement = np.squeeze(improvement)
        #print('standrad improvement:', improvement)

    if threshold is None:
        return -improvement
    else:
        return np.maximum(threshold - improvement, 0)

def QED(mol):
    """Drug like-ness measure.

    Args:
        mol (rdkit.Chem.rdchem.Mol): Molecule on which to compute the metric.

    Returns:
        float: QED or NaN if mol is None.
    """
    try:
        return QED_(mol)
    except:
        return np.nan

def loss_QED(mol, smiles, threshold=None, penalty=0):
    """QED (drug-likeness) loss.

    Args:
        mol (iterable or rdkit.Chem.rdchem.Mol): Batch of molecules on which to
            compute the metric.
        threshold (float, optional): Threshold for constrained optimization or
            None for unconstrained. Note: for this metric **greater** value is
            better.
        penalty (float, optional): Value given to invalid molecules.

    Returns:
        array: Loss values.
    """
    if isinstance(mol, rdkit.Chem.rdchem.Mol):
        mol = [mol]
    qed = np.array([QED(m) for m in mol])
    # replace NaNs with penalty score
    qed = np.nan_to_num(qed, nan=penalty)

    if threshold is None:
        return -qed
    else:
        return np.maximum(threshold - qed, 0)

def morgan_fingerprint(mol):
    """Molecular fingerprint using Morgan algorithm.

    Uses ``radius=2, nBits=2048``.

    Args:
        mol (rdkit.Chem.rdchem.Mol): Molecule on which to compute the fingerprint.

    Returns:
        np.ndarray: Fingerprint vector.
    """
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

def tanimoto_similarity(mol, fp_0):
    """Tanimoto similarity between two molecules.

    Args:
        mol (rdkit.Chem.rdchem.Mol): Molecule on which to compute the metric.
        fp_0 (array): Fingerprint vector of original molecule.

    Returns:
        float: Tanimoto similarity.
    """
    fp = morgan_fingerprint(mol)
    if fp is None:
        return 0
    return rdkit.DataStructs.TanimotoSimilarity(fp_0, fp)

def loss_tanimoto(mol, smiles, fp_0, threshold=None):
    """Tanimoto similarity loss.

    Args:
        mol (iterable or rdkit.Chem.rdchem.Mol): Batch of molecules on which to
            compute the loss.
        fp_0 (array): Fingerprint vector of original molecule.
        threshold (float, optional): Threshold for constrained optimization or
            None for unconstrained. Note: for this metric **greater** value is
            better.

    Returns:
        array: Loss values.
    """
    if isinstance(mol, rdkit.Chem.rdchem.Mol):
        mol = [mol]
    sim = np.array([tanimoto_similarity(m, fp_0) for m in mol])
    if threshold is None:
        return -sim
    else:
        sim_tre = np.maximum(threshold - sim, 0)
        #print('sim_pre:', sim_tre)

        if sim_tre.shape[0] != 1:
            sim_tre = sim_tre.reshape(-1, 1)
            # print('reshape improvement:', improvement)
            sim_tre = MinMaxScaler().fit_transform(sim_tre)
            sim_tre = np.squeeze(sim_tre)
            #print('stand_sim_tre:', sim_tre)

        return sim_tre

qed_ = Oracle('qed')
sa_ = Oracle('sa')
jnk_ = Oracle('JNK3')
gsk_ = Oracle('GSK3B')
logp_ = Oracle('logp')
drd2_ = Oracle('drd2')

def normalize_sa(smiles):
    mu = 2.230044
    sigma = 0.6526308
    sa_score = sa_(smiles)
    mod_score = np.maximum(sa_score, mu)
    return np.exp(-0.5 * np.power((mod_score - mu) / sigma, 2.))
def loss_sa(mol, smiles, threshold=None, penalty=0):
    #if isinstance(mol, rdkit.Chem.rdchem.Mol):
       # mol = [mol]
    sa = np.array([normalize_sa(smi) for smi in smiles])
    # replace NaNs with penalty score
    sa = np.nan_to_num(sa, nan=penalty)
    if threshold is None:
        return -sa
    else:
        return np.maximum(threshold - sa, 0)

def jnk(smi):
    try:
        return jnk_(smi)
    except:
        return np.nan
def loss_jnk(mol, smiles, threshold=None, penalty=0):
    #if isinstance(mol, rdkit.Chem.rdchem.Mol):
       # mol = [mol]
    jnk3 = np.array([jnk(smi) for smi in smiles])
    # replace NaNs with penalty score
    jnk3 = np.nan_to_num(jnk3, nan=penalty)
    if threshold is None:
        return -jnk3
    else:
        return np.maximum(threshold - jnk3, 0)

def gsk(smi):
    try:
        return gsk_(smi)
    except:
        return np.nan
def loss_gsk(mol, smiles, threshold=None, penalty=0):
    #if isinstance(mol, rdkit.Chem.rdchem.Mol):
       # mol = [mol]
    gskb = np.array([gsk(smi) for smi in smiles])
    # replace NaNs with penalty score
    gskb = np.nan_to_num(gskb, nan=penalty)
    if threshold is None:
        return -gskb
    else:
        return np.maximum(threshold - gskb, 0)

def drd(smi):
    try:
        return drd2_(smi)
    except:
        return np.nan
def loss_drd2(mol, smiles, threshold=None, penalty=0):
    #if isinstance(mol, rdkit.Chem.rdchem.Mol):
       # mol = [mol]
    drd2 = np.array([drd(smi) for smi in smiles])
    # replace NaNs with penalty score
    drd2 = np.nan_to_num(drd2, nan=penalty)
    if threshold is None:
        return -drd2
    else:
        return np.maximum(threshold - drd2, 0)

def loss_function(z, model, ww, weight=1, score=None, constraints=[],
                  weight_constraint=False):
    """Multi-objective loss function.

    Note:
        ``scores`` and ``constraints`` expect an iterable of functions which
        take only a list of `:class:~rdkit.Chem.rdchem.Mol`. Users should use
        `:func:functools.partial` to fill the additional parameters as need.

    Args:
        z (torch.Tensor): Batch of latent vectors.
        model (model): Model to use to decode zs.
        weight (float, optional): Scalar weight to apply to property losses.
        score (iterable, optional): Loss function to use as the unconstrained
            "score" loss.
        constraints (iterable, optional): Iterable of loss functions to use as
            constrained losses.
        weight_constraint (bool, optional): If ``True``, multiply the constraint
            loss by the weight instead of the property loss.

    Returns:
        array: Total loss for each item in the batch.
    """
    mols, smiles = model.decode(z)

    loss_property = score(mols, smiles) if score else 0

    loss_constraint = 0
    for c in constraints:
        loss_constraint += c(mols, smiles)
    #print('ww:',ww)
    return (ww*loss_property + loss_constraint*weight if weight_constraint else
            loss_property*weight + loss_constraint)

def loss_function_mul(z, model, ww, weight=1, weight2=1,weight3=1,score=None, constraints=[],
                  weight_constraint=False):
    """Multi-objective loss function.

    Note:
        ``scores`` and ``constraints`` expect an iterable of functions which
        take only a list of `:class:~rdkit.Chem.rdchem.Mol`. Users should use
        `:func:functools.partial` to fill the additional parameters as need.

    Args:
        z (torch.Tensor): Batch of latent vectors.
        model (model): Model to use to decode zs.
        weight (float, optional): Scalar weight to apply to property losses.
        score (iterable, optional): Loss function to use as the unconstrained
            "score" loss.
        constraints (iterable, optional): Iterable of loss functions to use as
            constrained losses.
        weight_constraint (bool, optional): If ``True``, multiply the constraint
            loss by the weight instead of the property loss.

    Returns:
        array: Total loss for each item in the batch.
    """
    mols, smiles = model.decode(z)

    loss_property = score(mols, smiles) if score else 0
    #print('loss_property :', loss_property )
    loss_constraint = 0
    w = [weight, weight2, weight3]
    i=0
    for c in constraints:
        #print('weight', w[i])
        #print('con_1:', c(mols, smiles))
        loss_constraint = loss_constraint + w[i]*c(mols, smiles)
        #print('loss_constraint:', loss_constraint)
        i=i+1
    #print('loss_constraint：', loss_constraint)
    return (ww*loss_property + loss_constraint if weight_constraint else
            loss_property*weight + loss_constraint)