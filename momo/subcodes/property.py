# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 12:45:28 2022

@author: 86136
"""

from functools import lru_cache
import torch
import numpy as np
import pandas as pd
import rdkit
from moses.metrics import QED as QED_
from moses.metrics import SA, logP
from rdkit.Chem import AllChem
#from DRD2.DRD2_predictor2 import *
#from drd.drd_model import *
from tdc import Oracle


from rdkit import Chem, DataStructs


def penalized_logP(mol):
    """Penalized logP.

    Computed as logP(mol) - SA(mol) as in JT-VAE.

    Args:
        mol (rdkit.Chem.rdchem.Mol): Molecule on which to compute the metric.

    Returns:
        float: Penalized logP or NaN if mol is None.
    """
    #mol = Chem.MolFromSmiles(mol)
    try:
        return logP(mol) - SA(mol)
    except:
        return np.nan


def QED(mol):
    """Drug like-ness measure.

    Args:
        mol (rdkit.Chem.rdchem.Mol): Molecule on which to compute the metric.

    Returns:
        float: QED or NaN if mol is None.
    """
    #mol = Chem.MolFromSmiles(mol)
    try:
        return QED_(mol)
    except:
        return np.nan
    
def morgan_fingerprint(mol):
    """Molecular fingerprint using Morgan algorithm.

    Uses ``radius=2, nBits=2048``.

    Args:
        mol (rdkit.Chem.rdchem.Mol): Molecule on which to compute the fingerprint.

    Returns:
        np.ndarray: Fingerprint vector.
    """
    #mol = Chem.MolFromSmiles(seq)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

def tanimoto_similarity(seq, fp_0):
    """Tanimoto similarity between two molecules.

    Args:
        mol (rdkit.Chem.rdchem.Mol): Molecule on which to compute the metric.
        fp_0 (array): Fingerprint vector of original molecule.

    Returns:
        float: Tanimoto similarity.
    """
    #mol = Chem.MolFromSmiles(seq)
    fp = morgan_fingerprint(seq)
    if fp is None:
        return 0
    return rdkit.DataStructs.TanimotoSimilarity(fp_0, fp)


def sim_2(mol1, mol2):
    """Tanimoto similarity between two molecules.

    Args:
        mol (rdkit.Chem.rdchem.Mol): Molecule on which to compute the metric.
        fp_0 (array): Fingerprint vector of original molecule.

    Returns:
        float: Tanimoto similarity.
    """
    #mol = Chem.MolFromSmiles(mol)
    fp_1 = morgan_fingerprint(mol1)
    fp_2 = morgan_fingerprint(mol2)
    if fp_1 is None:
        return 0
    return rdkit.DataStructs.TanimotoSimilarity(fp_1, fp_2)

'''
drd2_model = drd2_model()
def cal_DRD2(molecule_SMILES):
    return drd2_model(molecule_SMILES)
'''
def cal_SA(mol):
    #mol = Chem.MolFromSmiles(mol)
    try:
        return SA(mol)
    except:
        return np.nan


qed_ = Oracle('qed')
sa_ = Oracle('sa')
jnk_ = Oracle('JNK3')
gsk_ = Oracle('GSK3B')
logp_ = Oracle('logp')
drd2_ = Oracle('drd2')
mu = 2.230044
sigma = 0.6526308
def normalize_sa(smiles):
    sa_score = sa_(smiles)
    mod_score = np.maximum(sa_score, mu)
    return np.exp(-0.5 * np.power((mod_score - mu) / sigma, 2.))

def jnk(smi):
    """Drug like-ness measure.

    Args:
        mol (rdkit.Chem.rdchem.Mol): Molecule on which to compute the metric.

    Returns:
        float: QED or NaN if mol is None.
    """
    try:
        return jnk_(smi)
    except:
        return np.nan



def gsk(smi):
    """Drug like-ness measure.

    Args:
        mol (rdkit.Chem.rdchem.Mol): Molecule on which to compute the metric.

    Returns:
        float: QED or NaN if mol is None.
    """
    try:
        return gsk_(smi)
    except:
        return np.nan

def drd2(smi):
    """Drug like-ness measure.

    Args:
        mol (rdkit.Chem.rdchem.Mol): Molecule on which to compute the metric.

    Returns:
        float: QED or NaN if mol is None.
    """
    try:
        return drd2_(smi)
    except:
        return np.nan


'''
smi='COc1cccc(O[C@@H]2CC[C@H]([NH3+])C2)n1'
smi2 = 'COc1ccc(C(C)=O)c(OCC(=O)N2[C@@H](C)CCC[C@H]2C)c1'
print(qed_(smi))
print(logp_(smi))
mol_0 = Chem.MolFromSmiles(smi)
print(penalized_logP(mol_0))

print(logp_(smi2))
mol_2 = Chem.MolFromSmiles(smi2)
print(penalized_logP(mol_2))

print(jnk(smi))
print(gsk(smi))
print(drd2(smi))
'''









