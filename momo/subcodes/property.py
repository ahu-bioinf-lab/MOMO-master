# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 12:45:28 2022

@author: 86136
"""

from functools import lru_cache

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

'''
drd2_model = drd2_model()
def cal_DRD2(molecule_SMILES):
    return drd2_model(molecule_SMILES)
'''
def cal_SA(mol):
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

def normalize_sa(smiles):
    try:
        sa_score = sa_(smiles)
        normalized_sa = (10. - sa_score) / 9.
        return normalized_sa
    except:
        return np.nan

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


smi_ori = 'CC(C)C(=O)NCC[NH2+][C@@H](C)C[C@@H]1CCCCC[NH2+]1'
mol_ori = Chem.MolFromSmiles(smi_ori)
fp = morgan_fingerprint(mol_ori)
smi='CC(C)C(=O)NCCC1CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC1'
smi2 = 'CC(C)C(=O)NCCC1CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC1'

print(qed_(smi))
#print('logp',logp_(smi))
mol_0 = Chem.MolFromSmiles(smi)
mol_2 = Chem.MolFromSmiles(smi2)
print('plogp_1',penalized_logP(mol_0))
print('plogp_2',penalized_logP(mol_2))
sim_1 = tanimoto_similarity(mol_0,fp)
print('smi_1',sim_1)
sim_2 = tanimoto_similarity(mol_2,fp)
print('smi_2',sim_2)

print('drd_1',drd2_(smi))
print('drd_2',drd2_(smi2))
'''
# print(logp_(smi2))
# mol_2 = Chem.MolFromSmiles(smi2)
# print(penalized_logP(mol_2))

print(jnk(smi))
print(gsk(smi))
print(drd2(smi))

print('nom_sa:', normalize_sa(smi))
print('sa:', sa_(smi))

print('sa',SA(Chem.MolFromSmiles(smi)))


print('nor_sa3',normalize_sa(smi))
'''
