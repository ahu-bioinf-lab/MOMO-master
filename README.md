# multi-objective molecule optimization (MOMO)

Implementation of the method proposed in the paper "Evolutionary multi-objective molecule optimization in implicit chemical space" by Xin Xia, Yansen Su, Yiping Liu, Chunhou Zheng, Xiangxiang Zeng.<sup>1</sup>

### Dependencies
- [cddd](https://github.com/jrwnter/cddd)
  - Notice: You need download the pre-trained encoder-decoder CDDD model to mapping molecules between SMILES and continuous vectors. It can be load by the bash script:
```
./download_default_model.sh
```
The link is also provided on [cddd](https://drive.google.com/file/d/1ccJEclD1dxTNTUswUvIVygqRQSawUCnJ/view?usp=sharing). 
```

### Installing
- python=3.6
  - rdkit
  - pytorch=1.4.0
  - cudatoolkit=10.0
  - tensorboardX
  - pip>=19.1,<20.3
  - pytdc
  - pip:
    - molsets
    - cddd
- The installed environment can be downloaded from the cloud drive
  - [qmocddd](https://drive.google.com/file/d/1Wad0hxEfoqC5VzWGDPk9eBsFVkCi2o6Y/view?

### Data Description
- momo/data/qed_test.csv: dataset on Task1.
- momo/data/logp_test.csv: dataset on Task2.
- momo/data/qeddrd_test.csv: dataset on Task3.
- momo/data/docking_test.csv: dataset on Task4.
- momo/data/Gucamol_sample_800.csv: dataset on Task5.
- momo/data/gsk3_test.csv: dataset on Task6.
- momo/data/oripops_qed/: molecules for initial population on task1.
- momo/data/oripops_plogp/: molecules for initial population on task2.
- momo/data/oripops_qeddrd/: molecules for initial population on task3.

### File Description
- momo/sub_code/fitness.py: The script to calculate the objectives of optimization tasks.
- sub_code/property.py: The script to calculate the molecular properties.
- sub_code/generation_rule.py: The script to generate offspring molecules.
- sub_code/selection_rule.py: The script to compare and select molecules.
- sub_code/models.py: The encoder and decoder process.
- sub_code/calc_no.py: The script to calculate the docking scores.
- sub_code/mechanism.py: Guacamol tasks.
- sub_code/nonDominationSort.py: the non-dominated relationships between molecules.

- download_default_model.sh: download the pre-trained encoder-decoder.
- environment.yml: install the environment.
- MOMO_task1.py: optimization Task1. 
- MOMO_task2.py: optimization Task2. 
- MOMO_task3.py: optimization Task3. 
- MOMO_task4.py: optimization Task4.
- MOMO_task5.py: optimization Task5. 
- MOMO_task6.py: optimization Task6.

### Getting Started
- For QED and Similarity optimization Task 1, please run
'''
python MOMO_task1.py
'''
- For Plogp and Similarity optimization Task 2, please run
'''
python MOMO_task2.py
'''
- For QED, Drd2 and Similarity optimization Task 3, please run
'''
python MOMO_task3.py
'''
- For QED, 4LDE and Similarity optimization Task 4, please run
'''
python MOMO_task4.py
'''
- For optimization Task 5, please run 
'''
python MOMO_task5.py
'''
- For QED, GSK3b, SA and Similarity optimization Task 6, please run
'''
python MOMO_task6.py
'''

- The output results of molecules are summarized in smi_pro_tuple, and further save in .csv file.

### Writing your own Objective Function
The fitness function can wrap any function that has following properties:
- Takes a RDKit mol object as input and returns a number as score.
- Uses pyTDC platform to gets the properties such as QED, logp, Drd2, JNK3,...
- Uses docking platform such as qvina 2 to get the protein-ligand docking score.


## Citation
If you use this work, please cite the following:
```
@article{xia2024evolutionary,
  title={Evolutionary Multiobjective Molecule Optimization in an Implicit Chemical Space},
  author={Xia, Xin and Liu, Yiping and Zheng, Chunhou and Zhang, Xingyi and Wu, Qingwen and Gao, Xin and Zeng, Xiangxiang and Su, Yansen},
  journal={Journal of Chemical Information and Modeling},
  volume={64},
  number={13},
  pages={5161--5174},
  year={2024},
  publisher={ACS Publications}
}
```
