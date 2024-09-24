# multi-objective molecule optimization (MOMO)

Implementation of the method proposed in the paper "Evolutionary multi-objective molecule optimization in implicit chemical space" by Xin Xia, Yansen Su, Yiping Liu, Chunhou Zheng, Xiangxiang Zeng.<sup>1</sup>

### Dependencies
- [cddd](https://github.com/jrwnter/cddd)
Download the pre-trained CDDD model using the bash script:
```
./download_default_model.sh
- python=3.6
  - rdkit
  - pytorch=1.4.0
  - cudatoolkit=10.0
  - tensorboardX
  - pip>=19.1,<20.3
  - pip:
    - molsets
    - cddd
```

### Installing
```
cd MOMO
pip install pytdc
pip install rdkit.
```
### Getting Started
- For QED and Similarity optimization Task 1, please run python MOMO_task1.py

- For Plogp and Similarity optimization Task 2, please run python MOMO_task2.py

- For QED, Drd2 and Similarity optimization Task 3, please run python MOMO_task3.py

- For QED, 4LDE and Similarity optimization Task 4, please run python MOMO_task4.py

- For optimization Task 5, please run python MOMO_task5.py

- For QED, GSK3b, SA and Similarity optimization Task 6, please run python MOMO_task6.py

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
