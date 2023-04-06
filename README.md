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
pip install .
```
### Getting Started
For QED and Similarity optimization Task 1, please run python MOMO_task1.py
For Plogp and Similarity optimization Task 2, please run python MOMO_task2.py
For QED, Drd2 and Similarity optimization Task 3, please run python MOMO_task3.py

The output results of molecules are summarized in smi_pro_tuple, and further save in .csv file.



### Writing your own Objective Function
The fitness function can wrap any function that has following properties:
- Takes a RDKit mol object as input and returns a number as score.


## Citation

If you use this work, please cite the following:
```
@article{xia2022molecule,
  title={Molecule optimization via multi-objective evolutionary in implicit chemical space},
  author={Xia, Xin and Su, Yansen and Zheng, Chunhou and Zeng, Xiangxiang},
  journal={arXiv preprint arXiv:2212.08826},
  year={2022}
}
```
