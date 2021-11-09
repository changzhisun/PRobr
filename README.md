# PRobr
PyTorch code for our ACL 2021 findings paper:

[Probabilistic Graph Reasoning for Natural Proof Generation](https://arxiv.org/abs/2107.02418)



## Installation
This repository is tested on Python 3.8.3.  
You should install PRobr on a virtual environment. All dependencies can be installed as follows:
```
pip install -r requirements.txt
```

## Download Dataset
Download the dataset as follows:
```
bash scripts/download_data.sh
```

## Training PRobr
PRobr can be trained by running the following script:
```
bash train.sh
```
This will train PRobr on the ```depth-5``` dataset. Should you wish to train on any of depth-0, depth-1, etc, change the ```data_dir``` path in the script accordingly.  
You also can reproduce PRobr by modifying ```model``` in ```train.sh```.
The trained model folder will be saved inside ```output``` folder.

## Testing PRobr

The trained PRobr model can be tested by running the following script:
```
bash dev.sh
```
or
```
bash test.sh
```
This will output the all metrics (QA accuracy, Node accuracy, Edge accuracy, Proof accuracy and Full accuracy).

## Zero-shot Evaluation on Birds-Electricity
Run the above testing, inference and evaluation scripts to test the depth-5 trained PRobr model on the Birds-Electricity dataset by appropriately changing the ```data-dir``` path to ```data/birds-electricity``` in all the scripts and lines 187 and 188 in ```utils.py``` with ```test.jsonl``` and ```meta-test.jsonl```.


## Training PRobr on ParaRules dataset
Run the following scripts to train PRobr on the ParaRules dataset (following similar steps as before):
```
bash train_natlang.sh
bash dev_natlang.sh
bash test_natlang.sh
```


## Running Other Ablations
Ablation models from the paper can be run by uncommenting parts of the code (like choosing a particular depth). Please refer to the comments in [utils.py](./utils.py) for details.

## Trained Models
We also release our trained models on depth-5 dataset [here](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/acl2021/PRobr/probr-trained-on-d5.tgz). These contain the respective QA, node and edge predictions and you can reproduce the results from the paper by running the evaluation script.

## Visualizing Proofs
The script to visualize PRobr's proof graphs as pdfs is ```evaluation/print_graphs.py```. It takes the usual arguments (data directory, node and prediction files) along with a path to the directoty to save the graphs.

## Citation
```
@inproceedings{sun2020probabilistic,
  title={Probabilistic Graph Reasoning for Natural Proof Generation},
  author={Sun, Changzhi and Zhang, Xinbo and Chen, Jiangjie and Gan, Chun and Wu, Yuanbin and Chen, Jiaze and Zhou, Hao and Li, Lei},
  booktitle={ACL},
  year={2021}
}
```

## Acknowledgement
Our code is based on [PRover](https://github.com/swarnaHub/PRover). We wish to thank the authors of [PRover](https://github.com/swarnaHub/PRover) for providing the source code.
