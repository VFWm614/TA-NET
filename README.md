# TA-NET
This repo is the implementation of "[TA-NET: Empowering Highly Efficient Traffic Anomaly Detection through Multi-Head Local Self-Attention and Adaptive Hierarchical Feature Reconstruction](https://ieeexplore.ieee.org/document/10457982)". By [Junzhou Chen](https://scholar.google.com/citations?user=Uz0U5UIAAAAJ&hl=zh-CN&oi=ao), Jiajun Pu, Baiqiao Yin, [Ronghui Zhang](https://ise.sysu.edu.cn/teacher/teacher02/1362395.htm), [Jun Jie Wu](https://www.durham.ac.uk/staff/junjie-wu/)

# Datasets and extracted features
**The original video data for the TAD dataset and the UCF-crime dataset can be obtained from the following links:**
> [**TAD dataset**](https://github.com/ktr-hubrt/WSAL)
> 
> [**UCF-crime dataset**](https://www.crcv.ucf.edu/projects/real-world/)

Since our TA-NET focuses on traffic anomaly detection, we constructed a subset of the UCF-crime dataset containing only traffic scenes, named UCF-crime-traffic. The file partitioning for UCF-crime-traffic can be obtained from [**UCF-crime-traffic index**](https://github.com/VFWm614/TA-NET/tree/4485be41ba86014173001b57a49701de40e6f27e/UCF-crime-traffic%20index) in this project:


**Additionally, we also provide video features extracted by [UniformerV2](https://github.com/OpenGVLab/UniFormerV2):**

> [**TAD features**](https://1drv.ms/u/s!AjBCIlm4rsXenUCWS5MGu4H4qpBn?e=heWYaB)
> 
> [**UCF-crime-traffic features**](https://1drv.ms/u/s!AjBCIlm4rsXenULO0CPe0fQuoZof?e=zVTkiA) 

Note: Due to the limitations of our device, only the first 10 min features of training videos were extracted if it longer than 10 min.

# Training
Replace the path of dataset features and training sets in `train_config.yaml` with yours.

Then, simply run the following commands:

```
python Code/train.py
```

# Testing
Please find the model weights in the following:
> [**pretrained model**](https://1drv.ms/f/s!AjBCIlm4rsXenUN507nR1chiacwT?e=A7C6bJ)

Then, replace the paths of dataset features and model weight in `test_config.yaml` with yours.

After the setup, simply run the following commands:

```
python Code/test.py
```

# Citation
If you find this repo useful for your research, please consider citing our paper:

```
@ARTICLE{10457982,
  author={Chen, Junzhou and Pu, Jiajun and Yin, Baiqiao and Zhang, Ronghui and Wu, Jun Jie},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={TA-NET: Empowering Highly Efficient Traffic Anomaly Detection Through Multi-Head Local Self-Attention and Adaptive Hierarchical Feature Reconstruction}, 
  year={2024},
  volume={},
  number={},
  pages={1-13},
  keywords={Feature extraction;Anomaly detection;Task analysis;Hidden Markov models;Supervised learning;Benchmark testing;Training;Feature extraction;anomaly detection;traffic anomaly detection;weakly supervised learning;multi-instance learning;transformer},
  doi={10.1109/TITS.2024.3365820}}
```
