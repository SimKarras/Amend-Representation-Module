# Learning to Amend Facial Expression Representation via De-albino and Affinity
                                                Jiawei Shi and Songhao Zhu
                                     Nanjing University of Posts and Telecommunications
                                                    Nanjing, China
                                             {1319055608, zhush}@njupt.edu.cn

## Amend-Representation-Module

![image](https://github.com/sunmusik/Amend-Representation-Module/blob/master/imgs/Net.png)

## Abstract
  Facial Expression Recognition (FER) is a classification task that points to face variants. Hence, there are certain
intimate relationships between facial expressions. We call them affinity features, which are barely taken into account
by current FER algorithms. Besides, to capture the edge information of the image, Convolutional Neural Networks
(CNNs) generally utilize a host of edge paddings. Although they are desirable, the feature map is deeply eroded after
multi-layer convolution. We name what has formed in this process the albino features, which definitely weaken the representation
of the expression. To tackle these challenges, we propose a novel architecture named Amend Representation
Module (ARM). ARM is a substitute for the pooling layer. Theoretically, it could be embedded in any CNN
with a pooling layer. ARM efficiently enhances facial expression representation from two different directions: 1) reducing
the weight of eroded features to offset the side effect of padding, and 2) sharing affinity features over minibatch
to strengthen the representation learning. In terms of data imbalance, we designed a minimal random resampling
(MRR) scheme to suppress network overfitting. Experiments on public benchmarks prove that our ARM boosts the
performance of FER remarkably. The validation accuracies are respectively 90.55% on RAF-DB, 64.49% on Affect-Net,
and 71.38% on FER2013, exceeding current state-of-theart methods.

## Training
```
python train_raf-db.py
```
