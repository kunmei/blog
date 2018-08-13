---
layout:     post
title:      scikit-learn用户指南
subtitle:   监督学习-线性判别分析
date:       2018-07-17
author:     john
catalog: true
tags:
    - 机器学习
---
### LDA和QDA
线性判别分析和二次判别分析是两个经典的分类器，正如其名称所示，分别表示线性或者二次决策表面。

这些分类器很有吸引力，因为它们具有封闭形式的解决方案，可以轻松计算，本质上是多类的，已经证明在实践中运行良好并且没有超参数可以调整。

#### 使用线性判别分析进行降维
线性判别分析可用于执行有监督的降维，通过将输入数据投影到线性子空间，该线性子空间由最大化类之间分离的方向组成。输出的维度必然小于类的数量，因此这相当于一个相当强的维度减少，并且只在多类设置中有意义。

关于LDA和QDA的使用可以参见[notebook](https://github.com/kunmei/python_visualization/blob/master/scikit-learn/LDA-QDA.ipynb)。
