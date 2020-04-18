---
layout:     post
title:      scikit-learn用户指南
subtitle:   监督学习-最近邻
date:       2018-07-23
author:     john
catalog: true
tags:
    - 机器学习
---
### 非监督近邻
`NearestNeighbors`实现了非监督的近邻学习。它给三个不同的近邻算法提供了统一的接口，分别是:BallTree，KDTree，和基于`sklearn.metrics.pairwise`例子的蛮力算法。
