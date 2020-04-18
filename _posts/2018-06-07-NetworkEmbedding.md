---
layout:     post
title:      机器学习
subtitle:   Network Embedding Survey
date:       2018-06-07
author:     john
## header-img: img/post-bg-ios9-web.jpg
catalog: true
tags:
    - 深度学习
---
### DeepWalk
论文地址[DeepWalk: Online Learning of Social Representations](http://www.perozzi.net/publications/14_kdd_deepwalk.pdf)

DeepWalk是对图从一个节点开始使用random walk来生成类似文本的序列特征，然后将节点id作为一个个词使用skip gram训练得到词向量。

### node2vec
论文地址[node2vec: Scalable Feature Learning for Networks](https://cs.stanford.edu/~jure/pubs/node2vec-kdd16.pdf)

node2vec是在DeepWalk的基础上，定义了一个bias random walk的策略生成序列，仍然用skip gram进行训练。论文分析了DFS和BFS两种游走方式，保留的网络结构信息是不一样的。

### MMDW
论文地址[Max-Margin DeepWalk Discriminative Learning of Network Representation](https://www.ijcai.org/Proceedings/16/Papers/547.pdf)

MMDW是将DeepWalk和Max-Margin(SVM)结合起来解决监督学习问题。DW本身是非监督的，如果能够引入label数据，生成的向量对于分类问题会有更好的作用。

### TADW
论文地址[Network Representation Learning with Rich Text Information](https://ijcai.org/Proceedings/15/Papers/299.pdf)

DW实际上等同于对于一个特殊矩阵M的分解。在实际应用中，有一些节点会有文本信息，所以将文本直接以一个子矩阵的方式加入，会使学到的向量包含更丰富的信息。

### GraRep
论文地址[Learning Graph Representations with Global Structural Information](https://www.researchgate.net/profile/Qiongkai_Xu/publication/301417811_GraRep/links/5847ecdb08ae8e63e633b5f2/GraRep.pdf)

GraRep沿用矩阵分解的思路，分析了不同k-step所刻画的信息是不一样的，所以可以对每一个step的矩阵作分解，最后将每个步骤得到的向量表示拼接起来作为最后的结果。

### LINE
论文地址[Large scale information network embedding](http://www.www2015.it/documents/proceedings/proceedings/p1067.pdf)

LINE分析了一度相似性和二度相似性，其中一度相似性是两个点直接相连，边权重越大说明两个点越相似；二度相似性则是两个点之间共享了多少邻居，邻居越多，说明相似性越高。

### NEU
论文地址[Fast Network Embedding Enhancement via High Order Proximity Approximation](https://www.ijcai.org/proceedings/2017/0544.pdf)

文章分析了一些视为矩阵分解的embedding的方法，如果矩阵分解能更精确地包含高阶信息，效果会更好。

在上面的论文中我们只考虑了网络结构，但真实世界中的节点和边往往都会含有丰富的信息。例如在Quora场景中，每个用户自身会有一些label和文本，在一些场景里甚至边也会带上一些label，这些信息对于网络的构建其实是至关重要的，前面我们也看到了TADW将节点的文本信息纳入训练，下面罗列一些这个方向相关的论文。

### CANE
论文地址[Context-Aware Network Embedding for Relation Modeling](http://nlp.csai.tsinghua.edu.cn/~tcc/publications/acl2017_cane.pdf)

### CENE
论文地址[A General Framework for Content-enhanced Network Representation Learning](https://arxiv.org/pdf/1610.02906)

### Trans-Net
论文地址[Translation-Based Network Representation Learning for Social Relation Extraction](http://nlp.csai.tsinghua.edu.cn/~tcc/publications/ijcai2017_transnet_shenzhen.pdf)

### SSC-GCN
论文地址[Semi-Supervised Classification with Graph Convolutional Networks](https://openreview.net/pdf?id=SJU4ayYgl)

### SDNE
论文地址[Structural Deep Network Embedding](http://www.kdd.org/kdd2016/papers/files/rfp0191-wangAemb.pdf)

### PTE
论文地址[Predictive Text Embedding through Large-scale Heterogeneous Text Networks](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/fp292-Tang.pdf)

### HINES
论文地址[Heterogeneous Information Network Embedding for Meta Path based Proximity](https://arxiv.org/pdf/1701.05291)
