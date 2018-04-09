---
layout:     post
title:      Tensorflow中文版
subtitle:   手册
date:       2018-04-09
author:     john
catalog: true
tags:
    - 深度学习
    - tensorflow
---
### Tutorials
本章节包含如何在Tensorflow中做特别任务的手册说明。如果你是Tensorflow的新手，
我建议你们先阅读"Get Started"的章节。

#### 图像
下面手册覆盖了图像识别的不同方面:
- [MNIST](https://www.tensorflow.org/tutorials/layers?hl=zh-cn)，介绍了卷积神经网络，以及如何在Tensorflow中建立CNN。
- [图像识别](https://www.tensorflow.org/tutorials/image_recognition?hl=zh-cn)，介绍了图像识别的领域以及用一个预训练的模型去识别图像。
- [对于新的类别，如何重新训练Inception的最后一层](https://www.tensorflow.org/tutorials/image_retraining?hl=zh-cn)，是一个很好自我解释的标题。
- [卷积神经网络](https://www.tensorflow.org/tutorials/deep_cnn?hl=zh-cn)，展示了如何建立一个小的CNN网络用来进行图像识别。这个文档是针对高级的Tensorflow使用者的。

#### 序列
下面这些手册聚焦于处理序列数据的机器学习问题
- [循环神经网络](https://www.tensorflow.org/tutorials/recurrent?hl=zh-cn)，展示如何利用循环神经网络去预测一个序列中的下一个单词。
- [自然机器翻译手册](https://www.tensorflow.org/tutorials/seq2seq?hl=zh-cn)，展示了如何用一个序列到序列的模型去将文本从英语翻译成法语。
- [用于绘画分类的循环神经网络](https://www.tensorflow.org/tutorials/recurrent_quickdraw?hl=zh-cn)，建立一个分类模型用于绘画，直接从笔画的顺序开始。
- [简单的语音识别](https://www.tensorflow.org/tutorials/audio_recognition?hl=zh-cn)，展示了如何去建立一个基础的语音识别网络。

#### 数据表示
下面的手册将展示Tensorflow中不同的数据表示
- [Tensorflow线性模型](https://www.tensorflow.org/tutorials/wide?hl=zh-cn)，使用[特征列](https://www.tensorflow.org/api_docs/python/tf/feature_column?hl=zh-cn)将各种数据类型提供给线性模型，以解决分类问题。
- [Tensorflow宽和深的学习网络](https://www.tensorflow.org/tutorials/wide_and_deep?hl=zh-cn)，建立在上面线性模型手册的基础上，加上一个深度前馈神经网络部分和一个兼容DNN的数据表示。
- [单词的向量表示](https://www.tensorflow.org/tutorials/word2vec?hl=zh-cn)，展示如何建立一个词嵌入。
- [用显示的核方法提升线性模型](https://www.tensorflow.org/tutorials/kernel_methods?hl=zh-cn)，展示了如何用显示的核映射去改善线性模型的质量。

### 非机器学习
尽管Tensorflow专注于机器学习，你还可以使用Tensorflow去解决其他一些数学问题.
- [曼德博集合](https://www.tensorflow.org/tutorials/mandelbrot?hl=zh-cn)
- [偏微分方程](https://www.tensorflow.org/tutorials/pdes?hl=zh-cn)
