---
layout:     post
title:      scikit-learn用户指南
subtitle:   监督学习-支持向量机
date:       2018-07-19
author:     john
catalog: true
tags:
    - 机器学习
---
### 支持向量机
支持向量机（SVM）是一组用于分类，回归和异常值检测的监督学习方法。

支持向量机的优点是:
- 在高维空间内是有效的
- 在维度大小大于样本大小的情况下仍然有效
- 在决策函数中使用训练点的子集(称为支持向量)，因此它也是内存有效的
- 多功能:可以为决策功能指定不同的内核功能，提供了通用内核，也可以指定自定义内核

支持向量机的缺点是:
- 如果特征的数量远大于样本的数量，选择核函数和正则化项从而避免过拟合是至关重要的
- SVM不直接提供概率估计，这些是使用昂贵的五折交叉验证计算的

scikit-learn中的支持向量机支持密集和稀疏样本向量作为输入，但是要使用SVM对稀疏数据进行预测，必须也有同样的数据进行训练。

#### 分类
SVC，NuSVC和LinearSVC是能够对数据集执行多类分类的类。二分类的例子如下:
```python
>>> from sklearn import svm
>>> X = [[0, 0], [1, 1]]
>>> y = [0, 1]
>>> clf = svm.SVC()
>>> clf.fit(X, y)  
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
>>> clf.predict([[2., 2.]])
array([1])

>>> # get support vectors
>>> clf.support_vectors_
array([[ 0.,  0.],
       [ 1.,  1.]])
>>> # get indices of support vectors
>>> clf.support_
array([0, 1]...)
>>> # get number of support vectors for each class
>>> clf.n_support_
array([1, 1]...)
```

多分类可以实现两种不同的方式，一种是"one-against-one"方法，将构造出`n_class * (n_class - 1) / 2`个分类器。另一种是"one-over-rest"方法，将构造出`n_class`个分类器。

```python
>>> X = [[0], [1], [2], [3]]
>>> Y = [0, 1, 2, 3]
>>> clf = svm.SVC(decision_function_shape='ovo')
>>> clf.fit(X, Y)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
>>> dec = clf.decision_function([[1]])
>>> dec.shape[1] # 4 classes: 4*3/2 = 6
6
>>> clf.decision_function_shape = "ovr"
>>> dec = clf.decision_function([[1]])
>>> dec.shape[1] # 4 classes
4
```
同时，`LinearSVC`实现的是`one-vs-the-rest`多分类策略，因此训练`n_class`个模型。如果只有两个类别，将只有一个模型被训练。
```python
>>> lin_clf = svm.LinearSVC()
>>> lin_clf.fit(X, Y)
LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
     verbose=0)
>>> dec = lin_clf.decision_function([[1]])
>>> dec.shape[1]
4
```
#### 回归
支持向量分类的方法可以扩展到解决回归问题。此方法称为支持向量回归。

由支持向量分类（如上所述）产生的模型仅取决于训练数据的子集，因为用于构建模型的成本函数不关心超出边缘的训练点。类似地，支持向量回归生成的模型仅依赖于训练数据的子集，因为构建模型的成本函数忽略了接近模型预测的任何训练数据。

```python
>>> from sklearn import svm
>>> X = [[0, 0], [2, 2]]
>>> y = [0.5, 2.5]
>>> clf = svm.SVR()
>>> clf.fit(X, y)
SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
>>> clf.predict([[1, 1]])
array([ 1.5])
```
