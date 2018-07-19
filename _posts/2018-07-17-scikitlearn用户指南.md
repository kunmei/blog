---
layout:     post
title:      scikit-learn用户指南
subtitle:   监督学习-广义线性模型
date:       2018-07-17
author:     john
catalog: true
tags:
    - 机器学习
---
## 监督学习
### 广义线性模型
以下是一组用于回归的方法，其中目标值应该是输入变量的线性组合。在数学概念中，`y`是预测值。如下式所示:

$$
  \hat { y } (w,x)=w_{ 0 }+w_{ 1 }x_{ 1 }+...+w_{ p }x_{ p }
$$

在整个模块中，我们将向量$$w=(w_1,..., w_p)$$指定为coef_和intercept_。

如果使用广义线性回归进行分类，可以参看[逻辑回归](#lr)。

#### <span id="ols">普通最小二乘</span>
`线性回归`拟合具有系数的线性模型，以最小化数据集中观察到的响应与线性近似预测的响应之间的残差平方和。在数学上它解决了以下形式的问题:

$$
  \min _{ w }{ { { \left\| Xw-y \right\|  }_{ 2 } }^{ 2 } }
$$

LinearRegression将采用其拟合方法数组X，y，并将线性模型的系数存储在其coef_成员中。

```python
>>> from sklearn import linear_model
>>> reg = linear_model.LinearRegression()
>>> reg.fit ([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
>>> reg.coef_
array([ 0.5,  0.5])
```
##### 普通最小二乘的复杂度
该方法使用X的奇异值分解来计算最小二乘解，如果X是大小为(n, p)的矩阵，方法的时间复杂度是$$O(n{ p }^{ 2 })$$，假设$$n\ge p$$。

#### 岭回归
岭回归通过对系数的大小施加惩罚来解决普通最小二乘的一些问题。脊系数最小化了惩罚的残差平方和。如下式所示:

$$
  \min _{ w }{ { { \left\| Xw-y \right\|  }_{ 2 } }^{ 2 } } +\alpha { { \left\| w \right\|  }_{ 2 } }^{ 2 }
$$

这里$$\alpha\ge 0$$是一个控制收缩量的复杂性参数：值越大，收缩量越大，因此系数对共线性越强大。

与其他线性模型一样，Ridge将采用其拟合方法数组X，y并将线性模型的系数存储在其coef_成员中。

```python
>>> from sklearn import linear_model
>>> reg = linear_model.Ridge (alpha = .5)
>>> reg.fit ([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
Ridge(alpha=0.5, copy_X=True, fit_intercept=True, max_iter=None,
      normalize=False, random_state=None, solver='auto', tol=0.001)
>>> reg.coef_
array([ 0.34545455,  0.34545455])
>>> reg.intercept_
0.13636...
```
##### 岭回归的复杂度
岭回归同[普通最小平方](#ols)有相同的复杂度。
##### 设置正则化: 通用的交叉验证
RidgeCV通过alpha参数的内置交叉验证实现岭回归。
```python
>>> from sklearn import linear_model
>>> reg = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
>>> reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])       
RidgeCV(alphas=[0.1, 1.0, 10.0], cv=None, fit_intercept=True, scoring=None,
    normalize=False)
>>> reg.alpha_                                     
0.1
```
#### Lasso回归
`Lasso`是一种估计稀疏系数的线性模型。它在某些情况下很有用，因为它倾向于选择具有较少参数值的解决方案，从而有效地减少给定解决方案所依赖的变量数量。因此，Lasso及其变体是压缩传感领域的基础。在某些条件下，它可以恢复精确的非零权重集。
```python
>>> from sklearn import linear_model
>>> reg = linear_model.Lasso(alpha = 0.1)
>>> reg.fit([[0, 0], [1, 1]], [0, 1])
Lasso(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000,
   normalize=False, positive=False, precompute=False, random_state=None,
   selection='cyclic', tol=0.0001, warm_start=False)
>>> reg.predict([[1, 1]])
array([ 0.8])
```
由于Lasso回归可以产生稀疏模型，因此可以用来进行特征选择。

##### 设置正则化参数
- 使用交叉验证
- 基于信息标准的模型选择
- SVM正则化参数的比较

#### 多任务Lasso
MultiTaskLasso是一个线性模型，可以共同估计多个回归问题的稀疏系数:y是一个形如(n_samples, n_tasks)的二维数组。

下图比较了使用简单的Lasso或者MultiTaskLasso获得的W中的非零的位置。Lasso估计产生了分散的非零，而MultiTaskLasso的非零是完整列。

#### 弹性网络
弹性网络是用L1和L2先验作为正则的线性模型。这种组合允许学习稀疏模型，其中很少权重像Lasso一样非零，同时仍然保持Ridge的正则化特性。我们使用l1_ratio参数控制L1和L2的凸组合。

当存在多个彼此相关的特征时，弹性网络是有用的。当存在多个彼此相关的特征时，弹性网络是有用的，Lasso很可能随机选择其中一种，而弹性网络很可能都会选择。在Lasso和Ridge之间进行权衡的一个实际优势是它允许弹性网络在旋转下继承Ridge的一些稳定性。

#### 多任务弹性网络
MultiTaskElasticNet是一个弹性网络，可以共同估计多个回归问题的稀疏系数:`Y`是一个类似于(n_samples，n_tasks)形式的二维数组。

#### 最小角度回归
最小角度回归是一种对高维数据的回归算法。LARS类似于前向逐步回归，在每一步，它都会找到与响应最相关的预测变量。

LARS的优势是:
- 当维度的数量远远大于数据点的，即p >> n是有效的。
- 它在计算上与前向选择一样快，并且具有与普通最小二乘相同的复杂度。
- 它产生完整的分段线性解决方案路径，这在交叉验证或类似的模型调参中很有用。
- 如果两个变量与响应有相同的相关性，那么它们的系数应该以大致相同的速率增加。因此，算法表现为直觉所期望的，并且也是稳定的。

LARS的劣势是:
- 因为LARS是基于迭代重新拟合残差，所以它对噪声的影响非常敏感。

#### LARS Lasso
```python
>>> from sklearn import linear_model
>>> reg = linear_model.LassoLars(alpha=.1)
>>> reg.fit([[0, 0], [1, 1]], [0, 1])  
LassoLars(alpha=0.1, copy_X=True, eps=..., fit_intercept=True,
     fit_path=True, max_iter=500, normalize=True, positive=False,
     precompute='auto', verbose=False)
>>> reg.coef_    
array([ 0.717157...,  0.        ])
```

#### 贝叶斯回归
贝叶斯回归的优势:
- 适应于手头的数据
- 它可用于在评估过程中包括正则化参数

贝叶斯回归的劣势:
- 模型的推断可能是耗时的

```python
>>> from sklearn import linear_model
>>> X = [[0., 0.], [1., 1.], [2., 2.], [3., 3.]]
>>> Y = [0., 1., 2., 3.]
>>> reg = linear_model.BayesianRidge()
>>> reg.fit(X, Y)
BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, compute_score=False, copy_X=True,
       fit_intercept=True, lambda_1=1e-06, lambda_2=1e-06, n_iter=300,
       normalize=False, tol=0.001, verbose=False)
```

#### <span id="lr">逻辑回归</span>
logistic回归在sklearn中的实现可以通过LogisticRegression进行回归。该实现可以对多分类(one-vs-rest)logistic回归进行拟合，正则项为:L2或L1范式。

LogisticRegression的实现方式有: “liblinear”(c++封装库)，“newton-cg”，“lbfgs”和“sag”。其中“lbfgs”和“newton-cg”只支持L2罚项，对于高维数据集来说收敛更快。L1罚项会产生稀疏预测权重。关于不同solver的选择如下:

case  |  solver
--|--
小数据集或L1罚项  | liblinear
Multinomial loss  |  lbfgs或newton-cg
大数据集  |  sagd

#### 多项式回归:使用基函数扩展线性模型
机器学习中的一种常见模式是在非线性函数的数据上面使用线性模型。这种方法保持了线性方法的一般快速性能，同时允许它们适合更广泛的数据范围。

举个例子，可以通过从系数构造多项式特征来扩展简单的线性模型。在标准的线性回归情况下，对于二维数据模型可能看起来如下:

$$
  \hat{y}(w, x) = w_0 + w_1 x_1 + w_2 x_2
$$

如果我们想要抛物面拟合数据而不是平面，我们可以用二阶多项式组合特征，因此模型看起来如下所示:

$$
  \hat{y}(w, x) = w_0 + w_1 x_1 + w_2 x_2 + w_3 x_1 x_2 + w_4 x_1^2 + w_5 x_2^2
$$

从上式可以看出多项式回归和我们上面考虑的是同一种模型，可以通过相同的技术解决。通过考虑使用这些基函数构建的高维空间内的线性拟合，该模型可以灵活地适应更广泛的数据范围。

PolynomialFeatures该预处理器可以将输入数据矩阵变换为给定程度的新数据矩阵。

```python
>>> from sklearn.preprocessing import PolynomialFeatures
>>> from sklearn.linear_model import LinearRegression
>>> from sklearn.pipeline import Pipeline
>>> import numpy as np
>>> model = Pipeline([('poly', PolynomialFeatures(degree=3)),
...                   ('linear', LinearRegression(fit_intercept=False))])
>>> # fit to an order-3 polynomial data
>>> x = np.arange(5)
>>> y = 3 - 2 * x + x ** 2 - x ** 3
>>> model = model.fit(x[:, np.newaxis], y)
>>> model.named_steps['linear'].coef_
array([ 3., -2.,  1., -1.])
```
