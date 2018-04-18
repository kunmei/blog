---
layout:     post
title:      tensorflow
subtitle:   Session, Graph, Operation, Tensor
date:       2018-04-18
author:     john
catalog: true
tags:
    - 深度学习
    - tensorflow
---
### 前言
&emsp;&emsp;`Tensorflow`是基于图(Graph)的计算系统。图的节点是由操作(operation)构成的，图的各个节点之间是由张量(Tensor)作为边连接在一起的。Tensorflow的计算过程是一个Tensor流图，而图必须是在一个Session中计算的。

### Session
[Session](http://www.tensorfly.cn/tfdoc/api_docs/python/client.html)提供Operation执行和Tensor求值的环境。示例代码如下:
```python
import tensorflow as tf

# Build a Graph
a = tf.constant([1.0, 2.0])
b = tf.constant([3.0, 4.0])
c = a * b

# Launch the graph in a Session
sess = tf.Session()

# Evaluate the tensor 'c'
print(sess.run(c))
sess.close()
```
一个Session会包含一些资源，例如Variable或者Queue。当我们不再需要这个Session的时候，需要对这些资源进行释放。两种方式:
1. 调用`session.close()`方式
2. 使用`with`创建上下文来执行，当上下文退出时自动释放

如下所示:
```python
import tensorflow as tf

# Build a Graph
a = tf.constant([1.0, 2.0])
b = tf.constant([3.0, 4.0])
c = a * b

with tf.Session() as sess:
  print(sess.run(c))
```

### Graph
`Tensorflow`使用[tf.Graph](http://www.tensorfly.cn/tfdoc/api_docs/python/framework.html#Graph)类表示可计算的图。图是由操作Operation和Tensor来构成，其中Operation表示图的节点(即计算单元)，Tensor表示图的边(Operation之间的流动单元)。
```python
import tensorflow as tf
g1 = tf.Graph()
with g1.as_default():
  c1 = tf.constant([1.0])
with tf.Graph().as_default() as g2:
  c2 = tf.constant([2.0])

with tf.Session(graph=g1) as sess1:
  print(sess1.run(c1))
with tf.Session(graph=g2) as sess2:
  print(sess2.run(c2))
```

### Operation
一个[Operation](http://www.tensorfly.cn/tfdoc/api_docs/python/framework.html#Operation)是一个Graph中的一个计算节点。Operation对象的创建是通过直接调用operation方法或者Graph.create_op()。

### Tensor
[Tensor](http://www.tensorfly.cn/tfdoc/api_docs/python/framework.html#Tensor)表示的是Operation的输出结果。

### 简单示例
```python
import tensorflow as tf
a = tf.constant(1)
b = tf.constant(2)
c = tf.constant(3)
d = tf.constant(4)

add1 = tf.add(a, b)
mul1 = tf.mul(b, c)
add2 = tf.add(c, d)
output = tf.add(add1, mul1)
with tf.Session() as sess:
  print(sess.run(output))
```
