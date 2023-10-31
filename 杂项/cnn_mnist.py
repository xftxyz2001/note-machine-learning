# https://www.bilibili.com/video/BV13V411b78a#reply4201150084
# 谢谢大家的关注啊，祝你们成功
# 教程有用的话，可以到评论区祝我暴富噢
import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 若是使用tf2，可能会有警告或者错误，可使用tensorflow.compat.v1取代tensorflow
# import tensorflow.compat.v1 as tf
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# cpu版tf只能调用cpu来训练
# 若安装了gpu版tf，默认会调用gpu训练，倘若失败，可手动选择DEVICES为"0"，只想使用cpu训练则设"-1"
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)  # 若显存报错，可试试强制分配

# 导入input_data用于自动下载和安装MNIST数据集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
learning_rate = 1e-4

# 这个超参数调一调有奇效噢0.0
keep_prob_rate = 0.7
# drop out比例（补偿系数）为了保证神经元输出激活值的期望值与不使用dropout时一致，我们结合概率论的知识来具体看一下：
# 假设一个神经元的输出激活值为a，在不使用dropout的情况下，其输出期望值为a
# 如果使用了dropout，神经元就可能有保留和关闭两种状态，把它看作一个离散型随机变量，符合概率论中的0-1分布
# 其输出激活值的期望变为 p*a+(1-p)*0=pa，为了保持测试集与训练集中每个神经元输出的分布一致，简单来看有两种方法(默认2)
# 1：测试时乘以此系数 2：训练时输出节点按照keep_prob_rate概率置0，若未置0则以1/keep_prob的比例缩放该节点（而并非保持不变）

# 可以简单地计算一下，一次iter是取100个样本，训练集有55000个样本，所以max_epoch理论上要大于550才对
max_epoch = 1000

# 权重矩阵初始化


def weight_variable(shape):
    # tf.truncated_normal从截断的正态分布中输出随机值.
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# 偏置初始化


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 使用tf.nn.conv2d定义2维卷积


def conv2d(x, W):
    # 卷积核移动步长为1,填充padding类型为SAME,简单地理解为以0填充边缘, VALID采用不填充的方式，多余地进行丢弃
    # 计算给定的4-D input和filter张量的2-D卷积
    # input shape [batch, in_height, in_width, in_channels]
    # filter shape [filter_height, filter_width, in_channels, out_channels]
    # stride 长度为4的1-D张量,input的每个维度的滑动窗口的步幅
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# 使用tf.nn.max_pool定义2维max pool


def max_pool_2x2(x):
    # 采用最大池化，也就是取窗口中的最大值作为结果
    # x 是一个4维张量，shape为[batch,height,width,channels]
    # ksize表示pool窗口大小为2x2,也就是高2，宽2
    # strides表示在height和width维度上的步长都为2，其余两维皆为1
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 计算accuracy
# 以test集为例，输入为 v_xs (10000,784), y_ys (10000,10)


def compute_accuracy(v_xs, v_ys):
    global prediction
    # y_pre将v_xs输入模型后得到的预测值 (10000,10)
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    # argmax(axis) axis = 1 返回结果为：数组中每一行最大值所在“列”索引值
    # tf.equal返回布尔值，correct_prediction (10000，1)
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    # tf.cast将bool转成float32, tf.reduce_mean求均值，作为accuracy值(0到1)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


xs = tf.placeholder(tf.float32, [None, 784], name='x_input')
ys = tf.placeholder(tf.float32, [None, 10], name='y_input')
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])
# 输入2D转化为4D，便于conv操作
# 输入shape为[batch, 784]
# 变成4d的x_image，shape应该是[batch,28,28,1]，第四维代表通道数1
# -1表示自动推测这个维度的size

# 具体定义各层网络，以及它们之间的连接
#  卷积层 1
## conv1 layer ##
W_conv1 = weight_variable([5, 5, 1, 32])
# 初始化W_conv1为[5,5,1,32]的张量tensor，表示卷积核大小为5*5，1表示图像通道数，6表示卷积核个数，三个等价的概念(输出通道数、输出特征图数、卷积核数)
# 3                                    这个 0 阶张量就是标量，shape=[]
# [1., 2., 3.]                         这个 1 阶张量就是向量，shape=[3]
# [[1., 2., 3.], [4., 5., 6.]]         这个 2 阶张量就是二维数组，shape=[2, 3]
# [[[1., 2., 3.]], [[7., 8., 9.]]]     这个 3 阶张量就是三维数组，shape=[2, 1, 3]
# 即有几层中括号
b_conv1 = bias_variable([32])
# 5x5x1的卷积核作用在28x28x1的二维图上 output size 28x28x32
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# output size 14x14x32 卷积操作使用padding保持维度不变，只靠pool降维
h_pool1 = max_pool_2x2(h_conv1)
#  卷积层 2
## conv2 layer ##
W_conv2 = weight_variable([5, 5, 32, 64])  # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) +
                     b_conv2)  # output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)  # output size 7x7x64
#  全连接层 1
## fc1 layer ##
# 1024个神经元的全连接层
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 全连接层 2
## fc2 layer ##
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 计算交叉熵损失
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))

# 定义训练操作，使用ADAM优化器来做梯度下降
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# 图构建完毕，创建sess执行图
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print("step 0, test accuracy %g" % compute_accuracy(
        mnist.test.images, mnist.test.labels))
    start = time.time()
    for i in range(max_epoch):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        # 此batch是个2维tuple，batch[0]是(100，784)的样本数据数组，batch[1]是(100，10)的样本标签数组
        sess.run(train_step, feed_dict={
                 xs: batch_xs, ys: batch_ys, keep_prob: keep_prob_rate})
        if (i+1) % 50 == 0:
            print("step %d, test accuracy %g" % ((i+1), compute_accuracy(
                mnist.test.images, mnist.test.labels)))
    end = time.time()
    print('******************************************************')
    print("运行时间:%.2f秒" % (end - start))
