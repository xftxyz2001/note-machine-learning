# import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
train_img = mnist.train.images
test_img = mnist.test.images
train_label = mnist.train.labels
test_label = mnist.test.labels

# for i in range(5):
#     img = np.reshape(train_img[i, :], (28, 28))
#     label = np.argmax(train_label[i, :])
#     plt.rcParams['figure.figsize'] = (3.0, 3.0)
#     plt.matshow(img, cmap = plt.get_cmap('gray'))
#     plt.title('第%d张图片 标签为%d' %(i+1,label))
#     plt.show()

f, a = plt.subplots(1, 3, figsize=(10, 6))
for i in range(3):
    img = np.reshape(test_img[i, :], (28, 28))
    label = np.argmax(test_label[i, :])
    # a[i].matshow(img, cmap=plt.get_cmap('gray'))
    a[i].imshow(img, cmap=plt.get_cmap('gray'))
    a[i].set_title('第%d张图片 标签: %d' % (i+1, label))
    # a[0][i].imshow(np.reshape(test_img[i, :], (28, 28)), cmap=plt.get_cmap('gray'))
plt.show()

# img_merge = []
# label_l = []
# for i in range(4):
#     img = np.reshape(train_img[i, :], (28, 28))
#     label = np.argmax(train_label[i, :])
#     img_merge.append(img)
#     label_l.append(label)
# X = np.array(img_merge)
# Y = np.array(label_l)
# # 拼接4个二维img，一同显示
# X = [x for z in X for x in z]
# d = np.array(X)
# plt.matshow(d, cmap = plt.get_cmap('gray'))
# # 绘制矩阵图 colormap 序列化（连续化）色图
# # gray：0 - 255级灰度，0：黑色，1：白色，黑底白字；
# # gray_r：翻转gray的显示，如果gray将图像显示为黑底白字，gray_r会将其显示为白底黑字；
# # binary
# plt.title('训练集前4张图片')
# plt.show()
