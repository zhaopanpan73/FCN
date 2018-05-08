import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import pylab
print("TensorFlow Version: %s" % tf.__version__)

mnist = input_data.read_data_sets('MNIST_data', validation_size=0, one_hot=False)
# img = mnist.train.images[20]
# plt.imshow(img.reshape((28, 28)))
fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(20, 4))  # 是整个figure的大小，图片最后保存的大小为（20,4）*100像素
in_imgs = mnist.test.images[10:20]
noisy_imgs = in_imgs + 0.2* np.random.randn(*in_imgs.shape)  # numpy.random.randn(d0, d1, …, dn)是从标准正态分布中返回一个或多个样本值 形状由参数指定
noisy_imgs = np.clip(noisy_imgs, 0., 1.)
reconstructed = mnist.test.images[:10]

for images, row in zip([noisy_imgs, reconstructed], axes):
    for img, ax in zip(images, row):
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.imshow(img.reshape((28, 28)))
fig.tight_layout(pad=0.1)
plt.show()

# 图像显示问题
# 第一种引入 pylab
# import matplotlib.pyplot as plt
# import pylab
# plt.imshow(img)
# pylab.show()

# 第二种
# 在最后面 plt.show()