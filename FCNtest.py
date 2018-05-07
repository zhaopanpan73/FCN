import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
print("TensorFlow Version: %s" % tf.__version__)

mnist = input_data.read_data_sets('MNIST_data', validation_size=0, one_hot=False)
img = mnist.train.images[20]
plt.imshow(img.reshape((28, 28)))

inputs_ = tf.placeholder(tf.float32, (None, 28, 28, 1), name='inputs_')
targets_ = tf.placeholder(tf.float32, (None, 28, 28, 1), name='targets_')


conv1 = tf.layers.conv2d(inputs_, 64, (3,3), padding='same', activation=tf.nn.relu)
conv1 = tf.layers.max_pooling2d(conv1, (2,2), (2,2), padding='same')

conv2 = tf.layers.conv2d(conv1, 64, (3,3), padding='same', activation=tf.nn.relu)
conv2 = tf.layers.max_pooling2d(conv2, (2,2), (2,2), padding='same')

conv3 = tf.layers.conv2d(conv2, 32, (3,3), padding='same', activation=tf.nn.relu)
conv3 = tf.layers.max_pooling2d(conv3, (2,2), (2,2), padding='same')

conv4 = tf.image.resize_nearest_neighbor(conv3, (7,7))
conv4 = tf.layers.conv2d(conv4, 32, (3,3), padding='same', activation=tf.nn.relu)

conv5 = tf.image.resize_nearest_neighbor(conv4, (14,14))
conv5 = tf.layers.conv2d(conv5, 64, (3,3), padding='same', activation=tf.nn.relu)

conv6 = tf.image.resize_nearest_neighbor(conv5, (28,28))
conv6 = tf.layers.conv2d(conv6, 64, (3,3), padding='same', activation=tf.nn.relu)

logits_ = tf.layers.conv2d(conv6, 1, (3,3), padding='same', activation=None)

outputs_ = tf.nn.sigmoid(logits_, name='outputs_')

loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_, logits=logits_)
cost = tf.reduce_mean(loss)

optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

sess = tf.Session()

noise_factor = 0.5
epochs = 10
batch_size = 128
sess.run(tf.global_variables_initializer())

for e in range(epochs):
    for idx in range(mnist.train.num_examples // batch_size):
        batch = mnist.train.next_batch(batch_size)
        imgs = batch[0].reshape((-1, 28, 28, 1))

        # 加入噪声
        noisy_imgs = imgs + noise_factor * np.random.randn(*imgs.shape)
        noisy_imgs = np.clip(noisy_imgs, 0., 1.)
        batch_cost, _ = sess.run([cost, optimizer],
                                 feed_dict={inputs_: noisy_imgs,
                                            targets_: imgs})

        print("Epoch: {}/{} ".format(e + 1, epochs),
              "Training loss: {:.4f}".format(batch_cost))


fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(20,4))
in_imgs = mnist.test.images[10:20]
noisy_imgs = in_imgs + noise_factor * np.random.randn(*in_imgs.shape)
noisy_imgs = np.clip(noisy_imgs, 0., 1.)

reconstructed = sess.run(outputs_,
                         feed_dict={inputs_: noisy_imgs.reshape((10, 28, 28, 1))})

for images, row in zip([noisy_imgs, reconstructed], axes):
    for img, ax in zip(images, row):
        ax.imshow(img.reshape((28, 28)))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

fig.tight_layout(pad=0.1)
sess.close()