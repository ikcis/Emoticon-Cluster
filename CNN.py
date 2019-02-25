import os
import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import glob
import cv2
from sklearn.utils import shuffle

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# 读取数据集

def load_train(train_path, image_size, classes):
    images = []
    labels = []
    ids = []
    cls = []

    print('Reading training images')
    # 数据集由多个由标签命名的文件夹组成
    for fld in classes:
        index = classes.index(fld)
        print('Loading {} files (Index: {})'.format(fld, index))
        path = os.path.join(train_path, fld, '*g')
        files = glob.glob(path)
        for fl in files:
            image = cv2.imread(fl)
            image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            images.append(image)
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
            flbase = os.path.basename(fl)
            ids.append(flbase)
            cls.append(fld)
    images = np.array(images)
    labels = np.array(labels)
    ids = np.array(ids)
    cls = np.array(cls)

    return images, labels, ids, cls


def load_test(test_path, image_size):
    path = os.path.join(test_path, '*g')
    files = sorted(glob.glob(path))

    X_test = []
    X_test_id = []
    print("Reading test images")
    for fl in files:
        flbase = os.path.basename(fl)
        img = cv2.imread(fl)
        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        X_test.append(img)
        X_test_id.append(flbase)

    # 对测试集标准化
    X_test = np.array(X_test, dtype=np.uint8)
    X_test = X_test.astype('float32')
    X_test = X_test / 255

    return X_test, X_test_id


class DataSet(object):

    def __init__(self, images, labels, ids, cls):
        self._num_examples = images.shape[0]
        # 调整shape 由[num examples, rows, columns, depth]至[num examples, rows*columns]
        # 由[0, 255]至[0.0, 1.0]
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._ids = ids
        self._cls = cls
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def ids(self):
        return self._ids

    @property
    def cls(self):
        return self._cls

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        # 完成当前epoch
        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        # 根据这份data返回下一个batch的example
        return self._images[start:end], self._labels[start:end], self._ids[start:end], self._cls[start:end]


def read_train_sets(train_path, image_size, classes, validation_size=0):
    class DataSets(object):
        pass

    data_sets = DataSets()

    images, labels, ids, cls = load_train(train_path, image_size, classes)
    images, labels, ids, cls = shuffle(images, labels, ids, cls)  # shuffle the data

    if isinstance(validation_size, float):
        validation_size = int(validation_size * images.shape[0])

        validation_images = images[:validation_size]
        validation_labels = labels[:validation_size]
        validation_ids = ids[:validation_size]
        validation_cls = cls[:validation_size]

        train_images = images[validation_size:]
        train_labels = labels[validation_size:]
        train_ids = ids[validation_size:]
        train_cls = cls[validation_size:]

        data_sets.train = DataSet(train_images, train_labels, train_ids, train_cls)
        data_sets.valid = DataSet(validation_images, validation_labels, validation_ids, validation_cls)

    return data_sets


def read_test_set(test_path, image_size):
    images, ids = load_test(test_path, image_size)
    return images, ids


# 配置和选定超参数

# 卷积层1
filter_size1 = 5
num_filters1 = 64

# 卷积层2
filter_size2 = 3
num_filters2 = 64

# 全连接层1
fc1_size = 128

# 全连接层2
fc2_size = 128

# 图像通道数
num_channels = 3

# 图像维度
img_size = 64

# 平坦化成一维向量后的size
img_size_flat = img_size * img_size * num_channels

# 高度和宽度
img_shape = (img_size, img_size)

classes = ['Ali', 'Cangshu', 'Huaji', 'Panda', 'Sadfrog']
num_classes = len(classes)

batch_size = 32
# 验证集大小
validation_size = .2

# validation loss不再改善的等待时间
early_stopping = None

train_path = 'dataset'

# 读取测试集数据
data = read_train_sets(train_path, img_size, classes, validation_size=validation_size)
print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Validation:\t{}".format(len(data.valid.labels)))


# TensorFlow Graph
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


# 卷积层
def new_conv_layer(input, num_input_channels, filter_size, num_filters, use_pooling=True):
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)

    # strides在每个维度中都是1，第一个和最后一个一般都是1，因为第一个是图像数量，最后一个是输入通道
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')  # 输入输出的size相同
    # 增加biases
    layer += biases
    # 池化操作 保留主要的特征同时减少参数(降维)和计算量，防止过拟合，提高模型泛化能力
    if use_pooling:
        # 2x2 max-pooling
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')
    # 激活函数
    # 激活函数一般在池化操作之前执行，但如果relu(max_pool(x)) == max_pool(relu(x))
    # 那么激活函数放在池化操作之后进行可以节约大量激活函数的迭代
    layer = tf.nn.relu(layer)
    return layer, weights


# 平坦化layer
def flatten_layer(layer):
    # layer_shape == [num_images, img_height, img_width, num_channels]
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()

    # [num_images, num_features]
    layer_flat = tf.reshape(layer, [-1, num_features])

    # flattened layer [num_images, img_height * img_width * num_channels]
    return layer_flat, num_features


# 全连接层
def new_fc_layer(input, num_inputs, num_outputs, use_relu=True):
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    layer = tf.matmul(input, weights) + biases

    if use_relu:
        layer = tf.nn.relu(layer)
    return layer


# 占位符和变量
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)

# 卷积层1
layer_conv1, weights_conv1 = new_conv_layer(input=x_image, num_input_channels=num_channels, filter_size=filter_size1,
                                            num_filters=num_filters1, use_pooling=True)

# 卷积层2
layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1, num_input_channels=num_filters1,
                                            filter_size=filter_size2, num_filters=num_filters2, use_pooling=True)
# Flatten Layer
layer_flat, num_features = flatten_layer(layer_conv2)

# 全连接层1
layer_fc1 = new_fc_layer(input=layer_flat, num_inputs=num_features, num_outputs=fc1_size, use_relu=True)

# 全连接层2
layer_fc2 = new_fc_layer(input=layer_fc1, num_inputs=fc1_size, num_outputs=num_classes, use_relu=False)

# Predicted Class
y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, axis=1)

# 最佳化cost function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc2, labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

# 表现衡量
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# TensorFlow Run
session = tf.Session()
session.run(tf.global_variables_initializer())
train_batch_size = batch_size


def print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    # 计算训练集上的准确度
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%}, Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))


# 记录迭代次数
total_iterations = 0


def optimize(num_iterations):
    global total_iterations
    start_time = time.time()

    best_val_loss = float("inf")
    patience = 0

    for i in range(total_iterations,
                   total_iterations + num_iterations):

        # x_batch:image, y_true_batch:labels
        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(train_batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(train_batch_size)

        # [num examples, rows, columns, depth] -> [num examples, flattened image shape]
        x_batch = x_batch.reshape(train_batch_size, img_size_flat)
        x_valid_batch = x_valid_batch.reshape(train_batch_size, img_size_flat)

        feed_dict_train = {x: x_batch, y_true: y_true_batch}
        feed_dict_validate = {x: x_valid_batch, y_true: y_valid_batch}

        session.run(optimizer, feed_dict=feed_dict_train)

        # 输出实时结果
        if i % int(data.train.num_examples / batch_size) == 0:
            val_loss = session.run(cost, feed_dict=feed_dict_validate)
            epoch = int(i / int(data.train.num_examples / batch_size))
            print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss)
            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience = 0
                else:
                    patience += 1

                if patience == early_stopping:
                    break

    total_iterations += num_iterations

    end_time = time.time()
    time_dif = end_time - start_time
    print("Time elapsed: " + str(timedelta(seconds=int(round(time_dif)))))


optimize(num_iterations=1000)
x_test = data.valid.images.reshape(100, img_size_flat)
feed_dict_test = {x: x_test, y_true: data.valid.labels}
val_loss = session.run(cost, feed_dict=feed_dict_test)
val_acc = session.run(accuracy, feed_dict=feed_dict_test)
msg_test = "Test Accuracy: {0:>6.1%}"
print(msg_test.format(val_acc))
