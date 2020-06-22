import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gzip
import matplotlib; matplotlib.use('TkAgg')
import matplotlib
import os
from tensorflow import keras
# 修正pycharm中print表格时因行数/列数过多而出现省略号的bug
pd.set_option('display.width', 10000)  # 设置字符显示宽度
pd.set_option('display.max_rows', None)  # 设置显示最大行
pd.set_option('display.max_columns',10000)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth',10000)

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号


zhfont = matplotlib.font_manager.FontProperties(fname='M:/fashion/simhei.ttf')



def read_data():
    files = [
        'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
    ]
    # 我在当前的目录下创建文件夹，里面放入上面的四个压缩文件
    current = './data/fashion/'
    paths = []
    for i in range(len(files)):
        paths.append('./data/fashion/' + files[i])
    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)
    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)
    return (x_train, y_train), (x_test, y_test)


(train_images, train_labels), (test_images, test_labels) = read_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
class_names = ['短袖圆领T恤', '裤子', '套衫', '连衣裙', '外套',
              '凉鞋', '衬衫', '运动鞋','包', '短靴']

# # 创建一个新图形
# plt.figure()
#
# # 显示一张图片在二维的数据上 train_images[0] 第一张图
# plt.imshow(train_images[0])
#
# # 在图中添加颜色条
# plt.colorbar()
#
# # 是否显示网格线条,True: 显示，False: 不显示
# plt.grid(False)
# plt.show()
# # 训练图像缩放255，在0 和 1 的范围
# train_images = train_images / 255.0
#
# # 测试图像缩放
# test_images = test_images / 255.0
#
# # 保存画布的图形，宽度为 10 ， 长度为10
# plt.figure(figsize=(10, 10))

# 显示训练集的 25 张图像
# for i in range(25):
#     # 创建分布 5 * 5 个图形
#     plt.subplot(5, 5, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     # 显示照片，以cm 为单位。
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#
#     # 此处就引用到上面的中文字体，显示指定中文，对应下方的图片意思，以证明是否正确
#     plt.xlabel(class_names[train_labels[i]], fontproperties=zhfont)


def build_model():
    # 线性叠加
    model = tf.keras.models.Sequential()
    # 改变平缓输入
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    # 第一层紧密连接128神经元
    # model.add(tf.keras.layers.Dense(1048, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    # 第二层分10 个类别
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
    return model


model = build_model()
model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(train_images, train_labels, epochs=5)
model.save('fashionmodel.h5')
# 评估模型（主要是测试数据集）
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('测试损失：%f 测试准确率: %f' % (test_loss, test_acc))


predictions = model.predict(test_images)

# 提取20个数据集，进行预测判断是否正确
# for i in range(100):
#     pre = class_names[np.argmax(predictions[i])]
#     tar = class_names[test_labels[i]]
#     print("预测：%s   实际：%s" % (pre, tar))
#
# plt.figure(figsize=(10, 10))
picacc=0
# 预测 25 张图像是否准确，不准确为红色。准确为蓝色
for i in range(256):
    #创建分布 5 * 5 个图形
    plt.subplot(16, 16, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    # 显示照片，以cm 为单位。
    plt.imshow(test_images[i], cmap=plt.cm.binary)

    #预测的图片是否正确，黑色底表示预测正确，红色底表示预测失败
    predicted_label = np.argmax(predictions[i])
    true_label = test_labels[i]
    if predicted_label == true_label:
        color = 'black'
        picacc+=1/256
    else:
        color = 'red'
    plt.xlabel("{} ({})".format(class_names[predicted_label],
                                class_names[true_label]),
               color=color,
               fontproperties=zhfont)
plt.show()
print(picacc)

# def test_model(model):
#     """
#
#     :param model:
#     :return:
#     """
#     # 验证模型
#     test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
#     print('\ntest loss: {}, test acc: {:5.2f}'.format(test_loss, test_acc * 100))
#
#     # 预测模型
#     predictions = model.predict(test_images)
#     print(predictions[0])
#     print(np.argmax(predictions[0]), test_labels[0])
# new_model_with_weights = tf.keras.models.load_model('fashionmodel.h5')
# new_model_with_weights.summary()
# test_model(new_model_with_weights)