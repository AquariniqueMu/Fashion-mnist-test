from tensorflow import keras
import matplotlib; matplotlib.use('TkAgg')
fashion_mnist = keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()

# 衣服类别
class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal',
               'Shirt','Sneaker','Bag','Ankle boot']
print(train_images.shape,len(train_labels))
print(test_images.shape,len(test_labels))

# 预处理数据，将像素值除以255，使其缩放到0到1的范围
train_images = train_images / 255.0
test_images = test_images / 255.0


# 搭建网络结构
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(256,activation='relu'),
    keras.layers.Dense(10,activation='softmax')
])

# 设置损失函数、优化器及训练指标
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 训练模型
model.fit(train_images,train_labels,epochs=20)

# 模型评估
test_loss,test_acc=model.evaluate(test_images,test_labels,verbose=2)
print('测试准确率:',test_acc)

# 选择测试集中的图像进行预测
predictions=model.predict(test_images)


