import cv2
from imutils import paths
import os
import numpy as np
from sklearn.model_selection import train_test_split
import PyQt5

# 图像特征向量 调整图像大小 然后将图像平坦化为行像素列表
def feature_vectorized(image, size=(128, 128)):
    return cv2.resize(image, size).flatten()


# 提取颜色直方图 使用cv2.normalize从HSV颜色间距中提取3D颜色直方图，然后平坦化结果
def color_histogram(image, bins=(32, 32, 32)):
    # opencv默认为BGR而非RGB HSV更符合人对颜色相似性的主观感受
    # 颜色直方图放弃空间位置信息 而统计某一颜色出现的频数
    # 颜色直方图在BPNN方法中的效果比原始图片好得多
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 三个颜色通道 HSV颜色分量范围(0,180)(0,255)(0,255)
    his = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 255, 0, 255])
    # 标准化
    # 输入数组 输出数组
    cv2.normalize(his, his)
    # 平坦化为行像素列表
    return his.flatten()


folder_name = 'Pictures'

print('Start Processing Images...\n')

# 获取文件的列表
imagePaths = list(paths.list_images(folder_name))

# 初始化
raw_images = []
his_image = []
labels = []

# 读取文件
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    # 文件名的格式为 种类.序号.jpg
    label = imagePath.split(os.path.sep)[-1].split('.')[0]
    # 原始图片
    pixels = feature_vectorized(image)

    try:
        pixels = feature_vectorized(image)
    except Warning as e:
        print(imagePath)
        # os.remove(imagePath)
        continue
    # 直方图
    his = color_histogram(image)

    raw_images.append(pixels)
    his_image.append(his)
    labels.append(label)

raw_images = np.array(raw_images)
his_image = np.array(his_image)
labels = np.array(labels)

# 统计读入的数据
# 也起到检测的作用
print('Pictures : %d' % len(imagePaths))
print('Raw Image Matrix: {:.2f}MB'.format(raw_images.nbytes / (1024 * 1000.0)))
print('Histogram Image Matrix: {:.2f}MB\n'.format(his_image.nbytes / (1024 * 1000.0)))

# 设置成85%的训练集和15%的测试集
# 相同的random_state参数可以保证不同方法间的数据集是相同的
(train_ri, test_ri, train_ril, test_ril) = train_test_split(raw_images, labels, test_size=0.2, random_state=42)
(train_hi, test_hi, train_hil, test_hil) = train_test_split(his_image, labels, test_size=0.2, random_state=42)
