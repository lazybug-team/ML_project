import os
import cv2

data_path = r'plant-seedlings-classification'
train_path = os.path.join(data_path, 'train')
test_path = os.path.join(data_path, 'test')


def resize(image):
    image = cv2.resize(image, (224, 224))
    return image


# 图像均衡化，分三通道分别均衡化再合一
def equalize(image):
    b, g, r = cv2.split(image)
    # 依次均衡化
    b = cv2.equalizeHist(b)
    g = cv2.equalizeHist(g)
    r = cv2.equalizeHist(r)
    # 结合成一个图像
    equ_img = cv2.merge((b, g, r))

    return equ_img


# 提取图片中绿色（叶子）的部分
def extractGreen(image):
    lower_green = np.array([35, 43, 46], dtype="uint8")  # 绿色下限
    upper_green = np.array([90, 255, 255], dtype="uint8")  # 绿色上限

    # 高斯滤波
    img_blur = cv2.GaussianBlur(image, (3, 3), 0)
    img_blur = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
    # 根据阈值找到对应颜色，二值化
    mask = cv2.inRange(img_blur, lower_green, upper_green)

    # 掩膜函数
    output = cv2.bitwise_and(image, image, mask=mask)

    return output


def get_green_train(path):
    if os.path.exists('train'):
        return
    os.mkdir("train")
    img_classes = os.listdir(path)

    train_list = []
    eval_list = []

    label = 0
    cnt = 0
    for img_class in img_classes:

        img_class_path = os.path.join(path, img_class)
        imgs = os.listdir(img_class_path)
        for img in imgs:
            img_path = os.path.join(img_class_path, img)
            output_img_path = os.path.join(os.path.join('test', img_class), img)
            cv2.imwrite(output_img_path, extractGreen(equalize(resize(cv2.imread(img_path)))))


get_green_train(train_path)


def get_green_test(path):
    imgs = os.listdir(path)
    if os.path.exists('test'):
        return
    os.mkdir("test")
    for img in imgs:
        img_path = os.path.join(path, img)
        output_img_path = os.path.join('test', img)
        print(output_img_path)
        cv2.imwrite(output_img_path, extractGreen(equalize(resize(cv2.imread(img_path)))))


get_green_test(test_path)
