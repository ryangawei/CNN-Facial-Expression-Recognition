import numpy as np
import csv
import os
from PIL import Image
import cv2
import random

DATA_PATH = 'fer2013.csv'
TRAIN_PATH = 'train.csv'
TEST_PATH = 'test.csv'
VALID_TRAIN_PATH = 'valid_train.csv'
VALID_TEST_PATH = 'valid_test.csv'
TEMP_PATH = 'temp.csv'

TRAIN_SIZE = 28709
TEST_SIZE = 7179

VALID_TRAIN_SIZE = 14980
VALID_TEST_SIZE = 4769


def seperate_dataset():
    # 读取原始csv数据
    # 分为训练集和测试集
    readfile = open(DATA_PATH, mode='r')
    # writefile = open(train_data_path, mode='w', newline='')
    writefile = open(TEST_PATH, mode='w', newline='')

    reader = csv.reader(readfile)       # 通过file创建一个reader
    writer = csv.writer(writefile)      # reader是一个二维数组
    header = next(reader)       # 读取源数据标题
    writer.writerow(['emotion', 'pixels', 'Usage'])   # 给新数据写入标题行

    rows = []
    for row in reader:
        if row[2] == 'PublicTest' or row[2] == 'PrivateTest':
            rows.append(row)
    writer.writerows(rows)

    readfile.close()
    writefile.close()


def convert_csv_to_jpg():
    # 将csv转换为numpy数组，再通过PIL转换为jpg，进行可视化查看
    readfile = open(DATA_PATH, mode='r')
    reader = csv.reader(readfile)
    header = next(reader)    # 跳过第一行标题
    i = 0

    for row in reader:
        img_string = np.asarray(row[1].split())     # 以空格分割字符串，获得一个list
        img_int = [int(x) for x in img_string]
        img_int = np.reshape(img_int, newshape=[48, 48])    # 转换成48*48的np数组
        img = Image.fromarray(img_int).convert(mode='L')
        img.save('fer2013/img/' + str(i) + '.jpg')

        i += 1

    readfile.close()


def filter_data():
    # 删除无法识别人脸的样本
    # 加载opencv的人脸识别文件
    # 'haarcascade_frontalface_alt' higher accuracy, but slower
    # 'haarcascade_frontalface_default' lower accuracy, but faster and lighter
    detector = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

    # (Must)Load the image with absolute path
    # image_path = os.path.abspath('fer2013/train')
    image_path = os.path.abspath('fer2013/train')

    # Traverse all files in 'train'
    i = 0
    total = 0
    for dirpath, dirnames, filenames in os.walk(image_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            image = cv2.imread(file_path)
            # Detect faces
            # Return a list of rectangles
            # default, minNeighbors=3, got 12893 valid faces.
            # default, minNeighbors=2, got 10495 valid faces.
            # alt, minNeighbors=3, got 14891 valid faces.
            faces = detector.detectMultiScale(
                image=image,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(25, 25),
                flags=0
            )
            # for (x, y, w, h) in faces:
            #     cv2.rectangle(image, (x, y), (x + w, y + w), (0, 255, 0), 2)
            # # Show the image and wait for key press
            # cv2.imshow("Face found", image)
            # cv2.waitKey(0)
            if len(faces) == 0:
                os.remove(file_path)
            else:
                i+=len(faces)
            total+=1

    print ("In %d images, %d valid images with faces!" % (total, i))

    # 将有效样本写回.csv文件中
    writefile = open(VALID_TRAIN_PATH, mode='w', newline='')
    writer = csv.writer(writefile)      #
    writer.writerow(['emotion', 'pixels', 'Usage'])   # 给新数据写入标题行

    rows = []
    for dirpath, dirnames, filenames in os.walk('fer2013/train'):
        for filename in filenames:
            emotion = os.path.split(dirpath)[1]
            image = Image.open(os.path.join(dirpath, filename))
            pixellist = np.asarray(image).reshape([-1])
            pixels = ''     # 像素信息以字符串形式写入
            for p in pixellist:
                pixels = pixels + str(p) + ' '
            row = [emotion, pixels, 'Training']
            rows.append(row)
    writer.writerows(rows)
    writefile.close()
