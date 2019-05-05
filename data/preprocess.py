import numpy as np
import csv
import os
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow.data import TextLineDataset
import random

DATA_PATH = 'fer2013.csv'
TRAIN_PATH = 'train.csv'
TEST_PATH = 'test.csv'
FILTERED_TRAIN_PATH = 'filtered_train.csv'
FILTERED_TEST_PATH = 'filtered_test.csv'
TEMP_PATH = 'temp.csv'

TRAIN_SIZE = 28709
TEST_SIZE = 7179

FILTERED_TRAIN_SIZE = 17611
FILFERED_TEST_SIZE = 4397

img_size = 48
crop_size = 44


def seperate_dataset():
    # 读取原始csv数据
    # 分为训练集和测试集
    readfile = open(DATA_PATH, mode='r')
    # writefile = open(train_data_path, mode='w', newline='')
    writefile = open(TEST_PATH, mode='w', newline='')

    reader = csv.reader(readfile)       # 通过file创建一个reader
    writer = csv.writer(writefile)      # reader是一个二维数组
    header = next(reader)       # 读取源数据标题
    writer.writerow(['emotion', 'pixels', 'usage'])   # 给新数据写入标题行

    rows = []
    for row in reader:
        if row[2] == 'PublicTest' or row[2] == 'PrivateTest':
            rows.append(row)
    writer.writerows(rows)

    readfile.close()
    writefile.close()


def convert_csv_to_jpg(path, isTrain=True):
    # 将csv转换为numpy数组，再通过PIL转换为jpg，进行可视化查看
    readfile = open(path, mode='r')
    reader = csv.reader(readfile)
    header = next(reader)    # 跳过第一行标题
    i = 0
    jpg_path = os.path.abspath('img/')
    if isTrain:
        jpg_path += 'train/'
    else:
        jpg_path += 'test/'
    if not os.path.exists(jpg_path):
        os.makedirs(jpg_path)

    for i in range(5):
        row = next(reader)
        img_string = np.asarray(row[1].split())     # 以空格分割字符串，获得一个list
        img_int = [int(x) for x in img_string]
        img_int = np.reshape(img_int, newshape=[img_size, img_size])    # 转换成48*48的np数组
        img = Image.fromarray(img_int).convert(mode='L')
        img.save(jpg_path + str(i) + '.jpg')
        i += 1

    readfile.close()


def filter_data(src_path, dst_path):
    # 删除无法识别人脸的样本
    # 加载opencv的人脸识别文件
    # 'haarcascade_frontalface_alt' higher accuracy, but slower
    # 'haarcascade_frontalface_default' lower accuracy, but faster and lighter
    detector = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

    # 将有效样本写回.csv文件中
    writefile = open(dst_path, mode='w', newline='')
    writer = csv.writer(writefile)  #
    writer.writerow(['emotion', 'pixels', 'usage'])  # 给新数据写入标题行

    i = 0
    total = 0
    readfile = open(src_path, mode='r')
    reader = csv.reader(readfile)
    next(reader)  # 跳过第一行标题

    for line in reader:
        pixels = line[1].split()  # 像素值
        label = line[0]  # 第一项为标签
        usage = line[2]
        pixels = np.reshape(np.asarray([int(x) for x in pixels]), [img_size, img_size])
        result = crop_face_area(detector, pixels, img_size)
        if result is not None:
            pixels = np.reshape(result, [-1]).tolist()
            writer.writerow([label, ' '.join([str(x) for x in pixels]), usage])
            i += 1
        total += 1

    print("In %d images, %d valid images with faces!" % (total, i))

    writefile.close()
    readfile.close()


def crop_face_area(detector, image, img_size):
    """
    裁剪图像的人脸部分，并resize到img_size尺寸
    :param detector:
    :param image:
    :param img_size:
    :return:
    """
    p_img = Image.fromarray(image).convert(mode='RGB')
    cv_img = cv2.cvtColor(np.asarray(p_img), cv2.COLOR_RGB2GRAY)
    faces = detector.detectMultiScale(
        image=cv_img,
        scaleFactor=1.1,
        minNeighbors=1,
        minSize=(30, 30),
        flags=0
    )
    if len(faces) != 0:
        x, y, w, h = faces[0]
        cv_img = cv2.resize(cv_img[x:x + w, y:y + h], (img_size, img_size))
        return np.asarray(cv_img)
    else:
        return None


def count_lines(path):
    with open(path, mode='r') as f:
        i = 0
        while True:
            line = f.readline()
            if line == '':
                break
            else:
                i += 1
        print('Total row:', i)


def get_number_labels(path):
    labels = {}
    readfile = open(path, mode='r')
    reader = csv.reader(readfile)
    next(reader)  # 跳过第一行标题

    for line in reader:
        label = line[0]  # 第一项为标签
        if label not in labels:
            labels[label] = 0
        labels[label] += 1
    return list(zip(labels.items()))


def visualize_one_picture(path):
    readfile = open(path, mode='r')
    reader = csv.reader(readfile)
    next(reader)  # 跳过第一行标题

    line = next(reader)
    pixels = line[1].split()  # 像素值
    label = line[0]  # 第一项为标签
    usage = line[2]
    pixels = np.reshape(np.asarray([int(x) for x in pixels]), [img_size, img_size])
    p_img = Image.fromarray(pixels).convert(mode='RGB')
    cv_img = cv2.cvtColor(np.asarray(p_img), cv2.COLOR_RGB2GRAY)
    cv2.imshow('Sample', cv_img)
    cv2.waitKey()
    readfile.close()


if __name__ == '__main__':
    # data_enhance(TEST_PATH, ENHANCED_TEST_PATH)
    # data_enhance(TRAIN_PATH, ENHANCED_TRAIN_PATH)
    # data_enhance(FILTERED_TRAIN_PATH, E_F_TRAIN_PATH)
    # data_enhance(FILTERED_TEST_PATH, E_F_TEST_PATH)
    # filter_data(TRAIN_PATH, FILTERED_TRAIN_PATH)
    # filter_data(TEST_PATH, FILTERED_TEST_PATH)
    # count_lines(ENHANCED_TRAIN_PATH)
    # count_lines(ENHANCED_TEST_PATH)
    # count_lines(E_F_TRAIN_PATH)
    # count_lines(E_F_TEST_PATH)

    # sess = tf.Session()
    # with open(FILTERED_TRAIN_PATH, mode='r') as f:
    #     reader = csv.reader(f)
    #     next(reader)
    #     row = next(reader)
    #     img_string = row[1].split()  # 以空格分割字符串，获得一个list
    #     img_int = [int(x) for x in img_string]
    #     img_int = np.reshape(img_int, newshape=[1, img_size, img_size, 1]).astype(np.float32)  # 转换成48*48的np数组
    #     new_images = image_enhance(img_int, crop_size)
    #     i = 0
    #     for new_image in new_images:
    #         new_img = np.reshape(sess.run(new_image), [crop_size, crop_size])
    #         new = Image.fromarray(new_img).convert(mode='L')
    #         new.save('enhance_demo_%d.jpg' % i)
    #         i += 1
    # sess.close()
    # visualize_one_picture(FILTERED_TEST_PATH)
    print(get_number_labels(FILTERED_TRAIN_PATH))