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
ENHANCED_TRAIN_PATH = 'enhanced_train.csv'
ENHANCED_TEST_PATH = 'enhanced_test.csv'

E_F_TRAIN_PATH = 'e_f_train.csv'
E_F_TEST_PATH = 'e_f_test.csv'

E_F_TRAIN_SIZE = 60560
E_F_TEST_SIZE = 15168

TRAIN_SIZE = 28709
TEST_SIZE = 7179
ENHANCED_TRAIN_SIZE = 114836
ENHANCED_TEST_SIZE = 28712

FILTERED_TRAIN_SIZE = 15140
FILFERED_TEST_SIZE = 3792


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

    for row in reader:
        img_string = np.asarray(row[1].split())     # 以空格分割字符串，获得一个list
        img_int = [int(x) for x in img_string]
        img_int = np.reshape(img_int, newshape=[crop_size, crop_size])    # 转换成48*48的np数组
        img = Image.fromarray(img_int).convert(mode='L')
        img.save(jpg_path + str(i) + '.jpg')
        i += 1

    readfile.close()


def data_enhance(src_path, enhanced_path):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        train_dataset = TextLineDataset(src_path).skip(1).batch(3000)
        iterator = train_dataset.make_one_shot_iterator()
        next_element = iterator.get_next()

        writefile = open(enhanced_path, mode='w', newline='')
        writer = csv.writer(writefile)  # reader是一个二维数组
        writer.writerow(['emotion', 'pixels', 'usage'])  # 给新数据写入标题行

        images = []
        labels = []
        usages = []
        batch = 0
        i = 0
        while True:
            try:
                images.clear()
                labels.clear()
                usages.clear()
                lines = sess.run(next_element)
                for line in lines:
                    line_ = line.decode('utf-8').strip().split(',')
                    label = line_[0]  # 第一项为标签
                    pixels = line_[1].split()  # 像素值
                    usage = line_[2]
                    images.append([int(x) for x in pixels])
                    labels.append(int(label))
                    usages.append(usage)
                images_array = np.stack(images)
                images_array = images_array.reshape([-1, img_size, img_size, 1])

                # 进行数据增强
                # 返回形状为[3, batch_size, crop_height, crop_weight, 1]
                new_images = image_enhance(images_array, crop_size=crop_size)
                for new_image in new_images:
                    new_pixels = np.reshape(sess.run(new_image), [-1, crop_size * crop_size])
                    new_pixels = new_pixels.tolist()
                    length = len(new_pixels)
                    for _ in range(length):
                        print('处理第%d批，第%d个样本' % (batch, i))
                        writer.writerow([labels[_], ' '.join([str(x) for x in new_pixels[_]]), usages[_]])
                        i += 1
                batch += 1
            except tf.errors.OutOfRangeError:
                break
        writefile.close()


def image_enhance(image, crop_size):
    """
    对图片进行数据增强
    :param image: shape为[heigth, width, channel]的数组
    :param crop_size: 裁剪尺寸
    :return:
    """
    # 进行随机裁剪
    images_crop = tf.image.random_crop(image, [image.shape[0], crop_size, crop_size, 1])
    # 水平翻转
    image_flip = tf.image.flip_left_right(images_crop)
    # 随机对比度
    image_contrast_1 = tf.image.random_contrast(images_crop, 0.5, 1.5)
    image_contrast_2 = tf.image.random_contrast(image_flip, 0.5, 1.5)

    return images_crop, image_flip, image_contrast_1, image_contrast_2


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
        p_img = Image.fromarray(pixels).convert(mode='RGB')
        cv_img = cv2.cvtColor(np.asarray(p_img), cv2.COLOR_RGB2GRAY)
        # Detect faces
        # Return a list of rectangles
        # default, minNeighbors=3, got 12893 valid faces.
        # default, minNeighbors=2, got 10495 valid faces.
        # alt, minNeighbors=3, got 14891 valid faces.
        faces = detector.detectMultiScale(
            image=cv_img,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(25, 25),
            flags=0
        )
        if len(faces) == 0:
            pass
        else:
            pixels = np.reshape(pixels, [-1]).tolist()
            writer.writerow([label, ' '.join([str(x) for x in pixels]), usage])
            i += len(faces)
        total += 1

    print ("In %d images, %d valid images with faces!" % (total, i))

    writefile.close()
    readfile.close()


if __name__ == '__main__':
    data_enhance(TEST_PATH, ENHANCED_TEST_PATH)


