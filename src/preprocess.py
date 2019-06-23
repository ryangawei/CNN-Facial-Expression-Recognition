import numpy as np
import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from config import *
import dlib


def convert_csv_to_jpg(csv_path):
    # 加载opencv的人脸识别文件
    # 'haarcascade_frontalface_alt' higher accuracy, but slower
    # 'haarcascade_frontalface_default' lower accuracy, but faster and lighter
    # detector = cv2.CascadeClassifier(haarcascade_frontalface_alt)
    detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor(shape_predictor_68_face_landmarks)

    # 将csv转换为numpy数组，再通过PIL转换为jpg
    data = pd.read_csv(csv_path, encoding="ISO-8859-1")

    train_path_data = {'path': [], 'emotion': []}
    valid_path_data = {'path': [], 'emotion': []}
    test_path_data = {'path': [], 'emotion': []}

    train_landmark = []
    valid_landmark = []
    test_landmark = []

    if not os.path.exists(TRAIN_DIR):
        os.mkdir(TRAIN_DIR)

    if not os.path.exists(TEST_DIR):
        os.mkdir(TEST_DIR)

    if not os.path.exists(VALID_DIR):
        os.mkdir(VALID_DIR)

    total = 0
    for label, pixels, usage in tqdm(zip(data['emotion'], data['pixels'], data['Usage'])):
        img = np.asarray(pixels.split()).astype(np.int32).reshape([img_size, img_size])
        all_faces, all_landmarks = crop_face_area(detector, landmark_predictor, img, img_size)
        if all_faces is None:
            continue
        for img, landmarks in zip(all_faces, all_landmarks):
            img = Image.fromarray(img).convert(mode='L')
            fname = str(total) + '.jpg'
            if usage == 'Training':
                save_path = TRAIN_DIR + '/' + fname
                train_path_data['path'].append(fname)
                train_path_data['emotion'].append(label)
                train_landmark.append(landmarks)
            elif usage == 'PrivateTest':
                save_path = VALID_DIR + '/' + fname
                valid_path_data['path'].append(fname)
                valid_path_data['emotion'].append(label)
                valid_landmark.append(landmarks)
            else:
                save_path = TEST_DIR + '/' + fname
                test_path_data['path'].append(fname)
                test_path_data['emotion'].append(label)
                test_landmark.append(landmarks)
            img.save(save_path)
            total += 1

    train_landmark = np.asarray(train_landmark)
    valid_landmark = np.asarray(valid_landmark)
    test_landmark = np.asarray(test_landmark)

    np.savez(TRAIN_LANDMARK_PATH, landmark=train_landmark)
    np.savez(VALID_LANDMARK_PATH, landmark=valid_landmark)
    np.savez(TEST_LANDMARK_PATH, landmark=test_landmark)

    train_path_data = pd.DataFrame(train_path_data)
    train_path_data.to_pickle(TRAIN_PATH)
    valid_path_data = pd.DataFrame(valid_path_data)
    valid_path_data.to_pickle(VALID_PATH)
    test_path_data = pd.DataFrame(test_path_data)
    test_path_data.to_pickle(TEST_PATH)

    print('Total: {}, training: {}, valid: {}, test: {}'.format(
        total, len(train_path_data), len(valid_path_data), len(test_path_data)))


def crop_face_area(detector, landmark_predictor, image, img_size):
    """
    裁剪图像的人脸部分，并resize到img_size尺寸

    :param detector:
    :param image:
    :param img_size:
    :return: Two numpy arrays containing the area and landmarks of all faces.
    None if no face was detected.
    """
    # p_img = Image.fromarray(image).convert(mode='RGB')
    # cv_img = cv2.cvtColor(np.asarray(p_img), cv2.COLOR_RGB2GRAY)
    # faces = detector.detectMultiScale(
    #     image=cv_img,
    #     scaleFactor=1.1,
    #     minNeighbors=1,
    #     minSize=(30, 30),
    #     flags=0
    # )
    # if len(faces) != 0:
    #     x, y, w, h = faces[0]
    #     cv_img = cv2.resize(cv_img[x:x + w, y:y + h], (img_size, img_size))
    #     return np.asarray(cv_img)
    # else:
    #     return None

    p_img = Image.fromarray(image).convert(mode='RGB')
    cv_img = cv2.cvtColor(np.asarray(p_img), cv2.COLOR_RGB2GRAY)
    faces = detector(cv_img, 1)
    all_landmarks = []
    all_faces = []
    if len(faces) > 0:
        for face in faces:
            shape = landmark_predictor(cv_img, face)
            landmarks = np.ndarray(shape=[68, 2])
            for i in range(68):
                landmarks[i] = (shape.part(i).x, shape.part(i).y)
            all_landmarks.append(landmarks)
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            if x1 < 0:
                x1 = 0
            if x1 > cv_img.shape[1]:
                x1 = cv_img.shape[1]
            if x2 < 0:
                x2 = 0
            if x2 > cv_img.shape[1]:
                x2 = cv_img.shape[1]
            if y1 < 0:
                y1 = 0
            if y1 > cv_img.shape[0]:
                y1 = cv_img.shape[0]
            if y2 < 0:
                y2 = 0
            if y2 > cv_img.shape[0]:
                y2 = cv_img.shape[0]
            img = cv2.resize(cv_img[y1:y2, x1:x2], (img_size, img_size))
            all_faces.append(img)
        return np.asarray(all_faces), np.asarray(all_landmarks)
    else:
        return None, None


def count_lines(csv_path):
    data = pd.read_csv(csv_path, encoding="ISO-8859-1")
    return data.shape[0]


def show_class_distribution(file_path):
    label_num = np.zeros([7], dtype=np.int32)
    data = pd.read_csv(file_path)

    for label in data['emotion']:
        label_num[label] += 1

    rects = plt.bar(class_names, label_num)
    plt.title('Label Distribution')
    plt.xlabel("Label")
    plt.ylabel("Number")

    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha="center", va="bottom")
    plt.show()


if __name__ == '__main__':
    convert_csv_to_jpg(DATA_PATH)
