import os

import cv2
import numpy as np

from learn_code.darknet import cv2_image_out
from local_config import config


def absolute_path_generator(path):  # Generate all path in dataset folder.
    separator = "-"
    for folder, folders, _ in os.walk(path):
        for subfolder in folders:
            subject_path = os.path.join(folder, subfolder)
            key, _ = subfolder.split(separator)
            for image in os.listdir(subject_path):
                absolute_path = os.path.join(subject_path, image)
                yield absolute_path, key


def get_labels_and_faces(pic_dir_path=None):
    if not pic_dir_path:
        pic_dir = os.getcwd() + '\\video_hub\\cv2_faceid'
    else:
        pic_dir = pic_dir_path
    labels, faces = [], []
    for path, key in absolute_path_generator(pic_dir):
        file_data = cv2.resize(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY), (100, 100))
        faces.append(file_data)
        labels.append(int(key))
    return labels, faces


def get_test_data():
    return cv2_image_out('temp.jpg')


class CV2_Recongnizer():
    def __init__(self, load_conf=False):
        self.face_eigen = cv2.face.EigenFaceRecognizer_create()
        self.face_fisher = cv2.face.FisherFaceRecognizer_create()
        self.face_lbph = cv2.face.LBPHFaceRecognizer_create()
        self.Loaded_Flag = False
        if load_conf:
            self.load_weight()

    def load_weight(self):
        self.face_eigen.read(config["model_path"] + "cv2_face_eigen_model.yml")
        self.face_fisher.read(config["model_path"] + "cv2_face_fisher_model.yml")
        self.face_lbph.read(config["model_path"] + "cv2_face_lbph_model.yml")
        self.Loaded_Flag = True

    def train_weight(self, pic_dir_path=None):
        labels, faces = get_labels_and_faces(pic_dir_path)
        self.face_eigen.train(faces, np.array(labels))
        self.face_eigen.save(config["model_path"] + "cv2_face_eigen_model.yml")
        self.face_fisher.train(faces, np.array(labels))
        self.face_fisher.save(config["model_path"] + "cv2_face_fisher_model.yml")
        self.face_lbph.train(faces, np.array(labels))
        self.face_lbph.save(config["model_path"] + "cv2_face_lbph_model.yml")

    def predict_cv2_data(self, img):
        if not self.Loaded_Flag:
            print('Load data weight first!')
            return
        img = cv2.resize(img,(100,100))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pred, c = self.face_eigen.predict(img)
        pred1, c1 = self.face_fisher.predict(img)
        pred2, c2 = self.face_lbph.predict(img)
        result = {'eigen': (pred, c), 'fisher': (pred1, c1), 'lbph': (pred2, c2)}
        return result

    def predict_img_path(self, filename, detect_falg=False):
        if detect_falg:
            img = cv2_image_out(filename)
        else:
            img = cv2.imread(filename)
        self.predict_cv2_data(img)


def train():
    face_recongizer = CV2_Recongnizer()
    face_recongizer.train_weight()


def test():
    face_recongizer = CV2_Recongnizer(load_conf=True)
    out = face_recongizer.predict_img_path('test.jpg',detect_falg=True)
    print(out)

if __name__ == '__main__':
    test()


