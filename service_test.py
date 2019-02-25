import torch
import cv2
import numpy as np
import time
from learn_code.model import CNN,AngleCNN
import requests
import base64
from app import app_start

face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface.xml')
face_alt_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_alt.xml')
face_alt2_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_alt2.xml')
face_default_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
face_extend_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_extended.xml')
eye_cascade = cv2.CascadeClassifier('./cascades/haarcascade_eye.xml')
eyeglasses_cascade = cv2.CascadeClassifier('./cascades/haarcascade_eye_tree_eyeglasses.xml')
smile_cascade = cv2.CascadeClassifier('./cascades/haarcascade_smile.xml')


def toTensor(img):
    assert type(img) == np.ndarray,'the img type is {}, but ndarry expected'.format(type(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255).unsqueeze(0)


def test_img_para():
    net_angle = AngleCNN()
    net_angle.cuda()
    net_angle.load_state_dict(torch.load('net_angle_params.pkl'))
    net_angle.cuda()
    print(net_angle)
    net_face = CNN()
    net_face.cuda()
    net_face.load_state_dict(torch.load('net_face_params.pkl'))
    net_face.cuda()
    print(net_face)
    cap = cv2.VideoCapture(0)
    count = 0
    while True:
        ret, img = cap.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_alt2_cascade.detectMultiScale(gray, 1.3, 5)
        imgs = []
        for (x, y, w, h) in faces:
            img = cv2.rectangle(img, (x-5, y-5), (x + w + 5, y + h + 5), (255, 0, 0), 2)
            img_temp = img[y:y+h,x:x+w]
            img_temp = cv2.resize(img_temp, (100, 100))
            imgs.append(img_temp)
        cv2.imshow("recongnize", img)
        for temp in imgs:
            temp[:, :, 0] = cv2.equalizeHist(temp[:, :, 0])
            temp[:, :, 1] = cv2.equalizeHist(temp[:, :, 1])
            temp[:, :, 2] = cv2.equalizeHist(temp[:, :, 2])
            count = count + 1
            cv2.imwrite("outputdir/out"+ str(count) + ".jpg", temp)
            img = cv2.resize(temp, (100, 100))
            torch_img = toTensor(img)
            torch_img = torch_img.cuda()
            pred_face, last_layer = net_face(torch_img)
            pred_angle, last_layer = net_angle(torch_img)
            pred_face = torch.max(pred_face, 1)[1].cpu()
            pred_angle = torch.max(pred_angle, 1)[1].cpu()
            cv2.imshow('recongnize_local', img)
            print('test_img A:' + str(pred_angle[0]) + ' ID: ' + str(pred_face[0]))
        if cv2.waitKey(200) & 0xFF == ord('q'):
            break


def post_test():
    rul = r'http://192.168.1.101/post_pic'
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        path = 'post.jpg'
        cv2.imwrite(path, img)
        file_obj = open(path, 'rb')
        data_param = {"some_key": "yibeibanzhan", "timestamp": time.time()}
        img_file = {"img": file_obj}
        data_result = requests.post(r'http://127.0.0.1/post_yolo_pic', data_param, files=img_file)
        print(data_result.text)
        img = cv2.imread('process_img_rectangle.jpg', cv2.IMREAD_COLOR)
        cv2.imshow('show', img)
        output_img = cv2.imread('process_img_classify.jpg', cv2.IMREAD_COLOR)
        cv2.imshow('show_out', output_img)
        eq_img = cv2.imread('process_img_classify_equlize.jpg', cv2.IMREAD_COLOR)
        cv2.imshow('classify_iput', eq_img)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    cap.release()


def main():
    post_test()


if __name__ == "__main__":
    main()


