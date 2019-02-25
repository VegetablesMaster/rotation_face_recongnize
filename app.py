from flask import Flask, request
import json
import time
import base64

import torch
import cv2
import numpy as np
from learn_code.model import CNN, AngleCNN1
from learn_code.darknet import configDetect

face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface.xml')
face_alt_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_alt.xml')
face_alt2_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_alt2.xml')
face_default_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
face_extend_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_extended.xml')
eye_cascade = cv2.CascadeClassifier('./cascades/haarcascade_eye.xml')
eyeglasses_cascade = cv2.CascadeClassifier('./cascades/haarcascade_eye_tree_eyeglasses.xml')
smile_cascade = cv2.CascadeClassifier('./cascades/haarcascade_smile.xml')


class AI_service():
    def __init__(self):
        net_angle = AngleCNN1()
        net_angle.cuda()
        net_angle.load_state_dict(torch.load('net_angle_params.pkl'))
        net_angle.cuda()
        print(net_angle)
        net_face = CNN()
        net_face.cuda()
        net_face.load_state_dict(torch.load('net_face_params.pkl'))
        net_face.cuda()
        print(net_face)
        self.net_A = net_angle
        self.net_F = net_face

    def toTensor(self, img):
        assert type(img) == np.ndarray, 'the img type is {}, but ndarry expected'.format(type(img))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose((2, 0, 1)))
        return img.float().div(255).unsqueeze(0)

    def Get_output(self, img):
        torch_img = self.toTensor(img)
        torch_img = torch_img.cuda()
        pred_face, last_layer = self.net_F(torch_img)
        pred_angle, last_layer = self.net_A(torch_img)
        pred_face = torch.max(pred_face, 1)[1].cpu()
        pred_angle = torch.max(pred_angle, 1)[1].cpu()
        return pred_angle * 45, pred_face

    def detect(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_default_cascade.detectMultiScale(gray, 1.5, 5)
        imgs = []
        for (x, y, w, h) in faces:
            img = img[x:x + w, y:y + h, 3]
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            imgs.append(img)


app = Flask(__name__)
ai_service = AI_service()


@app.route('/')
def hello_world():
    date = {
        'camera1': {'id': 0, 'angle': 0, 'location': 0, 'length': 100},
        'camera2': {'id': 0, 'angle': 0, 'location': 0, 'length': 100},
        'camera3': {'id': 0, 'angle': 0, 'location': 0, 'length': 100},
    }
    return json.dumps(date)


@app.route('/take_photo/<username>', methods=['GET', 'POST'])
def take_photo(username):
    hub_path = 'learn_code/video_hub/app_photo/'+str(username)+'/'
    if request.method == 'POST':
        f = request.files['img']
        f.save(hub_path+str(time.time())+'.jpg')
        return 'OK'


@app.route('/post_yolo_pic', methods=['GET', 'POST'])
def post_yolo_pic():
    if request.method == 'POST':
        f = request.files['img']
        f.save('process_img.jpg')
        date = {
            'time': str(time.time()),
        }
        result = configDetect('process_img.jpg')  # TODO :: 提前在darknet.py中配置好yolo所在的环境
        img = cv2.imread('process_img.jpg')
        try:                                                         # 有时候检测不到人脸
            b = [int(num) for num in result[0][2]]
        except IndexError:
            return json.dumps(date)
        tr_point = (int(b[0] - 0.5 * b[2]), int(b[1] - 0.5 * b[3]))  # 在cv2中框出人脸
        lb_point = [int(b[0] + 0.5 * b[2]), int(b[1] + 0.5 * b[3])]
        tr_point = list(map(lambda x: x if (x > 0) else 0, tr_point))  # map 结果需要list
        lb_point[0] = lb_point[0] if (lb_point[0] < img.shape[1]) else img.shape[1]  # 防止越界
        lb_point[1] = lb_point[1] if (lb_point[1] < img.shape[0]) else img.shape[0]
        tr = (tr_point[0], tr_point[1])
        lb = (lb_point[0], lb_point[1])
        cv2.rectangle(img, tr, lb, (255, 0, 0))
        cv2.imwrite('process_img_rectangle.jpg', img)
        output_img = img[tr[1]:lb[1], tr[0]:lb[0]]
        cv2.imwrite('process_img_classify.jpg', output_img)
        img = cv2.imread('process_img_classify.jpg', cv2.IMREAD_COLOR)
        img_process = cv2.resize(img, (100, 100))
        img_process[:, :, 0] = cv2.equalizeHist(img_process[:, :, 0])
        img_process[:, :, 1] = cv2.equalizeHist(img_process[:, :, 1])
        img_process[:, :, 2] = cv2.equalizeHist(img_process[:, :, 2])
        cv2.imwrite('process_img_classify_equlize.jpg', img_process)
        torch_img = ai_service.toTensor(img_process)
        torch_img = torch_img.cuda()
        pred_face, last_layer = ai_service.net_F(torch_img)
        pred_angle, last_layer = ai_service.net_A(torch_img)
        pred_face = torch.max(pred_face, 1)[1].cpu()
        pred_angle = torch.max(pred_angle, 1)[1].cpu()
        print('test_img A:' + str(pred_angle[0]) + ' ID: ' + str(pred_face[0]))
        date['camera'] = {'id': str(pred_face[0]), 'angle': str(pred_angle[0] * 45), 'location': [b[0], b[1]],
                            'length': int(b[2]*b[3])}
        return json.dumps(date)
    else:
        return "True url for post"


@app.route('/post_pic', methods=['GET', 'POST'])
def post_pic():
    if request.method == 'POST':
        f = request.files['img']
        f.save('process_img.jpg')
        img = cv2.imread('process_img.jpg',cv2.IMREAD_COLOR)
        x, y, z = img.shape
        img = cv2.resize(img, (int(720*y/x), 720))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_alt2_cascade.detectMultiScale(gray, 1.3, 5)
        date = {
            'time': str(time.time()),
        }
        for (x, y, w, h) in faces:
            img = cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (255, 0, 0), 2)
            img_process = img[y:y + h, x:x + w]
            img_process = cv2.resize(img_process, (100, 100))
            location = x + w / 2
            area = w * h
            img_process[:, :, 0] = cv2.equalizeHist(img_process[:, :, 0])
            img_process[:, :, 1] = cv2.equalizeHist(img_process[:, :, 1])
            img_process[:, :, 2] = cv2.equalizeHist(img_process[:, :, 2])
            torch_img = ai_service.toTensor(img_process)
            torch_img = torch_img.cuda()
            pred_face, last_layer = ai_service.net_F(torch_img)
            pred_angle, last_layer = ai_service.net_A(torch_img)
            pred_face = torch.max(pred_face, 1)[1].cpu()
            pred_angle = torch.max(pred_angle, 1)[1].cpu()
            print('test_img A:' + str(pred_angle[0]) + ' ID: ' + str(pred_face[0]))
            date['camera'] = {'id': str(pred_face[0]), 'angle': str(pred_angle[0] * 45), 'location': int(location),
                            'length': int(area)}
        cv2.imshow("recongnize", img)
        return json.dumps(date)
    else:
        return "True url for post"


def app_start():
    app.debug = True
    app.run(host='0.0.0.0', port=80)


if __name__ == '__main__':
    app_start()

