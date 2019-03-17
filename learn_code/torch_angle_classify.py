import torch
import cv2
import numpy as np
from learn_code.model import CNN, AngleCNN1


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
        self.Id_map = {'0': 'jiaqi', '1': 'leyi', '2': 'zhechen', '3': 'zhangwen'}
        self.Angle_map = {'0': -120, '1': -90, '2': -60, '3': -30, '4': 0, '5': 30, '6': 60, '7': 90,
                          '8': 120, '9': 180}

    def totensor(self, img):
        assert type(img) == np.ndarray, 'the img type is {}, but ndarry expected'.format(type(img))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose((2, 0, 1)))
        return img.float().div(255).unsqueeze(0)

    def get_output(self, img):
        torch_img = self.totensor(img)
        torch_img = torch_img.cuda()
        pred_face, last_layer = self.net_F(torch_img)
        pred_angle, last_layer = self.net_A(torch_img)
        pred_face = torch.max(pred_face, 1)[1].cpu()
        pred_angle = torch.max(pred_angle, 1)[1].cpu()
        return self.Angle_map[str(int(pred_angle))], self.Id_map[str(int(pred_face))]

    def cv2_facedetect(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_default_cascade.detectMultiScale(gray, 1.5, 5)
        imgs = []
        for (x, y, w, h) in faces:
            img = img[x:x + w, y:y + h, 3]
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            imgs.append(img)

    def process_pic(self, img, b):
        tr_point = (int(b[0] - 0.5 * b[2]), int(b[1] - 0.5 * b[3]))  # 在cv2中框出人脸
        lb_point = [int(b[0] + 0.5 * b[2]), int(b[1] + 0.5 * b[3])]
        tr_point = list(map(lambda x: x if (x > 0) else 0, tr_point))  # map 结果需要list
        lb_point[0] = lb_point[0] if (lb_point[0] < img.shape[1]) else img.shape[1]  # 防止越界
        lb_point[1] = lb_point[1] if (lb_point[1] < img.shape[0]) else img.shape[0]
        tr = (tr_point[0], tr_point[1])
        lb = (lb_point[0], lb_point[1])
        cv2.rectangle(img, tr, lb, (255, 0, 0))
        output_img = img[tr[1]:lb[1], tr[0]:lb[0]]
        img_process = cv2.resize(output_img, (100, 100))
        img_process[:, :, 0] = cv2.equalizeHist(img_process[:, :, 0])
        img_process[:, :, 1] = cv2.equalizeHist(img_process[:, :, 1])
        img_process[:, :, 2] = cv2.equalizeHist(img_process[:, :, 2])
        return img, output_img, img_process, tr
