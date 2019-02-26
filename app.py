from flask import Flask, request
import json
import time
import cv2
import torch
import base64


from learn_code.cv2_Face_Recognize import CV2_Recongnizer
from learn_code.torch_angle_classify import AI_service, face_alt2_cascade
from learn_code.darknet import configDetect

app = Flask(__name__)
ai_service = AI_service()
cv_service = CV2_Recongnizer(load_conf=True)


@app.route('/')
def hello_world():
    date = {
        'camera1': {
            {'id': 0, 'angle': 0, 'location': 0, 'length': 100},
            {'id': 0, 'angle': 0, 'location': 0, 'length': 100}
        },
        'camera3': {
            'id': 0, 'angle': 0, 'location': 0, 'length': 100
        },
    }
    return json.dumps(date)


@app.route('/take_photo/<username>', methods=['GET', 'POST'])
def take_photo(username):
    hub_path = 'learn_code/video_hub/app_photo/' + str(username) + '/'
    if request.method == 'POST':
        f = request.files['img']
        f.save(hub_path + str(time.time()) + '.jpg')
        return 'OK'


@app.route('/post_yolo_pic', methods=['GET', 'POST'])
def post_yolo_pic():
    if request.method == 'POST':
        f = request.files['img']
        f.save('process_img.jpg')
        date = {
            'time': str(time.time()),
        }
        result = configDetect('process_img.jpg')                    # NOTICE :: 提前在darknet.py中配置好yolo所在的环境
        img = cv2.imread('process_img.jpg')
        response_data = []
        for face in result:
            try:  # 有时候检测不到人脸
                b = [int(num) for num in face[2]]
            except IndexError:
                return json.dumps(date)
            img, output_img, img_process, tr = ai_service.process_pic(img, b)
            pred_angle, pred_face = ai_service.get_output(img_process)
            cv_pred_face = cv_service.predict_cv2_data(img_process)
            response_data.append({'id': pred_face, 'angle': pred_angle, 'location': [b[0], b[1]], 'length': int(b[2] * b[3])})

            if app.debug:
                text = 'test_img A:' + str(pred_angle) + ' ID: ' + str(cv_pred_face)
                cv2.putText(img, text, (20,20), cv2.FONT_HERSHEY_PLAIN, 1, color=(0, 255, 0))
                cv2.imwrite('process_img_rectangle.jpg', img)
                cv2.imwrite('process_img_classify.jpg', output_img)
                cv2.imwrite('process_img_classify_equlize.jpg', img_process)
            # end for face in result
        date['camera'] = response_data
        return json.dumps(date)
    else:
        return "True url for post"


@app.route('/post_pic', methods=['GET', 'POST'])
def post_pic():
    if request.method == 'POST':
        f = request.files['img']
        f.save('process_img.jpg')
        img = cv2.imread('process_img.jpg', cv2.IMREAD_COLOR)
        x, y, z = img.shape
        img = cv2.resize(img, (int(720 * y / x), 720))
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
            torch_img = ai_service.totensor(img_process)
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
