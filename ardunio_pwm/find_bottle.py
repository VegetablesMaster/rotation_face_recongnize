import cv2
import json
import time
import requests
import os
yolo_enviroment_url = r'http://192.168.1.101/post_environment'


def mac_say(word):
    os.system("say "+word)


def find_bottle():
    cap = cv2.VideoCapture(0)
    x_high = 600
    y_high = 500
    x_low = 300
    y_low = 200
    while True:
        time.sleep(0.5)
        ret, img = cap.read()
        path = 'temp_pic/post.jpg'
        cv2.imwrite(path, img)
        file_obj = open(path, 'rb')
        img_file = {"img": file_obj}
        data_result = requests.post(yolo_enviroment_url, files=img_file)
        print(data_result.text)
        object_json = json.loads(data_result.text)

        for object in object_json:
            if object['name'] == 'book':
                # mac_say('你的书！')
                x = object['location'][0]
                y = object['location'][1]
                print(x, y)
                if x > x_high:
                    if y > y_high:
                        mac_say('右下边')
                    elif y < y_low:
                        mac_say('右上边')
                    else:
                        mac_say('右前边')
                elif x < x_low:
                    if y > y_high:
                        mac_say('左下边')
                    elif y < y_low:
                        mac_say('左上边')
                    else:
                        mac_say('左前方')
                else:
                    if y > y_high:
                        mac_say('正下方')
                    elif y < y_low:
                        mac_say('正上方')
                    else:
                        mac_say('正前方')


if __name__ == "__main__":
    find_bottle()