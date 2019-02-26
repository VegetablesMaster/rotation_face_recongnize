import torch
import cv2
import numpy as np
import time
import requests

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


if __name__ == "__main__":
    post_test()


