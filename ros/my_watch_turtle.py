import cv2
import requests
import time


class Watch_Turtle():
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

    def post(self):
        ret, img = self.cap.read()
        cv2.imshow('cap',img)
        path = 'post.jpg'
        cv2.imwrite(path, img)
        file_obj = open(path, 'rb')
        data_param = {"some_key": "yibeibanzhan", "timestamp": time.time()}
        img_file = {"img": file_obj}
        data_result = requests.post(r'http://192.168.1.101/post_pic', data_param, files=img_file)
        return data_result.text

    def time_operate(self):
        result_josn = self.post()
        print(result_josn)
        file = open("index_cut.txt", "w")
        file.write(result_josn)
        file.close()


if __name__ == "__main__":
    wtach_turtle = Watch_Turtle()
    while True:
        wtach_turtle.time_operate()
        if cv2.waitKey(250) & 0xFF == ord('q'):
            break