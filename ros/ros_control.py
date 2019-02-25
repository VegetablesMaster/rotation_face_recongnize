import requests
import cv2
import time
import json
import rospy
from geometry_msgs.msg import Twist
from math import radians

from test_3 import TestZero
from test_4 import TestOne
from test_5 import GoForWard
from test_6 import TestTwo
from test_7 import TestThree

cap = cv2.VideoCapture(1)
ret, img = cap.read()
cv2.imshow('cap',img)
path = 'post.jpg'
cap.release()
cv2.imwrite(path, img)
file_obj = open(path, 'rb')
data_param = {"some_key": "yibeibanzhan", "timestamp": time.time()}
img_file = {"img": file_obj}
data_result = requests.post(r'http://192.168.1.101/post_pic', data_param, files=img_file)
request_dict = data_result.text.json()
angle = request_dict['camera1']['angle']
distance = request_dict['camera1']['distance']

if angle == 0:
    tz = TestZero()
elif angle == 45:
    to = TestOne()
elif angle == 90:
    go = GoForWard()
elif angle == 135:
    tw = TestTwo()
else:
    tt = TestThree()