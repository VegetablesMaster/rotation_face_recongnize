import requests
import time
import cv2
import json
import os

yolo_enviroment_url = r'http://192.168.1.101/post_environment'
yolo_face_url = r'http://192.168.1.101/post_yolo_pic'
cap_rul = r'http://192.168.1.106/capture'
pwm_url = r'http://192.168.1.105/'



def mac_say(word):
    os.system("say " + word)


def cap_esp():
    path = "temp_pic/process_img.jpg"
    while True:
        time.sleep(0.2)
        r = requests.get(cap_rul)
        with open(path, "wb") as f:
            f.write(r.content)
            print('esp_camera ok!')
        file_obj = open(path, 'rb')
        data_param = {"some_key": "yibeibanzhan", "timestamp": time.time()}
        img_file = {"img": file_obj}
        data_result = requests.post(yolo_face_url, data_param, files=img_file)
        # print(data_result.text)
        r = json.loads(data_result.text)
        if r["camera"]:
            print(r["camera"][0]["location"][0])
            print(r["camera"][0]["location"][1])
            print(r["camera"][0]["length"])

            y_index = int(180 - r["camera"][0]["location"][0] / 15)
            payload = {'pwm_angle': y_index}
            try:
                r = requests.get("http://192.168.1.105/", params=payload)
                print(r.content)
            except:
                print("ardunio disconect")


def cap_local():
    cap = cv2.VideoCapture(0)
    while True:
        time.sleep(0.2)
        ret, img = cap.read()
        path = 'temp_pic/post.jpg'
        cv2.imwrite(path, img)
        file_obj = open(path, 'rb')
        data_param = {"timestamp": time.time()}
        img_file = {"img": file_obj}
        data_result = requests.post(yolo_face_url, data_param, files=img_file)
        # print(data_result.text)
        r = json.loads(data_result.text)
        # print(r["camera"])
        if r["camera"]:
            print(r["camera"][0]["location"][0])
            y_index = int(180 - r["camera"][0]["location"][0] / 15)
            payload = {'pwm_angle': y_index}
            try:
                r = requests.get(pwm_url, params=payload)
                print(r.content)
            except:
                print("ardunio disconect")


def post_environment_test():
    cap = cv2.VideoCapture(0)
    person_count_sum = 0
    dog_count_sum = 0
    bicycle_count_sum = 0
    car_count_sum = 0
    motorbikem_count_sum = 0
    bus_count_sum = 0
    trafficlight_count_sum = 0
    firehydrant_count_sum = 0
    bench_count_sum = 0
    horse_count_sum = 0
    umbrella_count_sum = 0
    backpack_count_sum = 0
    bottle_count_sum = 0
    while True:
        time.sleep(0.2)
        ret, img = cap.read()
        path = 'temp_pic/post.jpg'
        cv2.imwrite(path, img)
        file_obj = open(path, 'rb')
        img_file = {"img": file_obj}
        data_result = requests.post(yolo_enviroment_url, files=img_file)
        print(data_result.text)
        object_json = json.loads(data_result.text)

        person_count = 0
        dog_count = 0
        bicycle_count = 0
        car_count = 0
        motorbikem_count = 0
        bus_count = 0
        trafficlight_count = 0
        firehydrant_count = 0
        bench_count = 0
        horse_count = 0
        umbrella_count = 0
        backpack_count = 0
        bottle_count = 0

        for object in object_json:
            if object['name'] == 'person':
                person_count = person_count + 1
                person_count_sum = person_count_sum + 1
            if object['name'] == 'dog':
                dog_count = dog_count + 1
                dog_count = dog_count + 1
            if object['name'] == 'bicycle':
                bicycle_count = bicycle_count + 1
                bicycle_count_sum = bicycle_count_sum + 1
            if object['name'] == 'car':
                car_count = car_count + 1
                car_count_sum = car_count_sum + 1
            if object['name'] == 'motorbikem':
                motorbikem_count = motorbikem_count + 1
                motorbikem_count_sum = motorbikem_count_sum + 1
            if object['name'] == 'bus':
                bus_count = bus_count + 1
                bus_count_sum = bus_count_sum + 1
            if object['name'] == 'traffic light':
                trafficlight_count = trafficlight_count + 1
                trafficlight_count_sum = trafficlight_count_sum + 1
            if object['name'] == 'fire hydrant':
                firehydrant_count = firehydrant_count + 1
                firehydrant_count_sum = firehydrant_count_sum + 1
            if object['name'] == 'bench':
                bench_count = bench_count + 1
                bench_count_sum = bench_count_sum + 1
            if object['name'] == 'horse':
                horse_count = horse_count + 1
                horse_count_sum = horse_count_sum + 1
            if object['name'] == 'umbrella':
                umbrella_count = umbrella_count + 1
                umbrella_count_sum = umbrella_count_sum + 1
            if object['name'] == 'backpack':
                backpack_count = backpack_count + 1
                backpack_count_sum = backpack_count_sum + 1
            if object['name'] == 'bottle':
                bottle_count = bottle_count + 1
                bottle_count_sum = bottle_count_sum + 1

        if person_count != 0:
            if person_count == 1:
                if bicycle_count:
                    mac_say('你前面有人骑自行车来了')
                else:
                    mac_say('你前面来人了！')
            else:
                mac_say('你前同时来了' + str(person_count) + '个人！')
        if dog_count != 0:
            if dog_count == 1:
                mac_say('当心点，前面来了一条狗！')
            else:
                mac_say('哇，你前面来了一群狗，大概有' + str(dog_count) + '条这么多')
        if car_count:
            mac_say('你前面有辆车，当心点')
        if trafficlight_count:
            mac_say('你走到路口了，要过马路的话小心点')


if __name__ == "__main__":
    post_environment_test()


