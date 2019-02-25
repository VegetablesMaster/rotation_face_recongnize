

def post_test():

    cap = cv2.VideoCapture(1)
    while True:
        ret, img = cap.read()
        cv2.imshow('cap',img)
        path = 'post.jpg'
        cv2.imwrite(path, img)
        file_obj = open(path, 'rb')
        data_param = {"some_key": "yibeibanzhan", "timestamp": time.time()}
        img_file = {"img": file_obj}
        data_result = requests.post(r'http://192.168.1.101/post_pic', data_param, files=img_file)
        print(data_result.text)
        return data_result.text
        # if cv2.waitKey(200) & 0xFF == ord('q'):
        #     break