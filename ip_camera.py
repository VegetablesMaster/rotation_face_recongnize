import cv2

RTSP_flag = False
if RTSP_flag:
    url1 = 'rtsp://192.168.1.50/11'
    url2 = 'rtsp://192.168.1.100/11'
    url3 = 'rtsp://192.168.1.150/11'
    url4 = 'rtsp://192.168.1.200/11'
else:
    url1 = 0
    url2 = 1
    url3 = 2
    url4 = 3


class Camera_Video():
    def __init__(self, flag_1, flag_2, flag_3, flag_4):
        if flag_1:
            self.cap1 = cv2.VideoCapture(url1)
            if self.cap1.isOpened():
                self.cap1_flag = True
                print("camera one sucessful")
                ret, frame = self.cap1.read()
                cv2.imshow('cap1',frame)
        else:
            self.cap1_flag = False
        if flag_2:
            self.cap2 = cv2.VideoCapture(url2)
            if self.cap2.isOpened():
                self.cap2_flag = True
                print("camera two sucessful")
                ret, frame = self.cap1.read()
                cv2.imshow('cap2',frame)
        else:
            self.cap2_flag = False
        if flag_3:
            self.cap3 = cv2.VideoCapture(url3)
            if self.cap3.isOpened():
                self.cap3_flag = True
                print("camera three sucessful")
                ret, frame = self.cap3.read()
                cv2.imshow('cap3',frame)
        else:
            self.cap3_flag = False

        if flag_4:
            self.cap4 = cv2.VideoCapture(url4)
            if self.cap4.isOpened():
                self.cap4_flag = True
                print("camera four sucessful")
                ret, frame = self.cap4.read()
                cv2.imshow('cap4',frame)
        else:
            self.cap4_flag = False
        cv2.waitKey(100)

    def free_camare(self):
        self.cap1.release()
        self.cap2.release()
        self.cap3.release()
        self.cap4.release()

    def save_videos(self):
        fourc = cv2.VideoWriter_fourcc(*'mp4v')
        if self.cap1_flag:
            out1 = cv2.VideoWriter("out_put1.mp4", fourc, 25.0, (640, 480))
        if self.cap2_flag:
            out2 = cv2.VideoWriter("out_put2.mp4", fourc, 25.0, (640, 480))
        if self.cap3_flag:
            out3 = cv2.VideoWriter("out_put3.mp4", fourc, 25.0, (640, 480))
        if self.cap4_flag:
            out4 = cv2.VideoWriter("out_put4.mp4", fourc, 25.0, (640, 480))
        while True:
            if self.cap1_flag:
                if self.cap1.isOpened():
                    ret, frame = self.cap1.read()
                    frame = cv2.resize(frame, (512, 512))
                    cv2.imshow('cap1', frame)
                    out1.write(frame)
            if self.cap2_flag:
                if self.cap2.isOpened():
                    ret, frame = self.cap2.read()
                    frame = cv2.resize(frame, (512, 512))
                    cv2.imshow('cap2', frame)
                    out2.write(frame)
            if self.cap3_flag:
                if self.cap3.isOpened():
                    ret, frame = self.cap3.read()
                    frame = cv2.resize(frame, (512, 512))
                    cv2.imshow('cap3', frame)
                    out3.write(frame)
            if self.cap4_flag:
                if self.cap4.isOpened():
                    ret, frame = self.cap4.read()
                    frame = cv2.resize(frame, (512, 512))
                    cv2.imshow('cap4', frame)
                    out4.write(frame)
            cv2.waitKey(100)

    def show_videos(self):
        if self.cap1_flag:
            if self.cap1.isOpened():
                ret, frame = self.cap1.read()
                frame = cv2.resize(frame, (640, 480))
                cv2.imshow('cap1', frame)
        if self.cap2_flag:
            if self.cap2.isOpened():
                ret, frame = self.cap2.read()
                frame = cv2.resize(frame, (640, 480))
                cv2.imshow('cap2', frame)
        if self.cap3_flag:
            if self.cap3.isOpened():
                ret, frame = self.cap3.read()
                frame = cv2.resize(frame, (640, 480))
                cv2.imshow('cap3', frame)
        if self.cap4_flag:
            if self.cap4.isOpened():
                ret, frame = self.cap4.read()
                frame = cv2.resize(frame, (640, 480))
                cv2.imshow('cap4', frame)

    def get_all_frames(self):
        frames = []
        if self.cap1_flag:
            if self.cap1.isOpened():
                ret, frame = self.cap1.read()
                frame1 = cv2.resize(frame, (640, 480))
                frames.append({'frame1': frame1})
        if self.cap2_flag:
            if self.cap2.isOpened():
                ret, frame = self.cap2.read()
                frame2 = cv2.resize(frame, (640, 480))
                frames.append({'frame2': frame2})
        if self.cap3_flag:
            if self.cap3.isOpened():
                ret, frame = self.cap3.read()
                frame3 = cv2.resize(frame, (640, 480))
                frames.append({'frame3': frame3})
        if self.cap4_flag:
            if self.cap1.isOpened():
                ret, frame = self.cap1.read()
                frame4 = cv2.resize(frame, (640, 480))
                frames.append({'frame4': frame4})
        return frames


if __name__ == "__main__":
    Camera_App = Camera_Video(True, False, False, False)
    # while True:
    #     Camera_App.show_videos()
    #     if cv2.waitKey(100) & 0xFF == ord('q'):
    #         break
    Camera_App.save_videos()
