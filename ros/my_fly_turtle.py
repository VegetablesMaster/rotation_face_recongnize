import rospy
from geometry_msgs.msg import Twist
import json
import time
import threading
from math import radians


class Fly_turtle():
    def __init__(self):
        # initiliaze
        rospy.init_node('Fly_turtle', anonymous=False)
        rospy.loginfo("To stop TurtleBot CTRL + C")
        rospy.on_shutdown(self.shutdown)
        self.cmd_vel = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size=10)
        self.r = rospy.Rate(10)
        self.move_cmd = Twist()
        self.para = {'time': time.time()}

    def shutdown(self):
        # stop turtlebot
        rospy.loginfo("Stop")
        self.cmd_vel.publish(Twist())
        rospy.sleep(1)

    def clear(self):
        self.move_cmd.linear.x = 0
        self.move_cmd.angular.z = 0

    def move_now(self, vec, rad,test_flag = True):
        self.move_cmd.linear.x = vec
        self.move_cmd.angular.z = rad
        if test_flag:
            self.move_cmd.linear.x = 0
            self.move_cmd.angular.z = 0
            print("test warning! tertul don not move")
        while True:
            self.cmd_vel.publish(self.move_cmd)
            self.r.sleep()
            file = open("index_cut.txt", "r")
            while True:
                try:
                    para = json.loads(file.read())
                    break
                except:
                    continue
            file.close()
            if vec != 0.0 or rad != 0.0:
                print('moving!!', vec, rad)
            if para['time'] != self.para['time']:
                self.para = para
                print(self.para)
                return

    def go_frover(self):
        while True:
            if 'camera' in self.para:
                vec = (50000.0 - self.para['camera']['length'])/ 30000.0
                rad = (360.0 - self.para['camera']['location']) / 360.0 + 0.45
            else:
                vec = 0.0
                rad = 0.0
            self.move_now(vec, rad)


if __name__ == "__main__":
    fly_turtle = Fly_turtle()
    fly_turtle.go_frover()