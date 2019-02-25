import rospy
from geometry_msgs.msg import Twist
from math import radians

import requests
import json

url = 'http://192.168.1.102:5000/'
r = requests.get(url)

print("Status code:", r.status_code)

request_dict = r.json()

# angle = request_dict['camera1']['angle']
distance = request_dict['camera1']['length']


class TestOne():
    def __init__(self):
        # initiliaze
        rospy.init_node('testone', anonymous=False)

        # What to do you ctrl + c    
        rospy.on_shutdown(self.shutdown)

        self.cmd_vel = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size=10)

        # 5 HZ
        r = rospy.Rate(5);

        # create two different Twist() variables.  One for moving forward.  One for turning 45 degrees.

        # let's go forward at 0.2 m/s
        move_cmd = Twist()
        move_cmd.linear.x = 0.1
        # by default angular.z is 0 so setting this isn't required

        # let's turn at 45 deg/s
        turn_cmd = Twist()
        turn_cmd.linear.x = 0
        turn_cmd.angular.z = radians(90);  # 45 deg/s in radians/s

        # two keep drawing squares.  Go forward for 2 seconds (10 x 5 HZ) then turn for 2 second

        while not rospy.is_shutdown():
            # go forward 0.4 m (2 seconds * 0.2 m / seconds)
            rospy.loginfo("Going Straight")
            for x in range(0, int(50 * float(distance) / 1.4)):
                self.cmd_vel.publish(move_cmd)
                r.sleep()
            # turn 90 degrees
            rospy.loginfo("Turning")
            for x in range(0, 7):
                self.cmd_vel.publish(turn_cmd)
                r.sleep()
            # for x in range(0,int(50*float(distance)/1.4)-10):
            for x in range(0, int(float(distance / 1.4)) - 10):
                self.cmd_vel.publish(move_cmd)
                r.sleep()

            self.cmd_vel.publish(Twist())
            rospy.sleep(1)
            break

    def shutdown(self):
        # stop turtlebot
        rospy.loginfo("Stop Drawing Squares")
        self.cmd_vel.publish(Twist())
        rospy.sleep(1)


"""
if __name__ == '__main__':
    try:
	distance = input("provide the distance of target:")
        TestOne()
    except:
        rospy.loginfo("node terminated.")
"""
