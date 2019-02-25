import rospy
from geometry_msgs.msg import Twist

import requests
import json

url = 'http://192.168.1.102:5000/'
r = requests.get(url)

print("Status code:", r.status_code)

request_dict = r.json()

# angle = request_dict['camera1']['angle']
distance = request_dict['camera1']['length']


class GoForWard():
    def __init__(self):
        # initiliaze
        rospy.init_node('GoForward', anonymous=False)

        # tell user how to stop TurtleBot
        rospy.loginfo("To stop TurtleBot CTRL + C")

        # What function to call when you ctrl + c    
        rospy.on_shutdown(self.shutdown)

        # Create a publisher which can "talk" to TurtleBot and tell it to move
        # Tip: You may need to change cmd_vel_mux/input/navi to /cmd_vel if you're not using TurtleBot2
        self.cmd_vel = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size=10)

        # TurtleBot will stop if we don't keep telling it to move.  How often should we tell it to move? 10 HZ
        r = rospy.Rate(10);

        # Twist is a datatype for velocity
        move_cmd = Twist()
        # let's go forward at 0.2 m/s
        move_cmd.linear.x = 0.1
        # let's turn at 0 radians/s
        move_cmd.angular.z = 0

        # as long as you haven't ctrl + c keeping doing...
        while not rospy.is_shutdown():
            rospy.loginfo("Going Straight")
            # num = distance * 100
            for x in range(0, int(100 * float(distance))):
                self.cmd_vel.publish(move_cmd)
                r.sleep()
            self.cmd_vel.publish(Twist())
            rospy.sleep(1)
            break

    def shutdown(self):
        # stop turtlebot
        rospy.loginfo("Stop TurtleBot")
        # a default Twist has linear.x of 0 and angular.z of 0.  So it'll stop TurtleBot
        self.cmd_vel.publish(Twist())
        # sleep just makes sure TurtleBot receives the stop command prior to shutting down the script
        rospy.sleep(1)


"""
if __name__ == '__main__':
    try:
	distance = input("provide the distance of target:")
        GoForWard()
    except:
        rospy.loginfo("GoForward node terminated.")
"""
