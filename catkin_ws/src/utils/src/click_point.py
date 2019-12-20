#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PointStamped
from yumipy_service_bridge.srv import GotoPose, GotoPoseRequest

class ClickPoint:
    def __init__(self):
        rospy.Subscriber("/clicked_point", PointStamped, self.handle_point, queue_size=1)
        self.go_pose_srv = rospy.ServiceProxy("/goto_pose", GotoPose)

    def handle_point(self, msg):
        req = GotoPoseRequest()
        req.arm = 'left'
        req.position = [msg.point.x, msg.point.y, msg.point.z]
        req.quaternion = [0, 0, 0, 1]
        ret = self.go_pose_srv(req)
        print(ret)

if __name__ == '__main__':
    rospy.init_node('click_point_node')
    node = ClickPoint()
    rospy.spin()