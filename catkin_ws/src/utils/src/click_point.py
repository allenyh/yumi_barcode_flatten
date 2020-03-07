#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PointStamped
from yumipy_service_bridge.srv import GotoPose, GotoPoseRequest, GetPose, GetPoseRequest
from cv_bridge import CvBridge, CvBridgeError
meter_to_mm = 1000.0

class ClickPoint:
    def __init__(self):
        self.cv_bridge = CvBridge()
        rospy.Subscriber("/clicked_point", PointStamped, self.handle_point, queue_size=1)
        self.get_pose_srv = rospy.ServiceProxy("/get_pose", GetPose)
        self.go_pose_plan_srv = rospy.ServiceProxy("/goto_pose_plan", GotoPose)
        self.left_arm_quat = [0, 0, -0.9238795, 0.3826834]
        self.right_arm_quat = [0, 0, 0.9238795, 0.3826834]

    def handle_point(self, msg):
        print(msg.point)
        req = GotoPoseRequest()
        req.arm = 'right'

        current_pose = self.get_pose_srv(req.arm).pose
        c_position = current_pose[0:3]
        c_quat = current_pose[3:]

        req.position = [meter_to_mm*msg.point.x, meter_to_mm * c_position[1], meter_to_mm * c_position[2]]
        req.quat = self.right_arm_quat
        self.go_pose_plan_srv(req)
        rospy.sleep(1)

        req.position = [meter_to_mm * msg.point.x, meter_to_mm * msg.point.y, meter_to_mm * msg.point.z+50]
        req.quat = self.right_arm_quat
        ret = self.go_pose_plan_srv(req)
        rospy.sleep(1)

        req.position = [meter_to_mm * msg.point.x, meter_to_mm * msg.point.y, meter_to_mm * msg.point.z]
        req.quat = self.right_arm_quat
        ret = self.go_pose_plan_srv(req)


if __name__ == '__main__':
    rospy.init_node('click_point_node')
    node = ClickPoint()
    rospy.spin()