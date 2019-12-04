#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger, TriggerResponse
from cv_bridge import CvBridge, CvBridgeError

from charuco_info import CharucoProcessor, CharucoCornerPos
from camera_calibration import CameraCalibrator
from utils import ROWS, COLS, delete_old_calib_files, get_pose, dump_data


class HandEyeCalibration:
    def __init__(self):
        self.processor = CharucoProcessor()
        self.bridge = CvBridge()
        self.mark_count = 0
        rospy.Subscriber("/camera/color/image_raw", self.corner_detect_cb, queue=1)
        self.detection_pub = ("/charuco_detection", Image)
        rospy.Service("/mark_pose", Trigger, self.mark_pose)

    def corner_detect_cb(self, msg):
        image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        result = self.processor.detect_corners(image)
        ros_msg_result = self.bridge.cv2_to_imgmsg(result)
        self.detection_pub.publish(ros_msg_result)

    def mark_pose(self, req):
        res = TriggerResponse()
        pose_3d = CharucoCornerPos(get_pose())
        pose_2d, ids = self.processor.get_2d_pose()
        if pose_2d != None and len(pose_2d) == 10:
            dump_data(pose_3d, pose_2d, ids)
            res.success = True
            self.mark_count += 1
            res.message = "Successfully mark current pose."
        else:
            res.success = False
            res.message = "Fail mark current pose due to not detecting every feature point."
            return res
        if self.mark_count >= 3:
            calibrator = CameraCalibrator()
            error = calibrator.calibrate()
            res.message = "Successfully mark current pose, current calibration error: {}".format(error)
        return res


if __name__ == '__main__':
    rospy.init_node('hand_eye_calibration_node')
    node = HandEyeCalibration()
    rospy.spin()



