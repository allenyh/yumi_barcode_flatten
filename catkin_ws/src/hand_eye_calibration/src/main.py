#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image, PointCloud2
from std_srvs.srv import Trigger, TriggerResponse
from cv_bridge import CvBridge, CvBridgeError

from charuco_info import CharucoProcessor, CharucoCornerPos
from camera_calibration import CameraCalibrator
from utils import ROWS, COLS, delete_old_calib_files, get_pose, dump_data
import message_filters


class HandEyeCalibration:
    def __init__(self):
        self.processor = CharucoProcessor()
        self.bridge = CvBridge()
        self.mark_count = 0

        image_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        pc_sub = message_filters.Subscriber('/camera/depth_registered/points', PointCloud2)
        sub_sync = message_filters.TimeSynchronizer([image_sub, pc_sub], 10)
        sub_sync.registerCallback(self.corner_detect_cb)

        self.detection_pub = rospy.Publisher("/charuco_detection", Image, queue_size=1)
        rospy.Service("/mark_pose", Trigger, self.mark_pose)
        rospy.Service("calculate_error", Trigger, self.calculate_error)

    def corner_detect_cb(self, img_msg, pc_msg):
        image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        self.processor.update_pc_data(pc_msg)
        result = self.processor.detect_corners(image)
        ros_msg_result = self.bridge.cv2_to_imgmsg(result)
        self.detection_pub.publish(ros_msg_result)

    def mark_pose(self, req):
        res = TriggerResponse()
        trans, quat = get_pose()
        pose_3d = CharucoCornerPos(trans, quat)
        pose_3d_cam, ids = self.processor.get_3d_pose()
        if pose_3d_cam is not None and len(pose_3d_cam) == 10:
            dump_data(pose_3d, pose_3d_cam, ids)
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

    def calculate_error(self, req):
        res = TriggerResponse()
        calibrator = CameraCalibrator()
        error = calibrator.calibrate()
        res.success = True
        res.message = "Successfully mark current pose, current calibration error: {}".format(error)
        return res

if __name__ == '__main__':
    rospy.init_node('hand_eye_calibration_node')
    node = HandEyeCalibration()
    rospy.spin()



