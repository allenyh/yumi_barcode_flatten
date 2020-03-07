#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import PointStamped
from yumipy_service_bridge.srv import GotoPose, GotoPoseRequest, GetPose, GetPoseRequest, SetZ, SetZRequest
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_srvs.srv import Trigger, TriggerResponse
from cv_bridge import CvBridge, CvBridgeError
import cv2
import tf
import tf.transformations as tfm
from nn_predict.srv import GetPrediction, GetPredictionRequest, GetPredictionResponse
meter_to_mm = 1000.0
radian_to_degree = 57.29577951308232

class Tools:
    def __init__(self):
        self.cv_bridge = CvBridge()
        self.listener = tf.TransformListener()
        self.get_pose_srv = rospy.ServiceProxy("/get_pose", GetPose)
        self.go_pose_plan_srv = rospy.ServiceProxy("/goto_pose_plan", GotoPose)
        rospy.Service("~/flatten_dual", Trigger, self.flatten_cb)
        self.predict_srv = rospy.ServiceProxy('~/nn_predict', GetPrediction)
        self.left_arm_quat = [0, 0, -0.9238795, 0.3826834]
        self.right_arm_quat = [0, 0, 0.9238795, 0.3826834]

    def flatten_cb(self, req):
        contour_detect = False
        depth_msg = rospy.wait_for_message("/camera/aligned_depth_to_color/image_raw", Image)
        depth_img = self.cv_bridge.imgmsg_to_cv2(depth_msg, "16UC1")
        # filter out table
        depth_img[depth_img>530] = 0
        _, depth_threshold = cv2.threshold(depth_img, 127, 255, cv2.THRESH_BINARY)
        depth_threshold = depth_threshold.astype(np.uint8)
        depth_angle, depth_box = self.get_bbx(depth_threshold)
        if depth_box is -1:
            rospy.logerr("There's no object detected!!")
            return TriggerResponse(message="No object detected")
        min_depth_col = depth_box[np.argmin(depth_box[:, 0])][0]
        max_depth_col = depth_box[np.argmax(depth_box[:, 0])][0]
        depth_col_center = (max_depth_col + min_depth_col) / 2

        while not contour_detect:
            mask_msg = self.predict_srv().result
            mask = self.cv_bridge.imgmsg_to_cv2(mask_msg, "8UC1").astype(np.uint8)
            mask_angle, mask_box = self.get_bbx(mask)
            if mask_box is -1:
                continue
            if self.check_in_range(depth_box, mask_box):
                contour_detect = True

        # TODO first find out barcode's rotate angle, then find the length of barcode. after that,
        #  find out the flatten start/end point by center + 0.5*length*(cos/sin(angle)).
        barcode_roate_angle = 0 # retrieve this by some algorithm
        r_mask = self.rotate_img(mask, barcode_roate_angle)
        _, r_mask_box = self.get_bbx(mask)
        barcode_length = r_mask_box[np.argmax(r_mask_box[:, 0])][0] - r_mask_box[np.argmin(r_mask_box[:, 0])][0]

        min_col = mask_box[np.argmin(mask_box[:, 0])][0]
        max_col = mask_box[np.argmax(mask_box[:, 0])][0]
        min_row = mask_box[np.argmin(mask_box[:, 1])][1]
        max_row = mask_box[np.argmax(mask_box[:, 1])][1]
        mask_center = np.array([(max_col + min_col) / 2, (max_row + min_row) / 2])

        if mask_center[0] >= depth_col_center:
            fix_product_arm = 'right'
            fix_arm_quat = self.right_arm_quat
            flatten_arm = 'left'
            flatten_arm_quat = self.left_arm_quat
            fix_point = np.array([(min_depth_col+min_col)/2, mask_center[1]])
            flatten_start_point = np.array([mask_center-0.5*barcode_length*np.cos(barcode_roate_angle), mask_center[1]])
            flatten_end_point = np.array([mask_center+0.5*barcode_length*np.cos(barcode_roate_angle), mask_center[1]])
        else:
            fix_product_arm = 'left'
            fix_arm_quat = self.left_arm_quat
            flatten_arm = 'right'
            flatten_arm_quat = self.right_arm_quat
            fix_point = np.array([(max_depth_col + max_col) / 2, mask_center[1]])
            flatten_start_point = np.array([mask_center+0.5*barcode_length*np.cos(barcode_roate_angle), mask_center[1]])
            flatten_end_point = np.array([mask_center-0.5*barcode_length*np.cos(barcode_roate_angle), mask_center[1]])

        uvs = list()
        uvs.append(fix_point.tolist())
        uvs.append(flatten_start_point.tolist())
        uvs.append(flatten_end_point.tolist())

        pc = rospy.wait_for_message('/camera/depth_registered/points', PointCloud2)
        poses = list(pc2.read_points(pc, skip_nans=True, field_names=("x", "y", "z"), uvs=uvs))
        fix_pose = list(poses[0])
        fix_pose = self.transform_pose_to_base_link(fix_pose)
        flatten_start_pose = list(poses[1])
        flatten_start_pose = self.transform_pose_to_base_link(flatten_start_pose)
        flatten_end_pose = list(poses[2])
        flatten_end_pose = self.transform_pose_to_base_link(flatten_end_pose)

        fix_arm_current_pose = self.get_pose_srv(fix_product_arm).pose
        fix_arm_cur_position = fix_arm_current_pose[0:3]

        flatten_arm_current_pose = self.get_pose_srv(flatten_arm).pose
        flatten_arm_cur_position = flatten_arm_current_pose[0:3]

        # Fix product arm go to fix pose
        req = GotoPoseRequest()
        req.arm = fix_product_arm
        req.quat = fix_arm_quat
        req.position = [meter_to_mm * fix_pose[0],
                        meter_to_mm * fix_arm_cur_position[1],
                        meter_to_mm * fix_arm_cur_position[2]]
        self.go_pose_plan_srv(req)
        rospy.sleep(0.1)

        req.position = [meter_to_mm * fix_pose[0],
                        meter_to_mm * fix_pose[1],
                        meter_to_mm * fix_arm_cur_position[2]]
        self.go_pose_plan_srv(req)
        rospy.sleep(0.1)

        req.position = [meter_to_mm*fix_pose[0],
                        meter_to_mm*fix_pose[1],
                        meter_to_mm*fix_pose[2]-2]
        self.go_pose_plan_srv(req)
        rospy.sleep(0.1)

        # Faltten arm go to flatten pose
        req.arm = flatten_arm
        req.quat = flatten_arm_quat
        req.position = [meter_to_mm * flatten_start_pose[0],
                        meter_to_mm * flatten_arm_cur_position[1],
                        meter_to_mm * flatten_arm_cur_position[2]]
        self.go_pose_plan_srv(req)
        rospy.sleep(0.1)

        req.position = [meter_to_mm * flatten_start_pose[0],
                        meter_to_mm * flatten_start_pose[1],
                        meter_to_mm * flatten_arm_cur_position[2]]
        self.go_pose_plan_srv(req)
        rospy.sleep(0.1)

        req.position = [meter_to_mm * flatten_start_pose[0],
                        meter_to_mm * flatten_start_pose[1],
                        meter_to_mm * flatten_start_pose[2]]
        self.go_pose_plan_srv(req)
        rospy.sleep(0.1)
        # Do flatten action
        req.position = [meter_to_mm * flatten_end_pose[0],
                        meter_to_mm * flatten_end_pose[1],
                        meter_to_mm * flatten_end_pose[2]]
        self.go_pose_plan_srv(req)
        rospy.sleep(0.1)

        # both arm leave product
        req.arm = fix_product_arm
        req.quat = fix_arm_quat
        req.position = [meter_to_mm * fix_pose[0],
                        meter_to_mm * fix_pose[1],
                        meter_to_mm * fix_pose[2] + 50]
        self.go_pose_plan_srv(req)
        rospy.sleep(0.1)
        req.position = [meter_to_mm * fix_pose[0],
                        meter_to_mm * fix_arm_cur_position[1],
                        meter_to_mm * fix_pose[2] + 50]
        self.go_pose_plan_srv(req)
        rospy.sleep(0.1)

        req.arm = flatten_arm
        req.quat = flatten_arm_quat
        req.position = [meter_to_mm * flatten_end_pose[0],
                        meter_to_mm * flatten_end_pose[1],
                        meter_to_mm * flatten_end_pose[2]+50]
        self.go_pose_plan_srv(req)
        rospy.sleep(0.1)
        req.position = [meter_to_mm * flatten_end_pose[0],
                        meter_to_mm * flatten_arm_cur_position[1],
                        meter_to_mm * flatten_end_pose[2] + 50]
        self.go_pose_plan_srv(req)
        rospy.sleep(0.1)

        return TriggerResponse()

    def rotate_img(self, image, angle):
        (h, w) = image.shape[:2]
        center = (w / 2, h / 2)
        m = cv2.getRotationMatrix2D(center, angle*radian_to_degree, 1.0)
        rotated = cv2.warpAffine(image, m, (w, h))
        return rotated

    def get_bbx(self, image):
        _, contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not len(contours):
            return 0, -1
        contours = sorted(contours, key=cv2.contourArea)
        cnt = contours[-1]
        rect = cv2.minAreaRect(cnt)
        angle = rect[-1]
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        return angle, box

    def check_in_range(self, depth_box, mask_box):
        depth_x_range = [depth_box[np.argmin(depth_box[:, 0], 0)][0], depth_box[np.argmax(depth_box[:, 0], 0)][0]]
        depth_y_range = [depth_box[np.argmin(depth_box[:, 1], 0)][1], depth_box[np.argmax(depth_box[:, 1], 0)][1]]
        mask_x_range = [mask_box[np.argmin(mask_box[:, 0], 0)][0], mask_box[np.argmax(mask_box[:, 0], 0)][0]]
        mask_y_range = [mask_box[np.argmin(mask_box[:, 1], 0)][1], mask_box[np.argmax(mask_box[:, 1], 0)][1]]

        if mask_x_range[0] in range(depth_x_range[0], depth_x_range[1]) and \
            mask_x_range[1] in range(depth_x_range[0], depth_x_range[1]) and \
            mask_y_range[0] in range(depth_y_range[0], depth_y_range[1]) and \
            mask_y_range[1] in range(depth_y_range[0], depth_y_range[1]):
            return True

        return False

    def transform_pose_to_base_link(self, pose):
        pose.append(1)
        trans, quat = self.listener.lookupTransform('base_link', 'camera_color_optical_frame', rospy.Time(0))
        euler = tfm.euler_from_quaternion(quat)
        tf = tfm.compose_matrix(translate=trans, angles=euler)
        t_pose = np.dot(tf, pose)[:3]
        return t_pose


if __name__ == '__main__':
    rospy.init_node('click_point_node')
    node = Tools()
    rospy.spin()