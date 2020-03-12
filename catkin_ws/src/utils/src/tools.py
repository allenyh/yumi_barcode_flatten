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
from utils import MaskHandle
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
        self.arms_quat = {'left': [0, 0, -0.9238795, 0.3826834], 'right': [0, 0, 0.9238795, 0.3826834]}
        self.arms_wait_pose = {'left': [356, 200, 100], 'right': [356, -200, 100]}
        self.mask_handle = MaskHandle()

    def flatten_cb(self, req):
        color_msg = rospy.wait_for_message("/camera/color/image_raw", Image)
        color_img = self.cv_bridge.imgmsg_to_cv2(color_msg, "bgr8")
        self.mask_handle.set_color_img(color_img)
        rospy.loginfo("Color image set")
        depth_msg = rospy.wait_for_message("/camera/aligned_depth_to_color/image_raw", Image)
        depth_img = self.cv_bridge.imgmsg_to_cv2(depth_msg, "16UC1")
        self.mask_handle.set_depth_img(depth_img)
        rospy.loginfo("Depth image set")
        # mask_msg = self.predict_srv().result
        # mask = self.cv_bridge.imgmsg_to_cv2(mask_msg, "8UC1").astype(np.uint8)
        # self.mask_handle.set_barcode_img(mask)
        # rospy.loginfo("Barcode mask set")
        if not self.mask_handle.check_has_product():
            rospy.logerr("There's no object detected!!")
            return TriggerResponse(message="No object detected")

        while not self.mask_handle.check_in_range():
            mask_msg = self.predict_srv().result
            mask = self.cv_bridge.imgmsg_to_cv2(mask_msg, "8UC1").astype(np.uint8)
            self.mask_handle.set_barcode_img(mask)
        rospy.loginfo("Barcode mask found")

        # TODO first find out barcode's rotate angle, then find the length of barcode. after that,
        #  find out the flatten start/end point by center + 0.5*length*(cos/sin(angle)).
        strategy = self.mask_handle.get_handle_strategy()
        fix_product_arm, flatten_arm, fix_point, flatten_start_point, flatten_end_point = strategy
        # min_col = mask_box[np.argmin(mask_box[:, 0])][0]
        # max_col = mask_box[np.argmax(mask_box[:, 0])][0]
        # min_row = mask_box[np.argmin(mask_box[:, 1])][1]
        # max_row = mask_box[np.argmax(mask_box[:, 1])][1]
        # mask_center = np.array([(max_col + min_col) / 2, (max_row + min_row) / 2])
        # color_crop = color_img[min_row:max_row, min_col:max_col]
        #
        # barcode_angle = get_barcode_angle(color_crop, depth_angle)
        # rospy.loginfo("Depth angle:{}, Barcode angle:{}".format(depth_angle, barcode_angle/np.pi*180))
        # r_mask = self.rotate_img(mask, barcode_angle)
        # _, r_mask_box = self.get_bbx(mask)
        # barcode_length = r_mask_box[np.argmax(r_mask_box[:, 0])][0] - r_mask_box[np.argmin(r_mask_box[:, 0])][0]
        #
        # if mask_center[0] >= depth_col_center:
        #     fix_product_arm = 'right'
        #     flatten_arm = 'left'
        #     fix_point = np.array([(min_depth_col+min_col)/2, mask_center[1]])
        #     flatten_start_point = np.array([mask_center[0]-0.5*barcode_length*np.cos(barcode_angle),
        #                                     mask_center[1]+0.5*barcode_length*np.sin(barcode_angle)])
        #     flatten_end_point = np.array([mask_center[0]+0.5*barcode_length*np.cos(barcode_angle),
        #                                   mask_center[1]-0.5*barcode_length*np.sin(barcode_angle)])
        # else:
        #     fix_product_arm = 'left'
        #     flatten_arm = 'right'
        #     fix_point = np.array([(max_depth_col + max_col) / 2, mask_center[1]])
        #     flatten_start_point = np.array([mask_center[0]+0.5*barcode_length*np.cos(barcode_angle),
        #                                     mask_center[1]-0.5*barcode_length*np.sin(barcode_angle)])
        #     flatten_end_point = np.array([mask_center[0]-0.5*barcode_length*np.cos(barcode_angle),
        #                                   mask_center[1]+0.5*barcode_length*np.sin(barcode_angle)])
        fix_arm_quat = self.arms_quat[fix_product_arm]
        flatten_arm_quat = self.arms_quat[flatten_arm]
        uvs = list()
        uvs.append(fix_point.astype(int).tolist())
        uvs.append(flatten_start_point.astype(int).tolist())
        uvs.append(flatten_end_point.astype(int).tolist())

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
        # req.position = [meter_to_mm * fix_pose[0],
        #                 meter_to_mm * fix_arm_cur_position[1],
        #                 meter_to_mm * fix_arm_cur_position[2]]
        # self.go_pose_plan_srv(req)
        # rospy.sleep(0.1)

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
        # req.position = [meter_to_mm * flatten_start_pose[0],
        #                 meter_to_mm * flatten_arm_cur_position[1],
        #                 meter_to_mm * flatten_arm_cur_position[2]]
        # self.go_pose_plan_srv(req)
        # rospy.sleep(0.1)

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
        req.position = self.arms_wait_pose[fix_product_arm]
        self.go_pose_plan_srv(req)
        rospy.sleep(0.1)

        req.arm = flatten_arm
        req.quat = flatten_arm_quat
        req.position = [meter_to_mm * flatten_end_pose[0],
                        meter_to_mm * flatten_end_pose[1],
                        meter_to_mm * flatten_end_pose[2]+50]
        self.go_pose_plan_srv(req)
        rospy.sleep(0.1)
        req.position = self.arms_wait_pose[flatten_arm]
        self.go_pose_plan_srv(req)
        rospy.sleep(0.1)

        return TriggerResponse()

    def transform_pose_to_base_link(self, pose):
        pose.append(1)
        trans, quat = self.listener.lookupTransform('base_link', 'camera_color_optical_frame', rospy.Time(0))
        euler = tfm.euler_from_quaternion(quat)
        tf = tfm.compose_matrix(translate=trans, angles=euler)
        t_pose = np.dot(tf, pose)[:3]
        return t_pose


if __name__ == '__main__':
    rospy.init_node('tools_node')
    node = Tools()
    rospy.spin()