#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import PointStamped
from yumipy_service_bridge.srv import GotoPose, GotoPoseRequest, GetPose, \
    GetPoseRequest, SetZ, SetZRequest, MoveGripper, MoveGripperRequest, \
    GotoPoseSync, GotoPoseSyncRequest, GotoJoint, GotoJointRequest, \
    YumipyTrigger, YumipyTriggerRequest
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_srvs.srv import Trigger, TriggerResponse
from cv_bridge import CvBridge, CvBridgeError
import cv2
import tf
import tf.transformations as tfm
from nn_predict.srv import GetPrediction, GetPredictionRequest, GetPredictionResponse
from utils import MaskHandle
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
meter_to_mm = 1000.0
radian_to_degree = 57.29577951308232


class Tools:
    def __init__(self):
        self.cv_bridge = CvBridge()
        self.listener = tf.TransformListener()
        self.get_pose_srv = rospy.ServiceProxy("/get_pose", GetPose)
        self.go_pose_plan_srv = rospy.ServiceProxy("/goto_pose_plan", GotoPose)
        self.go_pose_srv = rospy.ServiceProxy("/goto_pose", GotoPose)
        self.go_pose_sync_srv = rospy.ServiceProxy("/goto_pose_sync", GotoPoseSync)
        self.move_gripper_srv = rospy.ServiceProxy("/move_gripper", MoveGripper)
        self.goto_wait_joint_srv = rospy.ServiceProxy("/goto_wait_joint", YumipyTrigger)
        rospy.Service("~/flatten_dual", Trigger, self.flatten_cb)
        rospy.Service("~/flatten_one", Trigger, self.flatten_one_cb)
        self.marker_pub = rospy.Publisher("~/flatten_marker", Marker, queue_size=1)
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
        if strategy is None:
            return TriggerResponse(success=False)
        fix_product_arm, flatten_arm, fix_point, flatten_start_point, flatten_end_point = strategy
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
        print(fix_pose)
        flatten_start_pose = list(poses[1])
        flatten_start_pose = self.transform_pose_to_base_link(flatten_start_pose)
        print(flatten_start_pose)
        flatten_end_pose = list(poses[2])
        flatten_end_pose = self.transform_pose_to_base_link(flatten_end_pose)
        print(flatten_end_pose)
        fix_arm_current_pose = self.get_pose_srv(fix_product_arm).pose
        fix_arm_cur_position = fix_arm_current_pose[0:3]

        flatten_arm_current_pose = self.get_pose_srv(flatten_arm).pose
        flatten_arm_cur_position = flatten_arm_current_pose[0:3]

        self.publish_marker(arms='dual', poses=[flatten_start_pose, flatten_end_pose])

        # Fix product arm go to fix pose
        req = GotoPoseRequest()
        req.wait_for_res = True
        req.arm = fix_product_arm
        req.quat = fix_arm_quat

        req.position = [meter_to_mm * fix_pose[0],
                        meter_to_mm * fix_pose[1],
                        meter_to_mm * fix_arm_cur_position[2]]
        self.go_pose_plan_srv(req)

        req.position = [meter_to_mm*fix_pose[0],
                        meter_to_mm*fix_pose[1],
                        meter_to_mm*fix_pose[2]]
        self.go_pose_plan_srv(req)

        # Faltten arm go to flatten pose
        req.arm = flatten_arm
        req.quat = flatten_arm_quat

        req.position = [meter_to_mm * flatten_start_pose[0],
                        meter_to_mm * flatten_start_pose[1],
                        meter_to_mm * flatten_arm_cur_position[2]]
        self.go_pose_plan_srv(req)

        req.position = [meter_to_mm * flatten_start_pose[0],
                        meter_to_mm * flatten_start_pose[1],
                        meter_to_mm * flatten_start_pose[2]]
        self.go_pose_plan_srv(req)
        # Do flatten action
        req.position = [meter_to_mm * flatten_end_pose[0],
                        meter_to_mm * flatten_end_pose[1],
                        meter_to_mm * flatten_end_pose[2]]
        self.go_pose_plan_srv(req)

        req.arm = flatten_arm
        req.quat = flatten_arm_quat
        req.position = [meter_to_mm * flatten_end_pose[0],
                        meter_to_mm * flatten_end_pose[1],
                        meter_to_mm * flatten_end_pose[2] + 50]
        self.go_pose_plan_srv(req)
        self.goto_wait_joint_srv(req.arm)

        req.arm = fix_product_arm
        req.quat = fix_arm_quat
        req.position = [meter_to_mm * fix_pose[0],
                        meter_to_mm * fix_pose[1],
                        meter_to_mm * fix_pose[2] + 50]
        self.go_pose_plan_srv(req)
        self.goto_wait_joint_srv(req.arm)

        return TriggerResponse()

    def flatten_one_cb(self, req):
        color_msg = rospy.wait_for_message("/camera/color/image_raw", Image)
        color_img = self.cv_bridge.imgmsg_to_cv2(color_msg, "bgr8")
        self.mask_handle.set_color_img(color_img)
        rospy.loginfo("Color image set")
        depth_msg = rospy.wait_for_message("/camera/aligned_depth_to_color/image_raw", Image)
        depth_img = self.cv_bridge.imgmsg_to_cv2(depth_msg, "16UC1")
        self.mask_handle.set_depth_img(depth_img)
        rospy.loginfo("Depth image set")
        if not self.mask_handle.check_has_product():
            rospy.logerr("There's no object detected!!")
            return TriggerResponse(message="No object detected")
        while not self.mask_handle.check_in_range():
            mask_msg = self.predict_srv().result
            mask = self.cv_bridge.imgmsg_to_cv2(mask_msg, "8UC1").astype(np.uint8)
            self.mask_handle.set_barcode_img(mask)
        rospy.loginfo("Barcode mask found")

        strategy = self.mask_handle.get_single_handle_strategy()
        if strategy is None:
            return TriggerResponse(success=False)
        pos, rad = strategy
        uvs = list()
        uvs.append(pos.astype(int).tolist())
        pc = rospy.wait_for_message('/camera/depth_registered/points', PointCloud2)
        poses = list(pc2.read_points(pc, skip_nans=True, field_names=("x", "y", "z"), uvs=uvs))

        pose = self.transform_pose_to_base_link(list(poses[0]))
        quat = tfm.quaternion_from_euler(np.pi, 0, np.pi/2+rad)
        self.publish_marker(arms='one', poses=[pose], quat=quat)
        req = GotoPoseRequest()
        req.quat = [quat[-1], quat[0], quat[1], quat[2]]#[0, 0.7071068, -0.7071068, 0]
        req.arm = 'right'
        if pose[1] > 0:
            req.arm = 'left'

        req.position = [meter_to_mm * pose[0],
                        meter_to_mm * pose[1],
                        meter_to_mm * pose[2] + 50]
        self.go_pose_plan_srv(req)
        rospy.sleep(0.1)

        req.position = [meter_to_mm * pose[0],
                        meter_to_mm * pose[1],
                        meter_to_mm * pose[2]]
        self.go_pose_plan_srv(req)
        rospy.sleep(0.1)

        self.move_gripper_srv(MoveGripperRequest(arm=req.arm, width=0.025))
        rospy.sleep(0.1)

        req.position = [meter_to_mm * pose[0],
                        meter_to_mm * pose[1],
                        meter_to_mm * pose[2] + 50]
        self.go_pose_plan_srv(req)
        rospy.sleep(0.1)

        req.quat = self.arms_quat[req.arm]
        req.position = self.arms_wait_pose[req.arm]
        self.goto_wait_joint_srv(req.arm)
        self.move_gripper_srv(MoveGripperRequest(arm=req.arm, width=0))
        self.mask_handle.resetAll()

        return TriggerResponse()

    def transform_pose_to_base_link(self, pose):
        pose.append(1)
        trans, quat = self.listener.lookupTransform('base_link', 'camera_color_optical_frame', rospy.Time(0))
        euler = tfm.euler_from_quaternion(quat)
        tf = tfm.compose_matrix(translate=trans, angles=euler)
        t_pose = np.dot(tf, pose)[:3]
        return t_pose

    def publish_marker(self, arms, poses, quat=None):
        if quat is None:
            quat = [0, 0, 0, 1]
        marker = Marker()
        marker.header.frame_id = 'base_link'
        marker.header.stamp = rospy.Time.now()
        marker.id = 0
        marker.type = 0
        marker.action = 0
        marker.scale.x = 0.05
        marker.scale.y = 0.01
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0
        marker.color.b = 0
        marker.lifetime = rospy.Duration(5)
        if arms == 'one':
            pose = poses[0]
            marker.pose.position.x = pose[0]
            marker.pose.position.y = pose[1]
            marker.pose.position.z = pose[2]
            marker.pose.orientation.x = quat[0]
            marker.pose.orientation.y = quat[1]
            marker.pose.orientation.z = quat[2]
            marker.pose.orientation.w = quat[3]
        else:
            marker.pose.position.x = poses[0][0]
            marker.pose.position.y = poses[0][1]
            marker.pose.position.z = poses[0][2]
            start_point = Point()
            start_point.x = poses[0][0]
            start_point.y = poses[0][1]
            start_point.z = poses[0][2]
            end_point = Point()
            end_point.x = poses[1][0]
            end_point.y = poses[1][1]
            end_point.z = poses[1][2]
            marker.points.append(start_point)
            marker.points.append(end_point)
        self.marker_pub.publish(marker)


if __name__ == '__main__':
    rospy.init_node('tools_node')
    node = Tools()
    rospy.spin()