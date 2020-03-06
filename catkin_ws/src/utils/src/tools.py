#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
from std_srvs.srv import Trigger, TriggerResponse
from copy import deepcopy
from nn.predict.srv import GetPrediction, GetPredictionRequest
from utils.srv import GetHandlePoints, GetHandlePointsResponse
import sensor_msgs.point_cloud2 as pc2


class Tools:
    def __init__(self):
        self.nn_srv = rospy.ServiceProxy('~/nn_predict', GetPrediction)
        self.bridge = CvBridge()
        rospy.Service("~/get_handle_points", GetHandlePoints, self.get_handle_points_cb)

    def rotate_image(self, image, angle, center=None, scale=1.0):
        (h, w) = image.shape[:2]

        if center is None:
            center = (w/2, h/2)

        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated_img = cv2.warpAffine(image, M, (w, h))
        return rotated_img

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

    def get_handle_points_cb(self, req):
        res = GetHandlePointsResponse()

        # step1: get color, depth, predict mask image
        color = rospy.wait_for_message('/camera/color/image_raw', Image)
        color = self.bridge.imgmsg_to_cv2(color, 'bgr8')
        depth = rospy.wait_for_message('/camera/aligned_depth_to_color/image_raw', Image)
        depth = self.bridge.imgmsg_to_cv2(depth, '16UC1')
        mask = self.nn_srv().result
        mask = self.bridge.imgmsg_to_cv2(mask, '8UC1')
        pc = rospy.wait_for_message('/camera/depth_registered/points', PointCloud2)

        # step2: process depth image to get product mask
        # Set points at non workspace area to 0
        depth[:, 0:390] = 0
        depth[:, 1070:] = 0
        # filter out table by distance
        depth[depth > 543] = 0
        depth = cv2.erode(depth, None, iterations=4)
        depth = cv2.dilate(depth, None, iterations=4)

        # step3: find the rotation of product
        _, d_thr = cv2.threshold(depth, 127, 255, cv2.THRESH_BINARY)
        d_thr = d_thr.astype(np.uint8)
        depth_angle, depth_box = self.get_bbx(d_thr)

        # step4: crop barcode part in color image
        r_mask = self.rotate_image(mask, -1 * depth_angle)
        r_color = self.rotate_image(color, -1 * depth_angle)
        mask_angle, mask_box = self.get_bbx(r_mask)
        min_col = mask_box[np.argmin(mask_box[:, 0])][0]
        max_col = mask_box[np.argmax(mask_box[:, 0])][0]
        min_row = mask_box[np.argmin(mask_box[:, 1])][1]
        max_row = mask_box[np.argmax(mask_box[:, 1])][1]
        color_crop = r_color[min_row:max_row, min_col:max_col]

        # step5: find the direction of barcode
        width, height = color_crop.shape[:2]
        gray = cv2.cvtColor(color_crop, cv2.COLOR_BGR2GRAY)
        nor = (gray - np.min(gray)) / float((np.max(gray) - np.min(gray))) * 255
        hor_g = np.gradient(nor[width / 2])
        ver_g = np.gradient(nor[:, height / 2])
        count = [0, 0]
        prev = [1.0, 1.0]
        if hor_g[0]
