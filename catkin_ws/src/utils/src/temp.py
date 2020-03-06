#!/usr/bin/env python

import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from sensor_msgs.msg import Image
from copy import deepcopy

if __name__ == '__main__':
    rospy.init_node('temp_node')
    b = CvBridge()
    img = rospy.wait_for_message('/camera/color/image_raw', Image)
    img = b.imgmsg_to_cv2(img, 'bgr8')
    cv2.imwrite('./color_img.jpg', img)

    mask = rospy.wait_for_message('/predict_mask', Image)
    mask = b.imgmsg_to_cv2(mask, '8UC1')
    cv2.imwrite('./mask.jpg', mask)

    pre = rospy.wait_for_message('/predict_img', Image)
    pre = b.imgmsg_to_cv2(pre, 'bgr8')
    cv2.imwrite('./predict_img.jpg', pre)

    depth = rospy.wait_for_message('/camera/aligned_depth_to_color/image_raw', Image)
    depth = b.imgmsg_to_cv2(depth, '16UC1')
    d_t = deepcopy(depth)
    m = np.mean(d_t)
    s = np.std(d_t)
    z = (d_t - m) / s * 255
    cv2.imwrite('./depth_img.jpg', z)

    d = deepcopy(depth)
    d[d>545] = 0
    d = cv2.erode(d, None, iterations=4)
    d = cv2.dilate(d, None, iterations=4)
    cv2.imwrite('./ero_and_dil.jpg', d)

    _, d_thr = cv2.threshold(d, 127, 255, cv2.THRESH_BINARY)
    d_thr = d_thr.astype(np.uint8)
    _, cnts, _ = cv2.findContours(d_thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea)
    cnt = cnts[-1]
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cnt_img = cv2.drawContours(img, [box], 0, (0, 0, 255), 1)
    cv2.imwrite('./product.jpg', cnt_img)




