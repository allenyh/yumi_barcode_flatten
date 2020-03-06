#!/usr/bin/env python

import rospy
import cv2
import os
import json
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from yumipy_service_bridge.srv import GotoPose, GotoPoseRequest, Trigger, TriggerResponse, TriggerRequest

data_dir = '/home/developer/yumi_barcode_flatten/catkin_ws/src/utils/data/'
img_dir = '/home/developer/yumi_barcode_flatten/catkin_ws/src/utils/data/img/'

meter_to_mm = 1000.0

class DataCollect:
    def __init__(self):
        self.bridge = CvBridge()
        self.img_rgb = None
        self.img_count = len(os.listdir(img_dir))
        self.detect_file = data_dir + 'detect.json'
        self.go_pose_srv = rospy.ServiceProxy("/goto_pose", GotoPose)
        self.goto_wait_joint_srv = rospy.ServiceProxy("/goto_wait_joint", Trigger)
        rospy.Subscriber("/camera/color/image_raw", Image, self.update_rgb, queue_size=1)
        #rospy.Service("/collect_img", Trigger, self.collect)

    def update_rgb(self, msg):
        image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.img_rgb = image

    def save_image(self):
        if self.img_rgb is not None:
            ret = cv2.imwrite(img_dir + "IMG_"+str(self.img_count)+".png", self.img_rgb)
            print("Save image:{}".format(ret))
            self.img_count += 1
        return TriggerResponse()

    def delete_last_image(self):
        imgs = os.listdir(img_dir)
        os.remove(img_dir + "IMG_" + str(len(imgs)-1) + ".png")

    def collect(self):
        raw_input("press enter when ready")
        data = {}
        # Save a new image to img_dir
        with open(self.detect_file, "r") as data_file:
            data = json.load(data_file)
        data_index = len(data)
        self.save_image()
        # Wait for click point msg and move robot
        print("Waiting for clicked_point msg...")
        pose_msg = rospy.wait_for_message("/clicked_point", PointStamped)

        req = GotoPoseRequest()
        req.arm = 'right'
        req.wait_for_res = False
        req.position = [meter_to_mm * pose_msg.point.x, meter_to_mm * pose_msg.point.y,
                        meter_to_mm * pose_msg.point.z + 200]
        req.quat = [0.69168259, -0.72066929, 0.04697844, 0.00199908]
        print("Moving robot to specified pose...")
        count = 0
        while count < 5:
            ret = self.go_pose_srv(req)
            if ret.success:
                break
            else:
                count += 1
        if count == 5:
            self.delete_last_image()
            rospy.logwarn("Fail planning path")
            return
        barcode = raw_input("barcode:")
        print("Detected barcode: {}".format(barcode))
        if len(barcode) == 0:
            data[data_index] = {
                "IMG_"+str(self.img_count-1)+".png": "false"
            }
            print("IMG_{}.png, result:{}".format(str(self.img_count - 1), "false"))
        else:
            data[data_index] = {
                "IMG_" + str(self.img_count - 1) + ".png": "true"
            }
            print("IMG_{}.png, result:{}".format(str(self.img_count - 1), "true"))
        with open(self.detect_file, "w") as data_file:
            json.dump(data, data_file)
        self.goto_wait_joint_srv(TriggerRequest())
        return


if __name__ == '__main__':
    rospy.init_node('data_collect_node')
    node = DataCollect()
    while 1:
        node.collect()
    rospy.spin()
