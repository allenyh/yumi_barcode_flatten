#!/usr/bin/env python

import rospy
from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse
from cv_bridge import CvBridge, CvBridgeError
import cv2
from sensor_msgs.msg import Image
import os


class YumiMain:
    def __init__(self):
        self.img_count = len(os.listdir('/home/developer/yumi_barcode_flatten/exp_data'))-1
        self.cv_bridge = CvBridge()
        self.flatten_dual_srv = rospy.ServiceProxy("~/flatten_dual", Trigger)
        self.flatten_one_srv = rospy.ServiceProxy("~/flatten_one", Trigger)
        self.scan_barcode_srv = rospy.ServiceProxy("~/read_barcode", Trigger)
        self.exp_file = open('/home/developer/yumi_barcode_flatten/exp_data/barcode_scan.txt', 'a')
        rospy.Service("~/start_dual", Trigger, self.dual_cb)
        rospy.Service("~/start_one", Trigger, self.one_cb)

    def dual_cb(self, req):
        res = TriggerResponse()
        color_msg = rospy.wait_for_message("/camera/color/image_raw", Image)
        color_img = self.cv_bridge.imgmsg_to_cv2(color_msg, "bgr8")
        cv2.imwrite('/home/developer/yumi_barcode_flatten/exp_data/img_'+str(self.img_count)+'.jpg', color_img)
        result = self.scan_barcode_srv(TriggerRequest()).message
        self.exp_file.write('img_'+str(self.img_count)+': '+result+'\n')
        self.img_count += 1
        if result != "ERROR":
            res.success = True
            res.message = "Succeed flattening barcode"
            return res
        redo_count = 0
        while redo_count < 5:
            self.flatten_dual_srv(TriggerRequest())
            result = self.scan_barcode_srv(TriggerRequest()).message
            if result != "ERROR":
                break
            redo_count += 1
        if redo_count == 5:
            res.success = False
            res.message = "Fail flattening barcode"
        else:
            res.success = True
            res.message = "Succeed flattening barcode at {} time(s)".format(redo_count+1)
        return res

    def one_cb(self, req):
        res = TriggerResponse()
        result = self.scan_barcode_srv(TriggerRequest()).message
        if result != "ERROR":
            res.success = True
            res.message = "Succeed flattening barcode"
            return res
        redo_count = 0
        while redo_count < 5:
            self.flatten_one_srv(TriggerRequest())
            result = self.scan_barcode_srv(TriggerRequest()).message
            if result != "ERROR":
                break
            redo_count += 1
        if redo_count == 5:
            res.success = False
            res.message = "Fail flattening barcode"
        else:
            res.success = True
            res.message = "Succeed flattening barcode at {} time(s)".format(redo_count+1)
        return res


if __name__ == '__main__':
    rospy.init_node('yumi_main')
    node = YumiMain()
    rospy.spin()
