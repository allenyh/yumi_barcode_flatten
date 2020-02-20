#!/usr/bin/env python

import numpy as np
import cv2
import os
import rospy
import rospkg
from cv_bridge import CvBridge, CvBridgeError
import torch
from fcn_resnet import build_fcn_resnet
from sensor_msgs.msg import Image
import message_filters


class FCNPrediction:
    def __init__(self):
        self.cv_bridge = CvBridge()
        r = rospkg.RosPack()
        path = r.get_path('nn_predict')
        model_name = 'FCNs_barcode_batch6_epoch49_RMSprop_lr0.0001.pkl'

        self.labels = ['background', 'barcode']
        self.network = build_fcn_resnet()
        self.use_gpu = torch.cuda.is_available()

        if self.use_gpu:
            self.network = self.network.cuda()

        state_dict = torch.load(os.path.join(path, "models", model_name))
        self.network.load_state_dict(state_dict)

        # Publisher
        self.image_pub = rospy.Publisher("~/predict_img", Image, queue_size=1)
        self.mask_pub = rospy.Publisher("~/predict_mask", Image, queue_size=1)

        # Subscriber
        image_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        sub_sync = message_filters.TimeSynchronizer([image_sub, depth_sub], 10)
        sub_sync.registerCallback(self.callback)

    def callback(self, img_msg, depth_msg):
        cv_image = self.cv_bridge.imgmsg_to_cv2(img_msg, "bgr8")
        img = cv_image.copy()
        predict_mask = self.predict(img)
        _, contours, _ = cv2.findContours(predict_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        img = cv2.drawContours(img, [box], 0, (0, 255, 0), 1)
        msg = Image()
        msg.data = self.cv_bridge.cv2_to_imgmsg(img)
        self.image_pub.publish(msg)

    def predict(self, img):
        means = np.array([103.939, 116.779, 123.68]) / 255.
        img = img[:, 160:1120]
        img = cv2.resize(img, (640, 480))
        img = img / 255.
        img[0] -= means[0]
        img[1] -= means[1]
        img[2] -= means[2]

        x = torch.from_numpy(img).permute(2, 0, 1)
        x = x.unsqueeze(0)
        if self.use_gpu:
            x = x.cuda()
        time = rospy.get_time()
        output = self.network(x)
        output = output.data.cpu().numpy()
        _, _, h, w = output.shape
        pred = output.transpose(0, 2, 3, 1).reshape(-1, len(self.labels)).argmax(axis=1).reshape(1, h, w)
        pred = np.uint8(pred)
        return pred

    def onShutdown(self):
        rospy.loginfo("Shutdown.")


if __name__ == '__main__':
    rospy.init_node('nn_prediction_node', anonymous=False)
    nn_prediction = FCNPrediction()
    rospy.on_shutdown(nn_prediction.onShutdown)
    rospy.spin()
