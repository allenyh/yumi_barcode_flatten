#!/usr/bin/env python

import numpy as np
import cv2
import os
import rospy
import rospkg
from cv_bridge import CvBridge, CvBridgeError
import torch
import torch.nn as nn
from fcn_resnet import build_fcn_resnet
from sensor_msgs.msg import Image
from nn_predict.srv import GetPrediction, GetPredictionResponse

class NNPrediction:
    def __init__(self):
        self.cv_bridge = CvBridge()
        r = rospkg.RosPack()
        path = r.get_path('nn_predict')
        model = rospy.get_param("model", "res_fcn")
        model_file, self.network = self.build_nn(model)

        self.labels = ['background', 'barcode']
        self.use_gpu = torch.cuda.is_available()
        self.num_gpu = list(range(torch.cuda.device_count()))

        if self.use_gpu:
            self.network = self.network.cuda()

        self.network = nn.DataParallel(self.network, device_ids=self.num_gpu)

        state_dict = torch.load(os.path.join(path, "models", model_file))
        self.network.load_state_dict(state_dict)

        # Services
        rospy.Service('~/nn_predict', GetPrediction, self.predict_cb)

        rospy.loginfo('nn predict node ready!')

    def build_nn(self, model):
        network = None
        if model == 'res_fcn':
            model_file = 'FCNs_barcode_batch4_epoch99_RMSprop_lr0.0001.pkl'
            network = build_fcn_resnet()
        # elif model_name == 'res_cafare_fcn':
        #     model_name = 'CARAFE_FCNs_barcode_batch6_epoch49_RMSprop_lr0.0001.pkl'
        #     network = build_carafe_fcn_resnet()
        # elif model_name == 'barcodenet':
        #     model_name = 'BarcodeNets_barcode_batch6_epoch49_RMSprop_lr0.0001.pkl'
        #     network = build_barcodenet()
        return model_file, network

    def predict_cb(self, req):
        img_msg = rospy.wait_for_message('/camera/color/image_raw', Image)
        cv_image = self.cv_bridge.imgmsg_to_cv2(img_msg, "bgr8")
        predict = self.predict(cv_image)
        mask = np.zeros((720, 1280))
        predict = cv2.resize(predict, (960, 720), interpolation=cv2.INTER_NEAREST)
        mask[:, 160:1120] = predict
        mask[mask != 0] = 255
        mask = mask.astype(np.uint8)
        res = GetPredictionResponse()
        res.result = self.cv_bridge.cv2_to_imgmsg(mask, "8UC1")
        return res

    def predict(self, img):
        means = np.array([103.939, 116.779, 123.68]) / 255.
        img = img[:, 160:1120]
        img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_NEAREST)
        img = img / 255.
        img[:, :, 0] -= means[0]
        img[:, :, 1] -= means[1]
        img[:, :, 2] -= means[2]

        x = torch.from_numpy(img).float().permute(2, 0, 1)
        x = x.unsqueeze(0)
        if self.use_gpu:
            x = x.cuda()
        output = self.network(x)
        output = output.data.cpu().numpy()
        _, _, h, w = output.shape
        pred = output.transpose(0, 2, 3, 1).reshape(-1, len(self.labels)).argmax(axis=1).reshape(1, h, w)
        pred = pred[0]
        pred = np.int8(pred)
        return pred

    def onShutdown(self):
        rospy.loginfo("Shutdown.")


if __name__ == '__main__':
    rospy.init_node('nn_prediction_node', anonymous=False)
    nn_prediction = NNPrediction()
    rospy.on_shutdown(nn_prediction.onShutdown)
    rospy.spin()
