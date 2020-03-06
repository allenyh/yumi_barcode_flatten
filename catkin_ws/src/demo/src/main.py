#!/usr/bin/env python

import rospy
from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse


class YumiMain:
    def __init__(self):
        self.flatten_dual_srv = rospy.ServiceProxy("~/flatten_dual", Trigger)
        self.scan_barcode_srv = rospy.ServiceProxy("~/read_barcode", Trigger)
        rospy.Service("~/start", Trigger, self.cb)

    def cb(self, req):
        redo_count = 0
        while redo_count < 5:
            self.flatten_dual_srv(TriggerRequest())
            result = self.scan_barcode_srv(TriggerRequest()).message
            if result != "ERROR":
                break
            redo_count += 1
        res = TriggerResponse()
        if redo_count == 5:
            res.success = False
            res.message = "Fail flattening barcode"
        else:
            res.success = True
            res.message = "Succeed flattening barcode"
        return res


if __name__ == '__main__':
    rospy.init_node('yumi_main')
    node = YumiMain()
    rospy.spin()
