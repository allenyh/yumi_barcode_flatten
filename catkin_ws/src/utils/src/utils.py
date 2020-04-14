import cv2
import numpy as np
import math
from copy import deepcopy
import rospy

radian_to_degree = 57.29577951308232
degree_to_radian = 0.017453292519943295


class MaskHandle:
    def __init__(self):
        # parameters with r_  means parameters retrieved from rotated image.
        self.color_img = None
        self.depth_img = None
        self.barcode_mask = None
        self.r_depth_img = None
        self.r_barcode_mask = None
        self.depth_angle = None
        self.depth_box = None
        self.barcode_box = None
        self.r_depth_box = None
        self.r_barcode_box = None
        self.depth_center = None
        self.barcode_center = None
        self.r_depth_center = None
        self.r_barcode_center = None

    def resetAll(self):
        self.color_img = None
        self.depth_img = None
        self.barcode_mask = None
        self.r_depth_img = None
        self.r_barcode_mask = None
        self.depth_angle = None
        self.depth_box = None
        self.barcode_box = None
        self.r_depth_box = None
        self.r_barcode_box = None
        self.depth_center = None
        self.barcode_center = None
        self.r_depth_center = None
        self.r_barcode_center = None

    def set_color_img(self, image):
        self.color_img = image
        cv2.imwrite("./color.jpg", image)

    def set_depth_img(self, image):
        image[:, 0:390] = 0
        image[:, 1070:] = 0
        image[image > 543] = 0
        _, depth_threshold = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        depth_threshold = depth_threshold.astype(np.uint8)
        cv2.imwrite("./depth.jpg", depth_threshold)
        self.depth_img = depth_threshold
        self.depth_angle, self.depth_box, self.depth_center = self.get_bbx(self.depth_img)
        # if self.depth_angle < 0:
        #     self.depth_angle += 90
        rospy.loginfo("depth angle: {}".format(self.depth_angle))
        self.r_depth_img = self.rotate_img(self.depth_img, -self.depth_angle)
        cv2.imwrite("./r_depth.jpg", self.r_depth_img)
        _, self.r_depth_box, self.r_depth_center = self.get_bbx(self.r_depth_img)

    def set_barcode_img(self, image):
        self.barcode_mask = image
        cv2.imwrite("./mask.jpg", image)
        _, self.barcode_box, self.barcode_center = self.get_bbx(image)
        self.r_barcode_mask = self.rotate_img(image, -1*self.depth_angle)
        cv2.imwrite("./r_mask.jpg", self.r_barcode_mask)
        _, self.r_barcode_box, self.r_barcode_center = self.get_bbx(self.r_barcode_mask)

    def check_has_product(self):
        if self.depth_box is None:
            return False
        return True

    def get_bbx(self, image):
        _, contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not len(contours):
            return 0, None, None
        contours = sorted(contours, key=cv2.contourArea)
        cnt = contours[-1]
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        if box is None:
            return 0, None, None
        for i in range(len(box) - 1):
            for j in range(i + 1, len(box)):
                if box[i][1] > box[j][1]:
                    box[i], box[j] = deepcopy(box[j]), deepcopy(box[i])
        top_left = box[0] if box[0][0] < box[1][0] else box[1]
        top_right = box[1] if (top_left == box[0]).all() else box[0]
        x = top_right[0] - top_left[0]
        y = -(top_right[1] - top_left[1])
        angle = np.arctan2(y, x) * radian_to_degree

        box = np.int0(box)
        min_col = box[np.argmin(box[:, 0])][0]
        max_col = box[np.argmax(box[:, 0])][0]
        min_row = box[np.argmin(box[:, 1])][1]
        max_row = box[np.argmax(box[:, 1])][1]
        center = np.array([(max_col + min_col) / 2,
                                   (max_row + min_row) / 2])
        return angle, box, center

    def get_barcode_center(self):
        return self.barcode_center

    # Check if mask_box is inside depth_box
    def check_in_range(self):
        # box = [[x0, y0], [x1, y1....]
        # o-------> x
        # |
        # |
        # v
        # y
        if self.barcode_box is None:
            return False
        depth_col_range = [self.r_depth_box[np.argmin(self.r_depth_box[:, 0], 0)][0],
                           self.r_depth_box[np.argmax(self.r_depth_box[:, 0], 0)][0]]
        depth_row_range = [self.r_depth_box[np.argmin(self.r_depth_box[:, 1], 0)][1],
                           self.r_depth_box[np.argmax(self.r_depth_box[:, 1], 0)][1]]
        barcode_col_range = [self.r_barcode_box[np.argmin(self.r_barcode_box[:, 0], 0)][0],
                             self.r_barcode_box[np.argmax(self.r_barcode_box[:, 0], 0)][0]]
        barcode_row_range = [self.r_barcode_box[np.argmin(self.r_barcode_box[:, 1], 0)][1],
                             self.r_barcode_box[np.argmax(self.r_barcode_box[:, 1], 0)][1]]

        if barcode_row_range[0] in range(depth_row_range[0], depth_row_range[1]) and \
                barcode_row_range[1] in range(depth_row_range[0], depth_row_range[1]) and \
                barcode_col_range[0] in range(depth_col_range[0], depth_col_range[1]) and \
                barcode_col_range[1] in range(depth_col_range[0], depth_col_range[1]):
            return True

        return False

    def apply_mask(self, matrix, mask, fill_value):
        masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
        return masked.filled()

    def apply_threshold(self, matrix, low_value, high_value):
        low_mask = matrix < low_value
        matrix = self.apply_mask(matrix, low_mask, low_value)

        high_mask = matrix > high_value
        matrix = self.apply_mask(matrix, high_mask, high_value)

        return matrix

    def color_balance(self, img, percent):
        assert img.shape[2] == 3
        assert 0 < percent < 100

        half_percent = percent / 200.0

        channels = cv2.split(img)

        out_channels = []
        for idx, channel in enumerate(channels):
            assert len(channel.shape) == 2
            # find the low and high precentile values (based on the input percentile)
            height, width = channel.shape
            vec_size = width * height
            flat = channel.reshape(vec_size)
            assert len(flat.shape) == 1

            flat = np.sort(flat)

            n_cols = flat.shape[0]

            low_val = flat[int(math.floor(n_cols * half_percent))]
            high_val = flat[int(math.ceil(n_cols * (1.0 - half_percent)))]

            # saturate below the low percentile and above the high percentile
            thresholded = self.apply_threshold(channel, low_val, high_val)
            # scale the channel
            normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
            out_channels.append(normalized)

        return cv2.merge(out_channels)

    def rotate_img(self, image, angle, borderValue=0):
        if angle < -80:
            return image
        (h, w) = image.shape[:2]
        center = (w / 2, h / 2)
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, m, (w, h), borderValue=(borderValue, borderValue, borderValue))
        return rotated

    def get_barcode_angle(self):
        count = [0, 0]  # horizontal, vertical
        r_image = self.rotate_img(self.color_img, -self.depth_angle, 255)
        cv2.imwrite('./r_color.jpg', r_image)
        crop = r_image[self.r_barcode_box[np.argmin(self.r_barcode_box[:, 1])][1]:
                       self.r_barcode_box[np.argmax(self.r_barcode_box[:, 1])][1],
                       self.r_barcode_box[np.argmin(self.r_barcode_box[:, 0])][0]:
                       self.r_barcode_box[np.argmax(self.r_barcode_box[:, 0])][0]]
        (h, w) = crop.shape[:2]
        center = (h / 2, w / 2)
        length = w if h > w else h
        crop = crop[center[0] - length / 2:center[0] + length / 2, center[1] - length / 2:center[1] + length / 2, :]
        cv2.imwrite("./barcode.jpg", crop)
        bal = self.color_balance(crop, 50)
        cv2.imwrite("./balance.jpg", bal)
        gray = cv2.cvtColor(bal, cv2.COLOR_BGR2GRAY)
        _, thr = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((2, 2), np.uint8)
        erode = cv2.erode(thr, kernel)
        cv2.imwrite("./erode.jpg", erode)
        lines = cv2.HoughLines(erode, 1, np.pi / 180, length / 4 + 1)
        if lines is None:
            return None
        for l in lines:
            degree = l[0, 1] * radian_to_degree
            if (180 >= degree >= 170) or (10 >= degree >= 0):
                count[0] += 1
            elif 100 >= degree >= 80:
                count[1] += 1
        if count[0] > count[1]:
            barcode_angle = 0
        else:
            barcode_angle = 90
        return barcode_angle

    def find_barcode_quadrant(self):
        # check which quadrant the barcode is on the product
        if self.r_barcode_center[0] > self.r_depth_center[0]:
            if self.r_barcode_center[1] < self.r_depth_center[1]:  # first quadrant
                quadrant = 1
            else:
                quadrant = 4
        else:
            if self.r_barcode_center[1] < self.r_depth_center[1]:
                quadrant = 2
            else:
                quadrant = 3
        return quadrant

    def get_single_handle_strategy(self):
        barcode_angle = self.get_barcode_angle()
        if barcode_angle is None:
            return None
        pos = self.barcode_center
        rad = (barcode_angle + self.depth_angle) * degree_to_radian
        return pos, rad

    def get_handle_strategy(self):
        barcode_angle = self.get_barcode_angle()
        if barcode_angle is None:
            return None
        rospy.loginfo("barcode angle {}".format(barcode_angle))
        quadrant = self.find_barcode_quadrant()

        depth_rad = self.depth_angle * degree_to_radian

        if barcode_angle == 0:  # horizontal flatten, decide lengths by col
            barcode_length = abs(self.r_barcode_box[np.argmax(self.r_barcode_box[:, 0])][0] -
                                 self.r_barcode_box[np.argmin(self.r_barcode_box[:, 0])][0])
            if quadrant == 1 or quadrant == 4:
                fix_arm = 'right'
                flatten_arm = 'left'
                side_length = self.r_barcode_box[np.argmin(self.r_barcode_box[:, 0])][0] - \
                              self.r_depth_box[np.argmin(self.r_depth_box[:, 0])][0]
                fix_point = np.array(
                    [self.barcode_center[0] - 0.5 * side_length * np.cos(depth_rad),
                     self.barcode_center[1] + 0.5 * side_length * np.sin(depth_rad)])
                flatten_start_point = np.array([self.barcode_center[0] - 0.5 * barcode_length * np.cos(depth_rad),
                                                self.barcode_center[1] + 0.5 * barcode_length * np.sin(depth_rad)])
                flatten_end_point = np.array([self.barcode_center[0] + 0.5 * barcode_length * np.cos(depth_rad),
                                              self.barcode_center[1] - 0.5 * barcode_length * np.sin(depth_rad)])
            else:
                fix_arm = 'left'
                flatten_arm = 'right'
                side_length = self.r_depth_box[np.argmax(self.r_depth_box[:, 0])][0] - \
                              self.r_barcode_box[np.argmax(self.r_barcode_box[:, 0])][0]
                fix_point = np.array(
                    [self.barcode_center[0] + 0.5 * side_length * np.cos(depth_rad),
                     self.barcode_center[1] - 0.5 * side_length * np.sin(depth_rad)])
                flatten_start_point = np.array([self.barcode_center[0] + 0.5 * barcode_length * np.cos(depth_rad),
                                                self.barcode_center[1] - 0.5 * barcode_length * np.sin(depth_rad)])
                flatten_end_point = np.array([self.barcode_center[0] - 0.5 * barcode_length * np.cos(depth_rad),
                                              self.barcode_center[1] + 0.5 * barcode_length * np.sin(depth_rad)])
        else:  # vertical flatten, decide lengths by row
            barcode_length = abs(self.r_barcode_box[np.argmax(self.r_barcode_box[:, 1])][1] -
                                 self.r_barcode_box[np.argmin(self.r_barcode_box[:, 1])][1])
            if quadrant == 1 or quadrant == 4:
                fix_arm = 'right'
                flatten_arm = 'left'
            else:
                fix_arm = 'left'
                flatten_arm = 'right'
            if quadrant == 1 or quadrant == 2:
                side_length = self.r_depth_box[np.argmax(self.r_depth_box[:, 1])][1] - \
                              self.r_barcode_box[np.argmax(self.r_barcode_box[:, 1])][1]
                fix_point = np.array(
                    [self.barcode_center[0] + 0.5 * side_length * np.sin(depth_rad),
                     self.barcode_center[1] + 0.5 * side_length * np.cos(depth_rad)])
                flatten_start_point = np.array([self.barcode_center[0] + 0.5 * barcode_length * np.sin(depth_rad),
                                                self.barcode_center[1] + 0.5 * barcode_length * np.cos(depth_rad)])
                flatten_end_point = np.array([self.barcode_center[0] - 0.5 * barcode_length * np.sin(depth_rad),
                                              self.barcode_center[1] - 0.5 * barcode_length * np.sin(depth_rad)])
            else:
                side_length = self.r_barcode_box[np.argmin(self.r_barcode_box[:, 1])][1] - \
                              self.r_depth_box[np.argmin(self.r_depth_box[:, 1])][1]
                fix_point = np.array(
                    [self.barcode_center[0] - 0.5 * side_length * np.sin(depth_rad),
                     self.barcode_center[1] - 0.5 * side_length * np.cos(depth_rad)])
                rospy.loginfo("center {}".format(self.barcode_center))
                rospy.loginfo("fix point {}".format(fix_point))
                flatten_start_point = np.array([self.barcode_center[0] - 0.5 * barcode_length * np.sin(depth_rad),
                                                self.barcode_center[1] - 0.5 * barcode_length * np.cos(depth_rad)])
                flatten_end_point = np.array([self.barcode_center[0] + 0.5 * barcode_length * np.sin(depth_rad),
                                              self.barcode_center[1] + 0.5 * barcode_length * np.sin(depth_rad)])
        self.resetAll()
        return fix_arm, flatten_arm, fix_point, flatten_start_point, flatten_end_point
