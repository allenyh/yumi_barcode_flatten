import tf.transformations as tfm
import numpy as np
import cv2
from cv2 import aruco
from cv_bridge import CvBridge, CvBridgeError
from utils import get_int, charuco_center_x, charuco_center_y, charuco_center_z,\
    ROWS, COLS, SQUARE_LENGTH, MARKER_LENGTH

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)


class CharucoCornerPos:
    def __init__(self, trans, quat):
        self.tf_map_link6 = tfm.compose_matrix(translate=trans, angles=tfm.euler_from_quaternion(quat))
        self.tf_link6_center = tfm.translation_matrix([charuco_center_x, charuco_center_y, charuco_center_z])
        tf_map_center = np.dot(self.tf_map_link6, self.tf_link6_center)
        self.trans_map_center = tf_map_center[0:3, 3].tolist()
        self.q_map_center = tfm.quaternion_from_matrix(tf_map_center).tolist()

    def getCorner(self, corner_idx):
        num_corners = (ROWS-1) * (COLS-1)
        y_bias = -(corner_idx*2/num_corners)
        if y_bias == 0:
            y_bias = 1
        z_bias = -(corner_idx % (ROWS-1) - (ROWS-2)/2)
        tf_link6_corner = tfm.translation_matrix([self.tf_link6_center[0, 3],
                                                  self.tf_link6_center[1, 3] - y_bias * SQUARE_LENGTH/2,
                                                  self.tf_link6_center[2, 3] + z_bias * SQUARE_LENGTH])
        tf_map_corner = np.dot(self.tf_map_link6, tf_link6_corner)
        return tfm.translation_from_matrix(tf_map_corner).tolist()


class CharucoProcessor:
    def __init__(self):
        self.board = aruco.CharucoBoard_create(ROWS, COLS, SQUARE_LENGTH, MARKER_LENGTH, aruco_dict)
        self.bridge = CvBridge()
        self.corners = None
        self.ids = None

    def draw_corners(self, image, corners):
        color_red = (0, 0, 255)
        for i, corner in enumerate(corners):
            image = cv2.circle(image, tuple(corner[0]), 3, color_red, -1)
            image = cv2.putText(image, str(i+1), tuple(corner[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, color_red, 1, cv2.LINE_AA)
        return image

    def detect_corners(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict)
        aruco.refineDetectedMarkers(gray, self.board, corners, ids, rejected_img_points)
        if ids is None:
            return image
        ret, c_corners, c_ids = aruco.interpolateCornersCharuco(corners, ids, gray, self.board)
        self.corners = c_corners
        self.ids = c_ids
        if ret:
            return self.draw_corners(image, c_corners)
        else:
            return image
        
    def get_2d_pose(self):
        return self.corners, self.ids
