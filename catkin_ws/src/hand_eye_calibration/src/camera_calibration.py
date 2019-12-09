import json
import copy
from numpy import array as npa
import numpy as np
from scipy.optimize import leastsq
import cv2
import tf.transformations as tfm
from utils import rod_to_quat, matrix_from_xyzquat, make_sure_path_exists, \
    get_int, get_ext_init_guess, data_base, data_file_name, ext_file_name


class Program(object):
    def __init__(self, point3Ds, point2Ds, x0, x0_int):
        self.point3Ds = point3Ds
        self.point2Ds = point2Ds
        self.x0 = x0
        self.x0_int = x0_int
        self.error_diff = 0.0

    def obj_func(self, cam):

        #fx, fy = self.x0_int[0], self.x0_int[1]
        #cx, cy = self.x0_int[2], self.x0_int[3]
        distCoeffs = (0.0, 0.0, 0.0, 0.0, 0.0)
        tvec = cam[0:3]  # x,y,z
        rvec = cam[3:6]  # rodrigues

        # project
        cameraMatrix = self.x0_int
        point2Ds_p, jacobian = cv2.projectPoints(
            npa(self.point3Ds, dtype=np.float), rvec, tvec, cameraMatrix, distCoeffs)
        point2Ds_pp = [list(p[0]) for p in point2Ds_p]
        diff = npa(point2Ds_pp, dtype=np.float) - \
               npa(self.point2Ds, dtype=np.float)
        diff = diff.flatten(1)
        res = np.linalg.norm(diff)
        self.error_diff = res / len(self.point3Ds)
        print(self.error_diff)
        return diff

    def run(self):

        res_1 = leastsq(self.obj_func, self.x0)
        trans = res_1[0][0:3]
        rod = res_1[0][3:6]
        q = rod_to_quat(rod)
        print('pose', list(trans) + list(q))

        transform = tfm.concatenate_matrices(
            tfm.translation_matrix(trans), tfm.quaternion_matrix(q))
        inversed_transform = tfm.inverse_matrix(transform)
        translation = tfm.translation_from_matrix(inversed_transform)
        quaternion = tfm.quaternion_from_matrix(inversed_transform)
        pose = translation.tolist() + quaternion.tolist()
        print('matrix:\n', np.linalg.inv(matrix_from_xyzquat(pose)))
        return list(trans) + list(q), self.error_diff


class CameraCalibrator:
    def save_ext(self, cam):
        data = {"extrinsics": cam}
        with open(data_base + ext_file_name, 'w') as data_file:
            json.dump(data, data_file)

    def calibrate(self):
        with open(data_base + data_file_name) as data_file:
            data = json.load(data_file)

        point3d = [data[str(d)]["corner3d"][0:3] for d in data]
        point2d = [data[str(d)]["corner2d"] for d in data]

        int = get_int()
        ext = get_ext_init_guess(copy.deepcopy(point2d), copy.deepcopy(point3d))

        # Find final extrinsics and report 2d error
        p = Program(point3d, point2d, ext, int)
        cam, error_diff = p.run()
        print("cam")
        print(cam)

        self.save_ext(cam)

        return error_diff
