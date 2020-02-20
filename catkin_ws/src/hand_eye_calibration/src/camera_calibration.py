import json
import copy
from numpy import array as npa
import numpy as np
from scipy.optimize import leastsq
import cv2
import tf.transformations as tfm
from utils import rod_to_quat, matrix_from_xyzquat, make_sure_path_exists, \
    get_ext_init_guess, data_base, data_file_name, ext_file_name


class Program(object):
    def __init__(self, point3Ds_arm, point3Ds_cam, x0):
        self.point3Ds_arm = point3Ds_arm
        self.point3Ds_cam = point3Ds_cam
        self.x0 = x0
        self.error_diff = 0.0

    def obj_func(self, cam):
        tvec = cam[0:3]  # x,y,z
        rvec = cam[3:6]  # rodrigues

        rot = matrix_from_xyzquat(tvec, rod_to_quat(rvec))[0:3, 0:3]

        # project
        point3Ds_p = []
        for p in self.point3Ds_arm:
            point3D_p = np.dot(rot, p)
            point3D_p[0] += tvec[0]
            point3D_p[1] += tvec[1]
            point3D_p[2] += tvec[2]
            point3Ds_p.append(point3D_p)

        diff = npa(point3Ds_p, dtype=np.float) - \
               npa(self.point3Ds_cam, dtype=np.float)
        diff = diff.flatten(1)
        res = np.linalg.norm(diff)
        self.error_diff = res / len(self.point3Ds_arm)
        print(self.error_diff)
        return diff

    def run(self):

        res_1 = leastsq(self.obj_func, self.x0)
        trans = res_1[0][0:3]
        rod = res_1[0][3:6]
        q = rod_to_quat(rod)
        print('pose', list(trans) + list(q))
        # tf from robot to cam
        transform = tfm.concatenate_matrices(
            tfm.translation_matrix(trans), tfm.quaternion_matrix(q))
        # tf from cam to robot
        inversed_transform = tfm.inverse_matrix(transform)
        translation = tfm.translation_from_matrix(inversed_transform)
        quaternion = tfm.quaternion_from_matrix(inversed_transform)
        pose = translation.tolist() + quaternion.tolist()
        print('matrix:\n', np.linalg.inv(matrix_from_xyzquat(pose)))
        return list(translation) + list(quaternion), self.error_diff


class CameraCalibrator:
    def save_ext(self, cam):
        data = {"extrinsics": cam}
        with open(data_base + ext_file_name, 'w') as data_file:
            json.dump(data, data_file)

    def calibrate(self):
        with open(data_base + data_file_name) as data_file:
            data = json.load(data_file)

        point3d_arm = [data[str(d)]["corner3d_arm"][0:3] for d in data]
        point3d_cam = [data[str(d)]["corner3d_cam"] for d in data]

        ext = get_ext_init_guess(copy.deepcopy(point3d_cam), copy.deepcopy(point3d_arm))

        # Find final extrinsics and report 2d error
        p = Program(point3d_arm, point3d_cam, ext)
        cam, error_diff = p.run()

        self.save_ext(cam)

        return error_diff
