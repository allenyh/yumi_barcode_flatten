import numpy as np
from numpy import  array as npa
import rospy
import os
import json
import shutil
import errno
import cv2
import tf.transformations as tfm
from yumipy_service_bridge.srv import GetPose, GetPoseResponse

#Charuco Board parameters
ROWS = 6
COLS = 3
SQUARE_LENGTH = 0.02
MARKER_LENGTH = 0.016
# The center position of charuco board w.r.t. link_6
charuco_center_x = 0.0055
charuco_center_y = 0
charuco_center_z = 0.085

#Place to save data
data_base = "/home/developer/yumi_barcode_flatten/catkin_ws/src/hand_eye_calibration/result/"
data_file_name = "data.json"
ext_file_name = "extrinsics.json"

# Service to get link6 cart. pose
robot_pose_ser = rospy.ServiceProxy("/get_pose", GetPose)


def get_ext_init_guess(point3d_cam, point3d_arm):
    """
    Get extrinsic matrix initial guess
    reference: http://post.queensu.ca/~sdb2/PAPERS/PAMI-3DLS-1987.pdf
    """

    u_point3d_cam = np.mean(point3d_cam, axis=0)
    u_point3d_arm = np.mean(point3d_arm, axis=0)

    H = np.matmul((point3d_arm - u_point3d_arm).T, point3d_cam - u_point3d_cam)

    U, _, V = np.linalg.svd(H)

    X = np.matmul(V.T, U.T)

    if not abs(np.linalg.det(X) - 1) < 1e-6:
        V = V.T
        V[2, 2] = -V[2, 2]
        X = np.matmul(V, U.T)

    R = X
    T = u_point3d_arm - np.matmul(R, u_point3d_cam)

    extrinsics = np.identity(4)
    extrinsics[0:3, 0:3] = R
    extrinsics[0:3, 3] = T.flatten()

    invmat = np.linalg.inv(np.array(extrinsics).reshape((4, 4)))
    rod = mat_to_rod(invmat)
    trans = [invmat[0, 3], invmat[1, 3], invmat[2, 3]]
    return trans + rod


def get_pose():
    robot_pose = robot_pose_ser(arm='left')
    trans = robot_pose.pose[0: 3]
    quat = robot_pose.pose[3:]
    return trans, quat


def dump_data(pose_3d_arm, pose_3d_cam, ids):
    data = {}
    with open(data_base + data_file_name, "r") as data_file:
        data = json.load(data_file)
    data_index = len(data)
    for i, id_ in enumerate(ids):
        corner_pose = pose_3d_arm.getCorner(i)
        data[data_index+i] = {
            "corner3d_arm": corner_pose,
            "corner3d_cam": pose_3d_cam[i]
        }
    with open(data_base + data_file_name, "w") as data_file:
        json.dump(data, data_file)


def mat_to_rod(rotmat):
    dst, jacobian = cv2.Rodrigues(rotmat[0:3][:, 0:3])
    return dst.T.tolist()[0]


def rod_to_quat(r):
    # q = [qx, qy, qz, qw]
    rotmat, jacobian = cv2.Rodrigues(npa(r))
    rotmat = np.append(rotmat, [[0, 0, 0]], 0)
    rotmat = np.append(rotmat, [[0], [0], [0], [1]], 1)
    q = tfm.quaternion_from_matrix(rotmat)
    return q.tolist()


def matrix_from_xyzquat(arg1, arg2=None):
    if arg2 is not None:
        translate = arg1
        quaternion = arg2
    else:
        translate = arg1[0:3]
        quaternion = arg1[3:7]

    return np.dot(tfm.compose_matrix(translate=translate),
                  tfm.quaternion_matrix(quaternion))


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            pass


def delete_old_calib_files():
    try:
        shutil.rmtree(data_base + "/camera_calib")
    except AttributeError:
        pass
