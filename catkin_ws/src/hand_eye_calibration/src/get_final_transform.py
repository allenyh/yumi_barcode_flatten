import tf
import tf.transformations as tfm
import json
import numpy as np

tran_cam_optical_to_cam_link = [0.015, -0.000, 0.000]
quat_cam_optical_to_cam_link = [0.503, -0.497, 0.500, 0.500]

data = {}
with open('../result/extrinsics.json', 'r') as f:
    data = json.load(f)

ext = data['extrinsics']
tran_base_to_cam_optical = ext[0:3]
quat_base_to_cam_optical = ext[3:]

euler_cam_optical_to_cam_link = tfm.euler_from_quaternion(quat_cam_optical_to_cam_link)
tf_cam_optical_to_cam_link = tfm.compose_matrix(translate=tran_cam_optical_to_cam_link,
                                                angles=euler_cam_optical_to_cam_link)
euler_base_to_cam_optical = tfm.euler_from_quaternion(quat_base_to_cam_optical)
tf_base_to_cam_optical = tfm.compose_matrix(translate=tran_base_to_cam_optical,
                                            angles=euler_base_to_cam_optical)
tf_base_to_cam_link = np.dot(tf_base_to_cam_optical, tf_cam_optical_to_cam_link)
quat_base_to_cam_link = tfm.quaternion_from_matrix(tf_base_to_cam_link)
tran_base_to_cam_link = tf_base_to_cam_link[0:3, 3]

print("trans: {}, quat: {}".format(tran_base_to_cam_link, quat_base_to_cam_link))