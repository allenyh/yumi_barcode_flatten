"""
Contains class that allows reading and writing stable pose data.
Run this file with the following command: python stp_file.py MIN_PROB DIR_PATH

MIN_PROB -- Minimum Probability of a given pose being realized.
DIR_PATH -- Path to directory containing .obj files to be converted to .stp files (e.g. ~/obj_folder/)

Author: Nikhil Sharma
"""

import os
import sys
import IPython
import math
import mesh
import numpy as np
import stable_poses as st
import obj_file
import stable_pose_class as spc
import similarity_tf as stf
import tfx

class StablePoseFile:
    """
    A StablePoseFile contains stable pose data for meshes located
    as the same directory as it.
    """

    def read(self, filename):
        """
        Reads in data from input filename and returns a list of StablePose objects for the mesh corresponding to the file.

        filename -- path to the file to be processed
        """
        print "Reading file: " + filename
        f, stable_poses = open(filename, "r"), []
        data = [line.split() for line in f]
        for i in range(len(data)):
            if len(data[i]) > 0 and data[i][0] == "p":
                p = float(data[i][1])
                r = [[data[i+1][1], data[i+1][2], data[i+1][3]], [data[i+2][0], data[i+2][1], data[i+2][2]], [data[i+3][0], data[i+3][1], data[i+3][2]]]
                r = np.array(r).astype(np.float64)
                x0 = np.array([data[i+4][1], data[i+4][2], data[i+4][3]]).astype(np.float64)
                stable_poses.append(spc.StablePose(p, r, x0))
        return stable_poses

    def write(self):
        """
        Writes stable pose data for meshes in the input directory to an stp file.

        path -- path to directory containing meshes to be converted to .stp
        """

        min_prob_str = sys.argv[1]
        min_prob = float(min_prob_str)
        mesh_index = 1

        mesh_files = [filename for filename in os.listdir(sys.argv[2]) if filename[-4:] == ".obj"]
        for filename in mesh_files:
            print "Writing file: " + sys.argv[2] + "/" + filename
            ob = obj_file.ObjFile(sys.argv[2] + "/" + filename)
            mesh = ob.read()
            mesh.remove_unreferenced_vertices()

            prob_mapping, cv_hull = st.compute_stable_poses(mesh), mesh.convex_hull()
            R_list = []
            for face, p in prob_mapping.items():
                if p >= min_prob:
                    vertices = [cv_hull.vertices()[i] for i in face]
                    basis = st.compute_basis(vertices, cv_hull)
                    R_list.append([p, basis])
            self.write_mesh_stable_poses(mesh, filename, min_prob)
    
    def write_mesh_stable_poses(self, mesh, filename, min_prob=0, vis=False):
        prob_mapping, cv_hull = st.compute_stable_poses(mesh), mesh.convex_hull()

        R_list = []
        for face, p in prob_mapping.items():
            if p >= min_prob:
                x0 = np.array(cv_hull.vertices()[face[0]])
                R_list.append([p, st.compute_basis([cv_hull.vertices()[i] for i in face], cv_hull), x0])

        if vis:
            print 'P', R_list[0][0]
            mv.figure()
            mesh.visualize()
            mv.axes()

            mv.figure()
            cv_hull_tf = cv_hull.transform(stf.SimilarityTransform3D(tfx.transform(R_list[0][1], np.zeros(3))))
            cv_hull_tf.visualize()
            mv.axes()
            mv.show()

        f = open(filename[:-4] + ".stp", "w")
        f.write("#############################################################\n")
        f.write("# STP file generated by UC Berkeley Automation Sciences Lab #\n")
        f.write("#                                                           #\n")
        f.write("# Num Poses: %d" %len(R_list))
        for _ in range(46 - len(str(len(R_list)))):
            f.write(" ")
        f.write(" #\n")
        f.write("# Min Probability: %s" %str(min_prob))
        for _ in range(40 - len(str(min_prob))):
            f.write(" ")
        f.write(" #\n")
        f.write("#                                                           #\n")
        f.write("#############################################################\n")
        f.write("\n")

        # adding R matrices to .stp file
        pose_index = 1
        for i in range(len(R_list)):
            f.write("p %f\n" %R_list[i][0])
            f.write("r %f %f %f\n" %(R_list[i][1][0][0], R_list[i][1][0][1], R_list[i][1][0][2]))
            f.write("  %f %f %f\n" %(R_list[i][1][1][0], R_list[i][1][1][1], R_list[i][1][1][2]))
            f.write("  %f %f %f\n" %(R_list[i][1][2][0], R_list[i][1][2][1], R_list[i][1][2][2]))
            f.write("x0 %f %f %f\n" %(R_list[i][2][0], R_list[i][2][1], R_list[i][2][2]))
        f.write("\n\n")


if __name__ == '__main__':
    mesh_filename = sys.argv[1]
    of = obj_file.ObjFile(mesh_filename)
    m = of.read()
    stp = StablePoseFile()
    stp.write_mesh_stable_poses(m, mesh_filename)

    stp.write()
    for filename in os.listdir("."):
        if filename[-4:] == ".stp":
            pose = stp.read("./" + filename)
            for el in pose:
                print(el.p, el.r)
