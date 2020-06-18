import cv2
import os
import numpy as np


def get_matrix(folder_path, output_path, pattern_size, show_corner):
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    img_points = []
    obj_points = []
    h = w = 0
    for filename in os.listdir(folder_path):
        im = cv2.imread(os.path.join(folder_path, filename))
        h, w = im.shape[:2]
        found, corners = cv2.findChessboardCorners(im, pattern_size)
        if found:
            cv2.drawChessboardCorners(im, pattern_size, corners, found)
            img_points.append(corners)
            obj_points.append(pattern_points)
            if show_corner:
                cv2.imshow(filename, im)
                cv2.waitKey(0)
        else:
            print("Image %s error." % filename)
    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)
    print("RMS:", rms)
    print("camera matrix:\n", camera_matrix)
    print("distortion coefficients:\n", dist_coefs[0])

if __name__ == '__main__':
    get_matrix("../left/", "../leftCamera", (9, 6), 0)