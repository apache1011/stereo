import cv2
import os
from calibrate import get_matrix


def my_undistort(input, camera_matrix, dist_coefs, alpha, output):
    for filename in os.listdir(input):
        im = cv2.imread(os.path.join(input, filename))
        size = im.shape
        if(alpha == 0):
            im = cv2.undistort(im, camera_matrix, dist_coefs)
        else:
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, size[:2], alpha)
            im = cv2.undistort(im, camera_matrix, dist_coefs, newCameraMatrix=new_camera_matrix)
        if not os.path.exists(output):
            os.mkdir(output)
        cv2.imwrite(os.path.join(output, filename), im)


if __name__ == '__main__':
    rms, camera_matrix, dist_coefs, rvecs, tvecs = get_matrix("../left/", (9, 6))
    alpha = 0
    my_undistort("../left/", camera_matrix, dist_coefs, alpha, "undistort"+str(alpha)+"/")
