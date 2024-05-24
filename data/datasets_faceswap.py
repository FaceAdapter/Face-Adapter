
import numpy as np
import cv2
from PIL import ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True

mean_face_lm5p_256 = np.array([
[(30.2946+8)*2+16, 51.6963*2],  # left eye pupil
[(65.5318+8)*2+16, 51.5014*2],  # right eye pupil
[(48.0252+8)*2+16, 71.7366*2],  # nose tip
[(33.5493+8)*2+16, 92.3655*2],  # left mouth corner
[(62.7299+8)*2+16, 92.2041*2],  # right mouth corner
], dtype=np.float32)



mean_box_lm4p_512 = np.array([
[80, 80], 
[80, 432], 
[432, 432], 
[432, 80],  
], dtype=np.float32)



def get_box_lm4p(pts):
    x1 = np.min(pts[:,0])
    x2 = np.max(pts[:,0])
    y1 = np.min(pts[:,1])
    y2 = np.max(pts[:,1])
    
    x_center = (x1+x2)*0.5
    y_center = (y1+y2)*0.5
    box_size = max(x2-x1, y2-y1)
    
    x1 = x_center-0.5*box_size
    x2 = x_center+0.5*box_size
    y1 = y_center-0.5*box_size
    y2 = y_center+0.5*box_size

    return np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]], dtype=np.float32)


def get_affine_transform(target_face_lm5p, mean_lm5p):
    mat_warp = np.zeros((2,3))
    A = np.zeros((4,4))
    B = np.zeros((4))
    for i in range(5):
        #sa[0][0] += a[i].x*a[i].x + a[i].y*a[i].y;
        A[0][0] += target_face_lm5p[i][0] * target_face_lm5p[i][0] + target_face_lm5p[i][1] * target_face_lm5p[i][1]
        #sa[0][2] += a[i].x;
        A[0][2] += target_face_lm5p[i][0]
        #sa[0][3] += a[i].y;
        A[0][3] += target_face_lm5p[i][1]

        #sb[0] += a[i].x*b[i].x + a[i].y*b[i].y;
        B[0] += target_face_lm5p[i][0] * mean_lm5p[i][0] + target_face_lm5p[i][1] * mean_lm5p[i][1]
        #sb[1] += a[i].x*b[i].y - a[i].y*b[i].x;
        B[1] += target_face_lm5p[i][0] * mean_lm5p[i][1] - target_face_lm5p[i][1] * mean_lm5p[i][0]
        #sb[2] += b[i].x;
        B[2] += mean_lm5p[i][0]
        #sb[3] += b[i].y;
        B[3] += mean_lm5p[i][1]

    #sa[1][1] = sa[0][0];
    A[1][1] = A[0][0]
    #sa[2][1] = sa[1][2] = -sa[0][3];
    A[2][1] = A[1][2] = -A[0][3]
    #sa[3][1] = sa[1][3] = sa[2][0] = sa[0][2];
    A[3][1] = A[1][3] = A[2][0] = A[0][2]
    #sa[2][2] = sa[3][3] = count;
    A[2][2] = A[3][3] = 5
    #sa[3][0] = sa[0][3];
    A[3][0] = A[0][3]

    _, mat23 = cv2.solve(A, B, flags=cv2.DECOMP_SVD)
    mat_warp[0][0] = mat23[0]
    mat_warp[1][1] = mat23[0]
    mat_warp[0][1] = -mat23[1]
    mat_warp[1][0] = mat23[1]
    mat_warp[0][2] = mat23[2]
    mat_warp[1][2] = mat23[3]

    return mat_warp




def transformation_from_points(points1, points2):
    points1 = np.float64(np.matrix([[point[0], point[1]] for point in points1]))
    points2 = np.float64(np.matrix([[point[0], point[1]] for point in points2]))

    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2
    #points2 = np.array(points2)
    #write_pts('pt2.txt', points2)
    U, S, Vt = np.linalg.svd(points1.T * points2)
    R = (U * Vt).T
    return np.array(np.vstack([np.hstack(((s2 / s1) * R,c2.T - (s2 / s1) * R * c1.T)),np.matrix([0., 0., 1.])])[:2])

