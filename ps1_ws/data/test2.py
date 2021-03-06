import math
import numpy as np

def compute_vanishing_point(points):
    """Computes vanishing point given four points on parallel line.
    Args:
        points: A list of all the points where each row is (x, y). Generally,
            it will contain four points: two for each parallel line.
            You can use any convention you'd like, but our solution uses the
            first two rows as points on the same line and the last
            two rows as points on the same line.
    Returns:
        vanishing_point: The pixel location of the vanishing point.
    """
    # construct (x1, y1) (x2, y2) (x3, y3) (x4, y4)
    x1 = points[0][0]; y1 = points[0][1];
    x2 = points[1][0]; y2 = points[1][1];
    x3 = points[2][0]; y3 = points[2][1];
    x4 = points[3][0]; y4 = points[3][1];

    # slopes
    m1 = (float)(y2 - y1) / (x2 - x1)
    m2 = (float)(y4 - y3) / (x4 - x3)
    # intercepts
    b1 = y2 - m1 * x2
    b2 = y4 - m2 * x4

    # vanishing point coordinates
    x = (b2 - b1) / (m1 - m2)
    y = m1 * ((b2 - b1)/(m1 - m2)) + b1
    vanishing_point = np.array([x, y])
    return vanishing_point


def compute_K_from_vanishing_points(vanishing_points):
    """Compute intrinsic matrix given vanishing points.
    Args:
        vanishing_points: A list of vanishing points.
    Returns:
        K: The intrinsic camera matrix (3x3 matrix).
    """
    # vanishing points used
    v1 = vanishing_points[0]
    v2 = vanishing_points[1]
    v3 = vanishing_points[2]

    # construct constraint matrix A from each pair of vanishing points
    A = np.zeros((3, 3))
    # 1 + 2
    vi = v1
    vj = v2
    A[0] = np.array([(vi[0]*vj[0]+vi[1]*vj[1]), (vi[0]+vj[0]), (vi[1]+vj[1])])

    # 1 + 3
    vi = v1
    vj = v3
    A[1] = np.array([(vi[0]*vj[0]+vi[1]*vj[1]), (vi[0]+vj[0]), (vi[1]+vj[1])])

    # 2 + 3
    vi = v2
    vj = v3
    A[2] = np.array([(vi[0]*vj[0]+vi[1]*vj[1]), (vi[0]+vj[0]), (vi[1]+vj[1])])

    # add one column of ones
    A_ones = np.ones((A.shape[0], 1))
    A = np.hstack((A, A_ones))

    # SVD
    U, s, VT = np.linalg.svd(A)
    w = VT[-1, :]
    omega = np.array([[w[0], 0, w[1]],
                      [0, w[0], w[2]],
                      [w[1], w[2], w[3]]])

    # find K matrix from omega
    KT_inv = np.linalg.cholesky(omega)
    K = np.linalg.inv(KT_inv.T)
    # normalize
    K /= K[2, 2]
    return K

def compute_angle_between_planes(vanishing_pair1, vanishing_pair2, K):
    """Compute angle between planes of the given pairs of vanishing points.
    Args:
        vanishing_pair1: A list of a pair of vanishing points computed from
            lines within the same plane.
        vanishing_pair2: A list of another pair of vanishing points from a
            different plane than vanishing_pair1.
        K: The camera matrix used to take both images.
    Returns:
        angle: The angle in degrees between the planes which the vanishing
            point pair comes from2.
    """
    omega_inv = K.dot(K.T)

    # a set of vanishing points on one plane
    v1 = np.hstack((vanishing_pair1[0], 1))
    v2 = np.hstack((vanishing_pair1[1], 1))

    # another set of vanishing points on the other plane
    v3 = np.hstack((vanishing_pair2[0], 1))
    v4 = np.hstack((vanishing_pair2[1], 1))

    # find two vanishing lines
    L1 = np.cross(v1.T, v2.T)
    L2 = np.cross(v3.T, v4.T)

    # find the angle between planes
    costheta = (L1.T.dot(omega_inv).dot(L2)) / (np.sqrt(L1.T.dot(omega_inv).dot(L1)) * np.sqrt(L2.T.dot(omega_inv).dot(L2)))
    theta = (np.arccos(costheta) / math.pi) * 180

    return theta

def compute_rotation_matrix_between_cameras(vanishing_pts1, vanishing_pts2, K):
    """Compute rotation matrix between two cameras given their vanishing points.
    Args:
        vanishing_pts1: A list of vanishing points in image 1.
        vanishing_pts2: A list of vanishing points in image 2.
        K: The camera matrix used to take both images.
    Returns:
        R: The rotation matrix between camera 1 and camera 2.
    """
    # a set of vanishing points on one image
    v1 = np.hstack((vanishing_pts1[0], 1))
    v2 = np.hstack((vanishing_pts1[1], 1))
    v3 = np.hstack((vanishing_pts1[2], 1))

    # another set of vanishing points on the other image
    v4 = np.hstack((vanishing_pts2[0], 1))
    v5 = np.hstack((vanishing_pts2[1], 1))
    v6 = np.hstack((vanishing_pts2[2], 1))

    # first image vanishing points directions
    d1 = np.linalg.inv(K).dot(v1) / np.linalg.norm(np.linalg.inv(K).dot(v1))
    d2 = np.linalg.inv(K).dot(v2) / np.linalg.norm(np.linalg.inv(K).dot(v2))
    d3 = np.linalg.inv(K).dot(v3) / np.linalg.norm(np.linalg.inv(K).dot(v3))

    print(d1)

    # second image vanishing points directions
    dPrime1 = np.linalg.inv(K).dot(v4) / np.linalg.norm(np.linalg.inv(K).dot(v4))
    dPrime2 = np.linalg.inv(K).dot(v5) / np.linalg.norm(np.linalg.inv(K).dot(v5))
    dPrime3 = np.linalg.inv(K).dot(v6) / np.linalg.norm(np.linalg.inv(K).dot(v6))

    di = np.zeros((3, 3))
    di[:, 0] = d1.T
    di[:, 1] = d2.T
    di[:, 2] = d3.T

    print(di)

    diPrime = np.zeros((3, 3))
    diPrime[:, 0] = dPrime1.T
    diPrime[:, 1] = dPrime2.T
    diPrime[:, 2] = dPrime3.T

    # find rotation matrix
    R = diPrime.dot(np.linalg.inv(di))
    return R

if __name__ == '__main__':
    # Part A: Compute vanishing points.
    v1 = compute_vanishing_point(np.array(
            [[674, 1826], [2456, 1060], [1094, 1340], [1774, 1086]]))
    v2 = compute_vanishing_point(np.array(
            [[674, 1826], [126, 1056], [2456, 1060], [1940, 866]]))
    v3 = compute_vanishing_point(np.array(
            [[1094, 1340], [1080, 598], [1774, 1086], [1840, 478]]))

    v1b = compute_vanishing_point(np.array(
            [[314, 1912], [2060, 1040], [750, 1378], [1438, 1094]]))
    v2b = compute_vanishing_point(np.array(
            [[314, 1912], [36, 1578], [2060, 1040], [1598, 882]]))
    v3b = compute_vanishing_point(np.array(
            [[750, 1378], [714, 614], [1438, 1094], [1474, 494]]))

    # Part B: Compute the camera matrix.
    vanishing_points = [v1, v2, v3]
    K_ours = compute_K_from_vanishing_points(vanishing_points)

    K_actual = np.array([[2448.0, 0, 1253.0], [0, 2438.0, 986.0], [0, 0, 1.0]])


    # Part D: Estimate the angle between the box and floor.
    floor_vanishing1 = v1
    floor_vanishing2 = v2
    box_vanishing1 = v3
    box_vanishing2 = compute_vanishing_point(np.array(
            [[1094, 1340], [1774, 1086], [1080, 598], [1840, 478]]))
    angle = compute_angle_between_planes(
            [floor_vanishing1, floor_vanishing2],
            [box_vanishing1, box_vanishing2], K_actual)
 
    # Part E: Compute the rotation matrix between the two cameras.
    rotation_matrix = compute_rotation_matrix_between_cameras(np.array([v1, v2, v3]), np.array([v1b, v2b, v3b]), K_actual)
    #z, y, x = mat2euler(rotation_matrix)
    #x_angle = x * 180 / math.pi
    #y_angle = y * 180 / math.pi
    #z_angle = z * 180 / math.pi
 