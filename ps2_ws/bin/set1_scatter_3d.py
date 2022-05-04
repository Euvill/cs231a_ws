import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread
from mpl_toolkits.mplot3d import Axes3D

def get_data_from_txt_file(filename, use_subset = False):
    with open(filename) as f:
            lines = f.read().splitlines()
    
    number_pts = int(lines[0])

    points = np.ones((number_pts, 3))
    
    for i in xrange(number_pts):
        split_arr = lines[i+1].split()
        x, y, z = split_arr
        points[i,2] = z
        points[i,0] = x 
        points[i,1] = y
    return points

def scatter_3D_axis_equal(X, Y, Z, ax):
    ax.scatter(X, Y, Z)
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0

    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

if __name__ == '__main__':

    structure = get_data_from_txt_file('../bin/set1_structure_esti.txt')
    points_3d = get_data_from_txt_file('../bin/set1_ground_truth.txt')

    # Plot the structure
    fig = plt.figure()
    ax = fig.add_subplot(121, projection = '3d')
    scatter_3D_axis_equal(structure[:,0], structure[:,1], structure[:,2], ax)
    ax.set_title('Factorization Method')
    ax = fig.add_subplot(122, projection = '3d')
    scatter_3D_axis_equal(points_3d[:,0], points_3d[:,1], points_3d[:,2], ax)
    ax.set_title('Ground Truth')

    plt.show()
