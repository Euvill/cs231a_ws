import numpy as np

np.set_printoptions(threshold=np.inf)

unit_test_camera_matrix = np.load('/home/euvill/Desktop/cs231a_ws/ps2_ws/data/unit_test_camera_matrix.npy')  
np.savetxt('unit_test_camera_matrix.txt', unit_test_camera_matrix.reshape((3,-1)), fmt='%0.18f')
#print(unit_test_camera_matrix)

unit_test_image_matches = np.load('/home/euvill/Desktop/cs231a_ws/ps2_ws/data/unit_test_image_matches.npy')  
np.savetxt('unit_test_image_matches.txt', unit_test_image_matches, fmt='%0.18f')
#print(unit_test_image_matches)

fundamental_matrices = np.load('/home/euvill/Desktop/cs231a_ws/ps2_ws/data/statue/fundamental_matrices.npy')  
np.savetxt('fundamental_matrices.txt', fundamental_matrices, fmt='%s')
#print(fundamental_matrices)

matches_subset = np.load('/home/euvill/Desktop/cs231a_ws/ps2_ws/data/statue/matches_subset.npy')[0, :]  
np.savetxt('matches_subset.txt', matches_subset, fmt='%s', newline='\n')


#dense_matches = np.load('/home/euvill/Desktop/cs231a_ws/ps2_ws/data/statue/dense_matches.npy') 
#np.savetxt('dense_matches.txt', dense_matches, fmt='%s')
#print(dense_matches)

#copy_dense_matches = np.load('/home/euvill/Desktop/cs231a_ws/ps2_ws/data/statue/copy_dense_matches.npy')  
#np.savetxt('copy_dense_matches.txt', copy_dense_matches, fmt='%0.18f')
#print(copy_dense_matches)