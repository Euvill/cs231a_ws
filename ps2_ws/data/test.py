import numpy as np

real_XY = np.load('/home/euvill/Desktop/ps1_ws/data/real_XY.npy')  # .npy文件
np.savetxt('real_XY.txt', real_XY, fmt='%0.18f')
print(real_XY)

front_image = np.load('/home/euvill/Desktop/ps1_ws/data/front_image.npy')  # .npy文件
np.savetxt('front_image.txt', front_image, fmt='%0.18f')
print(front_image)

back_image = np.load('/home/euvill/Desktop/ps1_ws/data/back_image.npy')  # .npy文件
np.savetxt('back_image.txt', back_image, fmt='%0.18f')
print(back_image)