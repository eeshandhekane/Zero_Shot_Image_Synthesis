# Dependencies
import cv2
import numpy as np
import argparse
import os, sys, re
import time


# Parse the npy name
parser = argparse.ArgumentParser()
parser.add_argument('--npy_name', type = str, required = True)
parser.add_argument('--out_name', type = str, required = True)
args = parser.parse_args()
npy_name = str(args.npy_name)
out_name = str(args.out_name)


# Load the npy
array = np.load(npy_name)
print('[DEBUG] Loaded array shape : ' + str(array.shape))
orig = array[0]
recn = array[1]
print('[DEBUG] Original array shape : ' + str(orig.shape))
print('[DEBUG] Reconstructed array shape : ' + str(recn.shape))
orig_1 = np.reshape(orig[0], [28, 28, 1])
orig_2 = np.reshape(orig[1], [28, 28, 1])
orig_3 = np.reshape(orig[2], [28, 28, 1])
orig_4 = np.reshape(orig[3], [28, 28, 1])
orig_5 = np.reshape(orig[4], [28, 28, 1])
recn_1 = np.reshape(recn[0], [28, 28, 1])
recn_2 = np.reshape(recn[1], [28, 28, 1])
recn_3 = np.reshape(recn[2], [28, 28, 1])
recn_4 = np.reshape(recn[3], [28, 28, 1])
recn_5 = np.reshape(recn[4], [28, 28, 1])
cv2.imwrite('temp00001.jpg', orig_1*255)
cv2.imwrite('temp00002.jpg', recn_1*255)
cv2.imwrite('temp00003.jpg', orig_2*255)
cv2.imwrite('temp00004.jpg', recn_2*255)
cv2.imwrite('temp00005.jpg', orig_3*255)
cv2.imwrite('temp00006.jpg', recn_3*255)
cv2.imwrite('temp00007.jpg', orig_4*255)
cv2.imwrite('temp00008.jpg', recn_4*255)
cv2.imwrite('temp00009.jpg', orig_5*255)
cv2.imwrite('temp00010.jpg', recn_5*255)
os.system('ffmpeg -f image2 -r 1/2 -i temp%05d.jpg -vcodec mpeg4 -y ' + 'Reconstructed_Dataset/Reconstruction_vid_' + str(out_name) + '.mp4')
time.sleep(0.5)
os.system('rm -f temp00001.jpg temp00002.jpg temp00003.jpg temp00004.jpg temp00005.jpg temp00006.jpg temp00007.jpg temp00008.jpg temp00009.jpg temp00010.jpg')
time.sleep(0.5)
