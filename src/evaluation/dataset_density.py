import cv2
import sys

from IPython import embed
#
# map = np.zeros((192,256))
# p = "/home/dataset/DHF1K/"
# maps = ["frames", "depth", "flow"]
# images = 0
# assert len(sys.argv)==2, "Add [frames, depth, flow]"
# assert sys.argv[1] in maps, "Add [frames, depth, flow]"
# set = sys.argv[1]
# if set == maps[0]:
# 	path = os.path.join(p,"dhf1k_frames")
# elif set == maps[1]:
# 	path = os.path.join(p,"dhf1k_depth")
# elif set == maps[2]:
# 	path = os.path.join(p,"dhf1k_flow")
#
# for folder in range(601,701):
# 	for img in os.listdir(os.path.join(path,str(folder))):
# 		images += 1
# 		map += cv2.imread(os.path.join(path,str(folder),img), 0)
#
# map /= images
# cv2.imwrite("plots/{}.png".format(set), map)
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

path = "/home/dataset/DHF1K/dhf1k_gt"

df_map = pd.DataFrame(columns=['x_gt_pos','y_gt_pos'])

# for folder in range(601,701):
# for img in os.listdir(os.path.join(path, str(601), "fixation")):
# video 14
# take nonzero from every frame and select 10 random elements
# before taking the 10 random elements, convert to [(),(),(),...]
# then sum and represent


# for continuous map as depth or optical flow take the pixel values when
# the pixels around the evaluated pixel are lower than itself

image = cv2.imread(os.path.join(path, str(601), "fixation", "0001.png"), 0)
index = np.nonzero(image)
for x,y in zip(index[0], index[1]):
	df_map.loc[-1] = [x, y]
	df_map.index = df_map.index + 1

density_map = sns.jointplot(x=df_map.x_gt_pos, y=df_map.y_gt_pos, kind='kde', cmap="Blues",xlim=(100,200), ylim=(200,450))
density_map.fig.axes[0].invert_yaxis()

plt.savefig('plots/density_map.png')
