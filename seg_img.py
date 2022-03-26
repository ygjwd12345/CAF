from skimage import io
from skimage import color
from skimage import segmentation
import matplotlib.pyplot as plt
import numpy as np
import os
import utils

path_gt='./voc-gt'
path_img='./voc_img'
save_path='./seg-img-new'
if not os.path.exists(save_path):
    os.makedirs(save_path)
name=os.listdir(path_gt)
for n in name:
    seg= plt.imread(os.path.join(path_gt,n)).copy()
    img=plt.imread(os.path.join(path_img,n.replace('gt','rgb')))
    ### remove invalid label
    # seg[seg>21]=0
    ### for 15-5 new
    seg[seg < 16] = 0
    seg=seg+1
    if 17 in np.unique(seg) or 18 in np.unique(seg) or 19 in np.unique(seg) or 20 in np.unique(seg) or 21 in np.unique(seg):
        print(np.unique(seg))

        p_s=os.path.join(save_path,n.replace('gt','old'))
        color_map=utils.color_map('voc')/256
        img_seg=color.label2rgb(seg, img,colors=color_map,bg_label=0,alpha=0.8)
        # img_seg=color.label2rgb(seg, img,bg_label=0,alpha=0.3, kind='avg')

        # io.imshow()
        # plt.show()

        plt.imsave(p_s, img_seg)
        # break


# Generate automatic colouring from classification labels
