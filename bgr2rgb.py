import cv2
import numpy as np
from PIL import Image
import os
# im_cv=cv2.imread('C:\\Users\\64180\\Desktop\\ail_vis\\vis\\voc_qua_sp\\0024rgb.jpg')
# # im_rgb = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
# Image.fromarray(im_cv).save('C:\\Users\\64180\\Desktop\\ail_vis\\vis\\voc_qua_sp\\0024rgb.jpg')

# im_cv=cv2.imread('C:\\Users\\64180\\Downloads\\Cross-Modal Attention for Incremental Learning in Semantic Segmentation\\latex\image\\vis\\voc_qua_sp\\0009rgb.jpg')
# # im_rgb = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
# Image.fromarray(im_cv).save('C:\\Users\\64180\\Downloads\\Cross-Modal Attention for Incremental Learning in Semantic Segmentation\\latex\\image\\vis\\voc_qua_sp\\0009rgb.jpg')
im_cv=cv2.imread('C:\\Users\\64180\\Downloads\\img_5.png')
# im_rgb = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
Image.fromarray(im_cv).save('C:\\Users\\64180\\Downloads\\img_5_new.png')
# im_cv=cv2.imread('C:\\Users\\64180\\Downloads\\Cross-Modal Attention for Incremental Learning in Semantic Segmentation\\latex\image\\vis\\voc_qua\\ours\\1250rgb.jpg')
# # im_rgb = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
# Image.fromarray(im_cv).save('C:\\Users\\64180\\Downloads\\Cross-Modal Attention for Incremental Learning in Semantic Segmentation\\latex\\image\\vis\\voc_qua\\ours\\1250rgb.jpg')



### cluster

# root='./voc-741'
# for i in range(1449):
#
#     name=str(i).zfill(4)+'rgb.jpg'
#     path=os.path.join(root,name)
#     im_cv=cv2.imread(path)
#     # im_rgb = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
#     Image.fromarray(im_cv).save(path)