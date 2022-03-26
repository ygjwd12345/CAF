import sys
import random

import dataset.cityscape as city
from dataset.cityscape import CitySegmentationIncremental,CitySegmentation
import dataset.transform as transforms
import torchvision as tv
import numpy as np
import os

data = CitySegmentation('./dataset', train=True)
def group_images(dataset, labels):
    # Group images based on the label in LABELS
    idxs = [[] for lab in labels]

    labels_cum = labels + [0, 255]
    for i in range(len(dataset)):
        cls = np.unique(np.array(dataset[i][1]))
        print(cls)
        if all(x in labels_cum for x in cls):
            for x in cls:
                if x in labels:
                    idxs[x].append(i)
        else:
            print(f"Not all in labels_cum: {cls}")
        if i % 1000 == 0:
            print(f"Done {i} / {len(dataset)}")
    return idxs

idxs = group_images(data, list(range(20)))
print(idxs)
class_members = {}
for i,l in enumerate(idxs):
    if i==0:
        continue
    class_members[i]=len(l)

#random.seed(42)
a = np.arange(20)
a=a[1:]
a=np.random.permutation(a)
ordered_classes = class_members.items()

# map the image to the class contained, the inverse of idxs.
imgs_to_cls = {}
for j,v in ordered_classes:
        for i in list(idxs[j]):
            if not i in imgs_to_cls.keys():
                imgs_to_cls[i]=[]
            imgs_to_cls[i].append(j)
print(len(imgs_to_cls))
for i in imgs_to_cls.keys():
    if len(imgs_to_cls[i])<1:
        print(i)
# make the order for the dataset!
# map_class_to_set maps the class to the corresponding step

base_len = 15 # first training step: # of images
inc_len = 1  # following training step (assume same dimension)
sets = 4      # Num of incremental steps. E.g. the 50-50-50 has 2 inc. steps

order = np.arange(1,20)
print(order)
# np.random.shuffle(order) # make a random order

map_set_to_class = {}
map_class_to_set = {}

map_set_to_class[0] = order[0:base_len]
map_class_to_set.update({x:0 for x in map_set_to_class[0]})
offset = base_len
for i in range(1, sets+1):
    map_set_to_class[i] = order[offset : offset + inc_len]
    offset = offset + inc_len
    map_class_to_set.update({x:i for x in map_set_to_class[i]})

print({i:sorted(list(x)) for i,x in map_set_to_class.items()})
print({i:len(x) for i,x in map_set_to_class.items()})
print(map_class_to_set[1])
# This is the code who actually make the split.

all_added = set()

imgs = {}
others = {}
every = set()
added = [0 for i in range(20)]
for i in range(sets + 1):
    others[i] = set()
    imgs[i] = set()


# This is the most imporant function.
# It computes the score of the image to be assinged to a class
def score(ratio, imgs, expected, minval=False, w1=120, w2=1000, w3=1):
    return (1. - ratio) * w1 + w3 * minval


for i in imgs_to_cls.keys():  # loop for every image in the dataset
    # img_counts = num of images per class for the class in the actual image
    img_counts = [class_members[j] for j in imgs_to_cls[i]]
    # ratio = images assigned to the class / total number of images for the class
    ratios = [(added[j] + 0.0) / class_members[j] for j in imgs_to_cls[i]]
    added_counts = [added[j] for j in imgs_to_cls[i]]
    # assignments = set of each class in the image
    assignments = [map_class_to_set[j] for j in imgs_to_cls[i]]

    scores = [score(ratios[c], added_counts[c], class_members[j], 1. / (img_counts[c] / sum(img_counts))) for c, j in
              enumerate(imgs_to_cls[i])]

    # take the highest scorer class
    cl = scores.index(max(scores))
    # take the set of the higher scorer class
    a = assignments[cl]

    # add the image to the step images
    imgs[a].update([i])
    for j, ac in enumerate(assignments):
        if ac == a:
            # increment the number of images assigned to the classes in the current image contained in the same step.
            added[imgs_to_cls[i][j]] += 1

for i in range(1, len(added)):
    assignment = map_class_to_set[i]
    ratios = [len(set(idxs[i]).intersection(imgs[j])) / class_members[i] for j in range(0, sets + 1)]
    if ratios[assignment] < 0.5 or (ratios[assignment] < 1. and added[i] < 100):
        print(i, ratios[assignment], sum(ratios[1:]), class_members[i],
              len(set(idxs[i]).intersection(imgs[assignment])))
s = 0
for i in range(sets + 1):
    print(len(imgs[i]))
    s += len(imgs[i])

print(s)
map_set_to_class_id = {}
for i in range(sets+1):
    map_set_to_class_id[i] = np.array(list(range(20)))[map_set_to_class[i]]
print(map_set_to_class_id)
path = './split/'
os.makedirs(path, exist_ok=True)
with open(path + 'order.txt', 'w') as f:
    f.write(str({i:sorted(list(x)) for i,x in map_set_to_class_id.items()}))
for i in range(sets+1):
    np.save(path + 'train-'+str(i)+'.npy',  np.array(list(imgs[i])))
    print(len(imgs[i]))