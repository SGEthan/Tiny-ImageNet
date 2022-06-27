# process validation dataset to match the torch util
import os
import shutil


# make new directory in validation folder
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        print(path)

val_dir = './tiny-imagenet/val'
with open('./tiny-imagenet/wnids.txt', 'r') as f:
    for line in f.readlines():
        line = line.strip('\n')
        folder = val_dir + '/' + line
        mkdir(folder)
        
val_image_dir = './tiny-imagenet/val/images'
with open('./tiny-imagenet/val/val_annotations.txt', 'r') as w:
    
    for line in w.readlines():
        line = line.strip('\n')
        dirlist = line.split()
        
        image_path = val_image_dir + '/' + dirlist[0]
        folder_path = val_dir + '/' + dirlist[1]
        
        shutil.copy(image_path, folder_path)