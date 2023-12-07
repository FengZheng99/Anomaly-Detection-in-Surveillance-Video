import numpy as np
import os
import glob


'''
FOR UCF CRIME
'''
root_path = 'D:/UMKC/Fall 2023/CS5588/UCF-Crime/Test'
dirs = os.listdir(root_path)
with open('ucf-c3d-test.list', 'w+') as f:
    normal = []
    for dir in dirs:
        files = sorted(glob.glob(os.path.join(root_path, dir, "*.npy")))
        for file in files:
            if '__' not in file:
                if 'Normal_' in file:
                    normal.append(file)
                else:
                    newline = file+'\n'
                    f.write(newline)
    for file in normal:
        newline = file+'\n'
        f.write(newline)
