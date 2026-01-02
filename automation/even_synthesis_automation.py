#import codecs
import jinja2
import os
import numpy as np
import sys
import shutil
import math
import pandas as pd

fsloader = jinja2.FileSystemLoader(r'./')
env = jinja2.Environment(loader=fsloader)

curdir = os.curdir

frqs = [166] #[166, 187, 214]
#n_trees_values = [10, 30, 50, 75, 100, 180, 200, 300, 700, 1080]
#max_depth_values = [5,7,9]

#couples = []
#for i in n_trees_values:
#    for j in max_depth_values:
#        couples.append((i,j))


df = pd.read_csv("../resource_estimation_model/data/resource_consumption_dataset.csv",sep=';')
df = df[["pes_per_path","n_paths","sample_dim"]]

couples = []
for idx, row in df.iterrows():
    if abs(row['pes_per_path'] - round(row["pes_per_path"])) > 0:
        continue
    else:
        if row['pes_per_path']>=10:
            if row['pes_per_path']%5==0:
                max_depth = 5
            else:
                max_depth=4
        else:
            max_depth = int(row['pes_per_path'])

        sets_per_path = row['pes_per_path']/max_depth
        n_sets = int(row['n_paths']*sets_per_path)
    n_attr = int((row['sample_dim'] - 224)/32)
    couples.append((8, max_depth, n_attr, 6, int(row['n_paths']), int(row['sample_dim']), n_sets))

#couples = [(1080,5,4,7,2,384),(180,7,4,7,2,384),(30,9,4,7,3,384),(900,5,6,6,1,448),(180,7,6,6,2,448),(30,9,6,6,3,448),(720,5,8,11,2,640),(135,7,8,11,3,640),(20,9,8,11,2,640)]

early_termination = 1 #set to 0 to skip early termination, 1 to enable it

for i,couple in enumerate(couples):

    n_trees = couple[0]
    max_depth = couple[1]
    min_depth = max_depth - 4
    n_depths = max_depth - min_depth + 1

    while(n_trees%n_depths != 0):
        n_trees += 1
    
    n_attr = couple[2]
    n_classes = couple[3]
    n_paths = couple[4]
    width = couple[5]
    n_sets = couple[6]

    if (n_sets <= 4 or max_depth!=5) and (n_sets==n_paths or max_depth!=4) :
        continue
    
    cmd = f"python3 generate_best_equal_apriori_paths_architecture.py {n_trees} {max_depth} {min_depth} {n_attr} {n_classes} {n_paths} {width} {early_termination} {frqs[0]} {n_sets}"
    success = os.system(cmd)

    if(success > 0):
        print("run failed")
        sys.exit(-10)

    print(f"Execution of synthesis {i} completed")
