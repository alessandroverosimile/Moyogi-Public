#import codecs
import jinja2
import os
import numpy as np
import sys
import shutil
import math

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

couples = [(1080,5),(180,7),(90,3)]
early_termination = 1 #set to 0 to skip early termination, 1 to enable it

for i,couple in enumerate(couples):
    
    n_trees = couple[0]
    max_depth = couple[1]
    min_depth = max_depth - 4
    n_depths = max_depth - min_depth + 1

    while(n_trees%n_depths != 0):
        n_trees += 1

    if((max_depth > 5 and n_trees > 225) or (max_depth > 7 and n_trees > 30)):
    	print("Exection with " + str(max_depth) + " depth, " + str(n_trees) + " estimators and " + str(frqs[0]) + "MHz frequency skipped")
    	continue
    
    n_classes = 6
    n_attr = 5

    cmd = f"python3 generate_best_equal_paths_architecture.py {n_trees} {max_depth} {min_depth} {n_attr} {n_classes} {early_termination} {frqs[0]} "
    success = os.system(cmd)

    if(success > 0):
        print("run failed")
        sys.exit(-10)

    print(f"Execution of synthesis {i} completed")