#import codecs
import jinja2
import os
import numpy as np
import pandas as pd
import sys
import shutil
import math
import pickle
from sklearn.preprocessing import PolynomialFeatures

VIVADO_VERSION = 2023.1

def main():
    
    total, used, free = shutil.disk_usage("/")
    print("Total, used, free = ", total/10**9, used/10**9, free/10**9)
    if free // (2**30) < 4:
        print("NO SPACE")
        sys.exit(-10)

    #Random Forests parameters

    n_trees = int(sys.argv[1])
    max_depth = int(sys.argv[2])
    min_depth = int(sys.argv[3])
    n_attr = int(sys.argv[4])
    n_classes = int(sys.argv[5])
    n_depths = max_depth - min_depth + 1

    #architecture parameters
    max_LUTs = 70560
    max_FFs = 141160
    max_BRAMs = 216

    frq = sys.argv[6]

    fsloader = jinja2.FileSystemLoader(r'./')
    env = jinja2.Environment(loader=fsloader)

    curdir = os.curdir

    width = 32*n_attr+(n_depths+n_classes)*16+48
    while width%32 != 0:
        width += 8

    bram_size = 36*1024
    instruction_per_bram = int(bram_size/64)
    max_trees_per_set = int(n_depths*(instruction_per_bram/(2**(max_depth-1))))
    necessary_set_of_pes = int(math.ceil(n_trees/max_trees_per_set))
    
    n_paths = 1 #necessary_set_of_pes

    print("N paths", n_paths)
    
    os.chdir(f"{curdir}/../chisel_project")

    set_of_pes = max(necessary_set_of_pes,n_paths)

    dma_bits = 2**int(np.log2(width))

    print("Execution with depth, n_trees, freq, n_paths, n_attr equals to ",  max_depth, n_trees, frq, n_paths, n_attr)
    
    #sys.exit() #activate to debug the resource estimation models
    
    cmd = f'sbt "runMain YoseUe_SATL.VerilogGenerator {n_trees} {max_depth} {min_depth} {n_attr} {n_classes} {n_paths} {width} {necessary_set_of_pes}"'
    success = os.system(cmd)

    if(success > 0):
        print("run failed")
        sys.exit(-10)

    os.chdir("../automation")

    print("Total PEs ", set_of_pes*max_depth)

    if VIVADO_VERSION==2021.2:
        ps_version = 3
    else:
        ps_version = 5

    template = env.get_template('vivadoScript.tcl.jinja')
    template.stream(n_pes=np.sum(best_combination), dma_bits=dma_bits, trgt_freq=frq, width=int(width/8), dma_bytes=int(dma_bits/8),ps_version=ps_version).dump('vivadoScript.tcl')
    
    cmd = f"source /xilinx/software/Vivado/2021.2/settings64.sh && vivado -nojournal -nolog -mode batch -source vivadoScript.tcl"
    success = os.system(cmd)

    if(success > 0):
        print("'project1' failed")
        sys.exit(-10)
    
    cmd = "source /xilinx/software/Vivado/2021.2/settings64.sh && vivado -nojournal -nolog -mode batch -source synth_and_impl.tcl"
    success = os.system(cmd)

    if(success > 0):
        print("'project1' failed")

    cmd = 'mv ./block_diagram/block_diagram.gen/sources_1/bd/design_2/hw_handoff/design_2.hwh ./block_diagram/block_diagram.gen/sources_1/bd/design_2/hw_handoff/design_2_wrapper.hwh'
    success = os.system(cmd)
    if(success > 0):
        print("Rename of .hwh failed")

    cmd = 'mkdir ../Deploys/DeployParametricDepth' + str(max_depth) + 'Trees' + str(n_trees) + 'Frq' + str(frq) + 'Paths' + str(n_paths) + 'Attr' + str(n_attr) + '/'
    success = os.system(cmd)
    if(success > 0):
        print("Directory not created")
        
    cmd = 'cp ./block_diagram/block_diagram.gen/sources_1/bd/design_2/hw_handoff/design_2_wrapper.hwh ../Deploys/DeployParametricDepth' + str(max_depth) + 'Trees' + str(n_trees) + 'Frq' + str(frq) + 'Paths' + str(n_paths) + 'Attr' + str(n_attr) + '/'
    success = os.system(cmd)

    cmd = 'cp ./block_diagram/block_diagram.runs/impl_1/design_2_wrapper.bit ../Deploys/DeployParametricDepth' + str(max_depth) + 'Trees' + str(n_trees) + 'Frq' + str(frq) + 'Paths' + str(n_paths) + 'Attr' + str(n_attr) + '/'
    success = os.system(cmd)
    if(success > 0):
        print("'build' failed")
        cmd = '>> ../Deploys/DeployParametricDepth' + str(max_depth) + 'Trees' + str(n_trees) + 'Frq' + str(frq) + 'Paths' + str(n_paths) + 'Attr' + str(n_attr) + 'Attr' + str(n_attr) + '/FAIL.txt'
        os.system(cmd)

    cmd = 'cp ./block_diagram/block_diagram.gen/utilization_report.txt ../Deploys/DeployParametricDepth' + str(max_depth) + 'Trees' + str(n_trees) + 'Frq' + str(frq) + 'Paths' + str(n_paths) + 'Attr' + str(n_attr) + '/'
    os.system(cmd)
    if(success > 0):
        print("Utilization report not created")

    cmd = 'cp ./block_diagram/block_diagram.gen/timing_report.txt ../Deploys/DeployParametricDepth' + str(max_depth) + 'Trees' + str(n_trees) + 'Frq' + str(frq) + 'Paths' + str(n_paths) + 'Attr' + str(n_attr) + '/'
    os.system(cmd)
    if(success > 0):
        print("Timing report not created")

    print("Synthesis with " + str(max_depth) + " depth, " + str(n_trees) + " estimators in " + str(n_paths) + " paths with " + str(n_attr) + " attributes completed")

if __name__ == "__main__":
    main()
