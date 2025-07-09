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
sys.path.append("../")
from software_host.MultiDepthRandomForest import MultiDepthNeuralRandomForestClassifier
from software_host.Dataset import import_accelerometer

VIVADO_VERSION = 2024.2

#Number of paths decided apriori to try by hands different architecture configurations

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
    n_paths = int(sys.argv[6])
    width = int(sys.argv[7])
    early_termination = int(sys.argv[8])
    n_depths = max_depth - min_depth + 1
    frq = sys.argv[9]

    fsloader = jinja2.FileSystemLoader(r'./')
    env = jinja2.Environment(loader=fsloader)

    curdir = os.curdir

    bram_size = 36*1024
    instruction_per_bram = int(bram_size/64)
    max_trees_per_set = int(n_depths*(instruction_per_bram/(2**(max_depth-1))))
    necessary_set_of_pes = int(math.ceil(n_trees/max_trees_per_set))
     
    print("N paths", n_paths)
    print("Width", width)
    
    os.chdir(f"{curdir}/../chisel_project")

    set_of_pes = max(necessary_set_of_pes,n_paths)

    dma_bits = 2**int(np.log2(width))

    max_votation = int(n_trees/4) #NON-WEIGHTED CASE, COMMENT THIS LINE AND UNCOMMENT MODEL TRAINING IF YOU WANT TO CONSIDER THE WEIGHTED CASE 

    print("Execution with depth, n_trees, freq, n_paths, n_attr equals to ",  max_depth, n_trees, frq, n_paths, n_attr)
    
    #sys.exit() #activate to debug the resource estimation models
    string = f"N paths: {n_paths}, Overall sets of PEs: {set_of_pes}, Width: {width}"

    cmd = f'sbt "runMain YoseUe_SATL.VerilogGenerator {n_trees} {max_depth} {min_depth} {n_attr} {n_classes} {n_paths} {width} {necessary_set_of_pes} {early_termination} {max_votation}"'
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
    width_bytes = int(width/8)
    dma_bytes = int(dma_bits/8)
    template.stream(n_pes=set_of_pes*max_depth, dma_bits=dma_bits, trgt_freq=frq, width=width_bytes, dma_bytes=dma_bytes,ps_version=ps_version).dump('vivadoScript.tcl')

    cmd = f"/bin/bash -c 'source /home/xilinx/Vivado/{VIVADO_VERSION}/settings64.sh && vivado -nojournal -nolog -mode batch -source vivadoScript.tcl'"
    success = os.system(cmd)

    if(success > 0):
        print("'project1' failed")
        sys.exit(-10)
        
    cmd = f"/bin/bash -c 'source /home/xilinx/Vivado/{VIVADO_VERSION}/settings64.sh && vivado -nojournal -nolog -mode batch -source synth_and_impl.tcl'"
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

    cmd = "echo " + string + " > ../Deploys/DeployParametricDepth" + str(max_depth) + 'Trees' + str(n_trees) + 'Frq' + str(frq) + 'Paths' + str(n_paths) + 'Attr' + str(n_attr) + '/description.txt' 
    os.system(cmd)

    print("Synthesis with " + str(max_depth) + " depth, " + str(n_trees) + " estimators in " + str(n_paths) + " paths with " + str(n_attr) + " attributes completed")

if __name__ == "__main__":
    main()
