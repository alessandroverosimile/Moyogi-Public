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

VIVADO_VERSION =  2024.2

#Optimal number of paths estimated by the resource estimation model

def get_best_architecture(min_width, max_depth, n_trees, necessary_set_of_pes, max_LUTs, max_FFs, max_BRAMs, early_termination, LUTs_tolerance = 700, FFs_tolerance = 4000, BRAMs_tolerance = 16, wns_tolerance = 50):
    if early_termination:
        models_dir = "resource_estimation_models_et"
    else:
        models_dir = "resource_estimation_models"
    
    with open(f'{models_dir}/LUTs_model.pkl', 'rb') as f:
        LUTs_model = pickle.load(f)
    
    with open(f'{models_dir}/FFs_model.pkl', 'rb') as f:
        FFs_model = pickle.load(f)
    
    with open(f'{models_dir}/BRAMs_model.pkl', 'rb') as f:
        BRAMs_model = pickle.load(f)
    
    with open(f'{models_dir}/PS8LUTs_model.pkl', 'rb') as f:
        PS8LUTs_model = pickle.load(f)

    with open(f'{models_dir}/PS8FFs_model.pkl', 'rb') as f:
        PS8FFs_model = pickle.load(f)

    #with open('resource_estimation_models/WNS_model.pkl', 'rb') as f:
    #    WNS_model = pickle.load(f)

    df = pd.read_csv('resource_estimation_models/resource_consumption_dwc.csv', delimiter=';')
    dwc_dict = {}
    for row in df.iterrows():
        dwc_dict[row[1]['width']] = (row[1]['LUTs'],row[1]['FFs'],row[1]['BRAMs'])

    stop = False
    found = False
    n_paths = 1
    best_n_paths = 1
    while not stop:
        if necessary_set_of_pes >= n_paths:
            if necessary_set_of_pes % n_paths == 0:
                total_pes = max_depth*necessary_set_of_pes
                pes_per_path = total_pes/n_paths
            else:
                n_paths += 1
                continue
        else:
            pes_per_path = max_depth

        maximum_width = 2**int(np.log2(min_width)+1) + 32
        best_LUTs = np.inf
        best_FFs = np.inf
        best_BRAMs = np.inf
        best_w = min_width
        for w in range(min_width,maximum_width,32):

            sample = [[pes_per_path,n_paths,w]]

            poly_features = PolynomialFeatures(degree=3)
            sample_poly = poly_features.fit_transform(sample)

            sample_red_poly = np.delete(sample_poly,[0,4,10,11,12,15,16,17,18], axis=1)
            wrapper_LUTs = LUTs_model.predict(sample_red_poly)[0]

            sample_red_poly = np.delete(sample_poly,[0,4,7,9,10,11,12,13,15,16,17,18], axis=1)
            wrapper_FFs = FFs_model.predict(sample_red_poly)[0]

            sample_red_poly = np.delete(sample_poly,[3,4,6,8,9,12,13,14,15,17,18,19], axis=1)
            PS8_LUTs = PS8LUTs_model.predict(sample_red_poly)[0]

            sample_red_poly = np.delete(sample_poly,[0,3,6,7,8,9,12,14,17,18], axis=1)
            PS8_FFs = PS8FFs_model.predict(sample_red_poly)[0]

            poly_features = PolynomialFeatures(degree=2)
            sample_poly = poly_features.fit_transform(sample)
            sample_red_poly = np.delete(sample_poly,[0,3,5], axis=1)
            wrapper_BRAMs = BRAMs_model.predict(sample_red_poly)[0]

            #sample_red_poly = np.delete(sample_poly,[4,7,8], axis=1)
            #wns = WNS_model.predict(sample_red_poly)[0]
            
            DWC_LUTs = dwc_dict[w][0]
            DWC_FFs = dwc_dict[w][1]
            DWC_BRAMs = dwc_dict[w][2]

            BControllers_LUTs = 235*(pes_per_path*n_paths)
            BControllers_FFs = 207*(pes_per_path*n_paths)
            
            LUTs = wrapper_LUTs + PS8_LUTs + DWC_LUTs + BControllers_LUTs
            FFs = wrapper_FFs + PS8_FFs + DWC_FFs + BControllers_FFs
            BRAMs = wrapper_BRAMs + DWC_BRAMs

            if LUTs < best_LUTs:
                best_LUTs = LUTs
                best_FFs = FFs
                best_BRAMs = BRAMs
                best_w = w
                #best_wns = wns
        
        print(f"Trial with N PATHS = {n_paths}")
        print("LUTs, FFs, BRAMs")
        print(best_LUTs, best_FFs, best_BRAMs)
        if best_LUTs < max_LUTs - LUTs_tolerance and best_FFs < max_FFs - FFs_tolerance and best_BRAMs < max_BRAMs - BRAMs_tolerance:# and best_wns >= wns_tolerance:
            found = True
            saved_width = best_w
            expected_consumption = (best_LUTs, best_FFs, best_BRAMs)
            #expected_wns = best_wns
            print(f"Configuration with pes_per_path, n_paths, dim = {pes_per_path}, {n_paths}, {best_w} is synthesizable. Trying one more path...")
            best_n_paths = n_paths
            n_paths += 1

        else:
            stop = True
            if found:
                print(f"Configuration with pes_per_path, n_paths, dim = {pes_per_path}, {n_paths}, {best_w} is not synthesizable due to predicted resource overutilization. Best configuration with n_paths = {best_n_paths}")
            else:
                print(f"Configuration with pes_per_path, n_paths, dim = {pes_per_path}, {n_paths}, {best_w} is not synthesizable. No configuration found")
            
    if not found:
        sys.exit()
    
    return best_n_paths, saved_width, expected_consumption, expected_wns

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
    early_termination = int(sys.argv[6])
    n_depths = max_depth - min_depth + 1

    #architecture parameters
    max_LUTs = 70560
    max_FFs = 141160
    max_BRAMs = 216

    frq = sys.argv[7]

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
    
    n_paths, best_width, expected_consumption, expected_wns = get_best_architecture(width, max_depth, n_trees, necessary_set_of_pes, max_LUTs, max_FFs, max_BRAMs, early_termination)
    
    print("N paths", n_paths)
    print("Best width", best_width)
    print("Expected consumption", expected_consumption)
    #print("Expected wns", expected_wns)
    
    os.chdir(f"{curdir}/../chisel_project")

    set_of_pes = max(necessary_set_of_pes,n_paths)

    dma_bits = 2**int(np.log2(width))

    # WEIGHTED CASE, UNCOMMENT THIS LINES IF YOU WANT TO CONSIDER THE WEIGHTED CASE 
    '''
    print("Model training...")

    X_sets, Y_sets = import_accelerometer([0.6,0.2,0.2])
    X_train = X_sets[0]
    X_val = X_sets[1]
    X_test = X_sets[2]
    Y_train = Y_sets[0]
    Y_val = Y_sets[1]
    Y_test = Y_sets[2]
    mdcls = MultiDepthNeuralRandomForestClassifier(total_estimators=n_trees,min_depth=min_depth,max_depth=max_depth, random_state=3)
    mdcls.fit(X_train, Y_train, X_val, Y_val)
    print("model accuracy: ", mdcls.score(X_test,Y_test))

    print("N DTs: ", len(mdcls.trees_and_weights))
    
    max_votation = 0
    for i in range(len(mdcls.trees_and_weights)):
        max_votation += mdcls.trees_and_weights[i][1]

    '''

    max_votation = n_trees #NON-WEIGHTED CASE, COMMENT THIS LINE AND UNCOMMENT MODEL TRAINING IF YOU WANT TO CONSIDER THE WEIGHTED CASE 

    print("Execution with depth, n_trees, freq, n_paths, n_attr equals to ",  max_depth, n_trees, frq, n_paths, n_attr)
    
    #sys.exit() #activate to debug the resource estimation models
    string = f"N paths: {n_paths}, Overall sets of PEs: {set_of_pes}, Width: {best_width}"

    cmd = f'sbt "runMain YoseUe_SATL.VerilogGenerator {n_trees} {max_depth} {min_depth} {n_attr} {n_classes} {n_paths} {best_width} {necessary_set_of_pes} {early_termination} {max_votation}"'
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
