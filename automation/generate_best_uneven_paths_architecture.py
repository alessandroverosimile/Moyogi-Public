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
import itertools

VIVADO_VERSION = 2023.1

def get_best_architecture(min_width, max_depth, n_trees, necessary_set_of_pes, max_LUTs, max_FFs, max_BRAMs, LUTs_tolerance = 1200, FFs_tolerance = 1200, BRAMs_tolerance = 16, wns_tolerance = -58):
    with open('resource_estimation_models/LUTs_model.pkl', 'rb') as f:
        LUTs_model = pickle.load(f)
    
    with open('resource_estimation_models/FFs_model.pkl', 'rb') as f:
        FFs_model = pickle.load(f)
    
    with open('resource_estimation_models/BRAMs_model.pkl', 'rb') as f:
        BRAMs_model = pickle.load(f)
    
    with open('resource_estimation_models/PS8LUTs_model.pkl', 'rb') as f:
        PS8LUTs_model = pickle.load(f)

    with open('resource_estimation_models/PS8FFs_model.pkl', 'rb') as f:
        PS8FFs_model = pickle.load(f)

    with open('resource_estimation_models/WNS_model.pkl', 'rb') as f:
        WNS_model = pickle.load(f)

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
            total_pes = int(pes_per_path*n_paths)

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

            sample_red_poly = np.delete(sample_poly,[4,7,8], axis=1)
            wns = WNS_model.predict(sample_red_poly)[0]
            
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
                best_wns = wns
        
        print(best_LUTs, best_FFs, best_BRAMs, best_wns)
        if best_LUTs < max_LUTs - LUTs_tolerance and best_FFs < max_FFs - FFs_tolerance and best_BRAMs < max_BRAMs - BRAMs_tolerance and best_wns >= wns_tolerance:
            found = True
            saved_width = best_w
            expected_consumption = (best_LUTs, best_FFs, best_BRAMs)
            expected_wns = best_wns
            print(f"Configuration with pes_per_path, n_paths, dim = {pes_per_path}, {n_paths}, {best_w} is synthesizable. Trying one more path...")
            best_n_paths = n_paths
            best_total_pes = total_pes
            n_paths += 1

        else:
            stop = True
            if found:
                if best_wns < 50:
                    print(f"Configuration with pes_per_path, n_paths, dim = {pes_per_path}, {n_paths}, {best_w} is not synthesizable due to predicted timing issues. Best configuration with n_paths = {best_n_paths}")
                else:
                    print(f"Configuration with pes_per_path, n_paths, dim = {pes_per_path}, {n_paths}, {best_w} is not synthesizable due to predicted resource overutilization. Best configuration with n_paths = {best_n_paths}")
            else:
                print(f"Configuration with pes_per_path, n_paths, dim = {pes_per_path}, {n_paths}, {best_w} is not synthesizable. No configuration found")
            
    if not found:
        sys.exit()

    #increase the number of paths by 1 keeping fixed the total number of PEs
    best_n_paths += 1
    stop = False
    removed = 0
    while(not stop):

        pes_per_path = best_total_pes/best_n_paths
        sample = [[pes_per_path,best_n_paths,saved_width]]

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

        sample_red_poly = np.delete(sample_poly,[4,7,8], axis=1)
        wns = WNS_model.predict(sample_red_poly)[0]
        
        DWC_LUTs = dwc_dict[saved_width][0]
        DWC_FFs = dwc_dict[saved_width][1]
        DWC_BRAMs = dwc_dict[saved_width][2]

        BControllers_LUTs = 235*(pes_per_path*best_n_paths)
        BControllers_FFs = 207*(pes_per_path*best_n_paths)
        
        LUTs = wrapper_LUTs + PS8_LUTs + DWC_LUTs + BControllers_LUTs
        FFs = wrapper_FFs + PS8_FFs + DWC_FFs + BControllers_FFs
        BRAMs = wrapper_BRAMs + DWC_BRAMs

        if LUTs < max_LUTs - LUTs_tolerance and FFs < max_FFs - FFs_tolerance and BRAMs < max_BRAMs - BRAMs_tolerance and wns >= wns_tolerance:
            print(f"In order to add a path, {removed} PEs have been removed")
            expected_consumption = (LUTs, FFs, BRAMs)
            expected_wns = wns
            stop = True
        else:
            removed += 1
            best_total_pes -= 1
    
    return best_n_paths, best_total_pes, saved_width, expected_consumption, expected_wns

def get_feasible_combinations(min_depth, max_depth, n_paths, total_pes):
    feasible_comb = []
    numbers = list(range(min_depth, max_depth+1))
    combinations = list(itertools.combinations_with_replacement(numbers, n_paths))
    for comb in combinations:
        if np.sum(comb) == total_pes and max_depth in comb:
            feasible_comb.append(comb)

    return feasible_comb

def get_depths_distribution(combination,n_trees,max_depth,min_depth,instruction_per_bram):
    trees_per_depth = math.ceil(n_trees/(max_depth-min_depth+1))
    removed = 0
    distribution = []
    for i in range(max_depth,min_depth-1,-1):
        trees_per_set = instruction_per_bram/(2**(i-1))
        count = 0
        for length in combination:
            if i == length:
                count += trees_per_set
            elif i < length:
                count += trees_per_set*(2**(length-i-1))
        
        if count > trees_per_depth+removed:
            distribution.append(trees_per_depth+removed)
            removed = 0
        else:
            distribution.append(count)
            removed = trees_per_depth + removed - count
        
    assert removed == 0

    return distribution

def estimate_latency(combination, distribution):
    supp_distr = distribution.copy() #just to avoid to lose the value in the original variable
    avg_trees_for_path = int(np.sum(distribution)/len(combination))
    combination = np.array(combination)
    max_depth = max(combination)
    latencies = np.zeros(len(combination))
    path_distribution = []
    overall_dist = np.zeros(len(combination))
    for i in range(len(distribution)):
        dist = np.zeros(len(combination))
        while(supp_distr[i] > 0):
            mask = combination<max_depth-i
            support_latencies = latencies.copy()
            mask_dt_threshold = overall_dist > (avg_trees_for_path*1.15) #threshold too spreaded distribution of DTs over the paths
            support_latencies[mask] = np.inf
            support_latencies[mask_dt_threshold] = np.inf
            index = np.argmin(support_latencies)
            latencies[index] += combination[index]*2+4
            supp_distr[i] -= 1
            overall_dist[index] += 1
            dist[index] += 1
        path_distribution.append(dist)
    
    return np.max(latencies), path_distribution

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
    
    n_paths, n_pes, best_width, expected_consumption, expected_wns = get_best_architecture(width, max_depth, n_trees, necessary_set_of_pes, max_LUTs, max_FFs, max_BRAMs)

    print("N paths", n_paths)
    print("N PEs", n_pes)
    print("Best width", best_width)
    print("Expected consumption", expected_consumption)
    print("Expected wns", expected_wns)

    #Generate all the possible combinations of configurations of n_pes in n_paths satisfying the constraints
    combinations = get_feasible_combinations(min_depth, max_depth, n_paths, n_pes)

    #Calculates the forced number of trees per depth caused by each possible combination
    distributions = []
    for comb in combinations:
        distributions.append(get_depths_distribution(comb,n_trees,max_depth,min_depth,instruction_per_bram))
    
    #for i in range(len(combinations)):
    #    print(combinations[i], distributions[i])
    
    # Comparing all the combinations in terms of expected latency, choosing the best one

    if len(combinations) == 0:
        print("Uneven configuration not available, for this RF switch to the even case")
        sys.exit()
    else:
        latencies = []
        path_distributions =[]
        for i in range(len(combinations)):
            latency, path_distribution = estimate_latency(combinations[i],distributions[i])
            latencies.append(latency)
            path_distributions.append(path_distribution)
            print(combinations[i],distributions[i],latency)
        
        choice = 0 # 0 if latency, 1 if accuracy
        best_in_acc = np.argmax(np.array(distributions)[:,0])
        best_in_lat = np.argmin(latencies)

        if choice:
            index = best_in_acc
        elif choice==0:
            index = best_in_lat
        else:
            index = 4

        best_combination = np.array(combinations[index])
        best_path_distribution = np.array(path_distributions[index],dtype='int')
        best_path_distribution = np.sum(best_path_distribution,axis=0)
    
    print(best_combination, best_path_distribution)

    os.chdir(f"{curdir}/../chisel_project")

    dma_bits = 2**int(np.log2(width))

    print("Execution with depth, n_trees, freq, n_paths, n_attr equals to ",  max_depth, n_trees, frq, n_paths, n_attr)
    
    #sys.exit() #activate to debug the resource estimation models
    
    cmd = f'sbt "runMain YoseUe_SATL.UnevenPathsVerilogGenerator {n_trees} {max_depth} {min_depth} {n_attr} {n_classes} {n_paths} {best_width}{best_combination}{best_path_distribution} "'
    cmd = cmd.replace('[',' ')
    cmd = cmd.replace(']',' ')
    print(cmd)
    success = os.system(cmd)

    if(success > 0):
        print("run failed")
        sys.exit(-10)

    os.chdir("../automation")

    print("Total PEs ", np.sum(best_combination))

    if VIVADO_VERSION==2021.2:
        ps_version = 3
    else:
        ps_version = 5

    template = env.get_template('vivadoScript.tcl.jinja')
    template.stream(n_pes=np.sum(best_combination), dma_bits=dma_bits, trgt_freq=frq, width=int(width/8), dma_bytes=int(dma_bits/8),ps_version=ps_version).dump('vivadoScript.tcl')

    cmd = f"source /xilinx/software/Vivado/{VIVADO_VERSION}/settings64.sh && vivado -nojournal -nolog -mode batch -source vivadoScript.tcl"
    success = os.system(cmd)

    if(success > 0):
        print("'project1' failed")
        sys.exit(-10)
    
    cmd = f"source /xilinx/software/Vivado/{VIVADO_VERSION}/settings64.sh && vivado -nojournal -nolog -mode batch -source synth_and_impl.tcl"
    success = os.system(cmd)

    if(success > 0):
        print("'project1' failed")

    cmd = 'mv ./block_diagram/block_diagram.gen/sources_1/bd/design_2/hw_handoff/design_2.hwh ./block_diagram/block_diagram.gen/sources_1/bd/design_2/hw_handoff/design_2_wrapper.hwh'
    success = os.system(cmd)
    if(success > 0):
        print("Rename of .hwh failed")

    cmd = 'mkdir ../Deploys/DeployParametricDepth' + str(max_depth) + 'Trees' + str(n_trees) + 'Frq' + str(frq) + 'Paths' + str(n_paths) + 'Attr' + str(n_attr) + '-' + str(choice) + '/'
    success = os.system(cmd)
    if(success > 0):
        print("Directory not created")
        
    cmd = 'cp ./block_diagram/block_diagram.gen/sources_1/bd/design_2/hw_handoff/design_2_wrapper.hwh ../Deploys/DeployParametricDepth' + str(max_depth) + 'Trees' + str(n_trees) + 'Frq' + str(frq) + 'Paths' + str(n_paths) + 'Attr' + str(n_attr) + '-' + str(choice) + '/'
    success = os.system(cmd)

    cmd = 'cp ./block_diagram/block_diagram.runs/impl_1/design_2_wrapper.bit ../Deploys/DeployParametricDepth' + str(max_depth) + 'Trees' + str(n_trees) + 'Frq' + str(frq) + 'Paths' + str(n_paths) + 'Attr' + str(n_attr) + '-' + str(choice) + '/'
    success = os.system(cmd)
    if(success > 0):
        print("'build' failed")
        cmd = '>> ../Deploys/DeployParametricDepth' + str(max_depth) + 'Trees' + str(n_trees) + 'Frq' + str(frq) + 'Paths' + str(n_paths) + 'Attr' + str(n_attr) + 'Attr' + str(n_attr) + '-' + str(choice) + '/FAIL.txt'
        os.system(cmd)

    cmd = 'cp ./block_diagram/block_diagram.gen/utilization_report.txt ../Deploys/DeployParametricDepth' + str(max_depth) + 'Trees' + str(n_trees) + 'Frq' + str(frq) + 'Paths' + str(n_paths) + 'Attr' + str(n_attr) + '-' + str(choice) + '/'
    os.system(cmd)
    if(success > 0):
        print("Utilization report not created")

    cmd = 'cp ./block_diagram/block_diagram.gen/timing_report.txt ../Deploys/DeployParametricDepth' + str(max_depth) + 'Trees' + str(n_trees) + 'Frq' + str(frq) + 'Paths' + str(n_paths) + 'Attr' + str(n_attr) + '-' + str(choice) + '/'
    os.system(cmd)
    if(success > 0):
        print("Timing report not created")

    cmd = "echo " + str(best_combination) + str(best_path_distribution) + " > ../Deploys/DeployParametricDepth" + str(max_depth) + 'Trees' + str(n_trees) + 'Frq' + str(frq) + 'Paths' + str(n_paths) + 'Attr' + str(n_attr) + '-' + str(choice) + '/description.txt' 
    os.system(cmd)
    
    print("Synthesis with " + str(max_depth) + " depth, " + str(n_trees) + " estimators in " + str(n_paths) + " paths with " + str(n_attr) + " attributes completed")
    
    
if __name__ == "__main__":
    main()