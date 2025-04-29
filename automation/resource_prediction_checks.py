import jinja2
import os
import numpy as np
import pandas as pd
import sys
import shutil
import math
import pickle
from sklearn.preprocessing import PolynomialFeatures

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

n_attr = 6
n_classes = 6
min_width = 32*n_attr+(5+n_classes)*16+48
while min_width%32 != 0:
    min_width += 8

pes_per_path = 27
n_paths = 1

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

    print("W: ", w)
    print("LUTs: ", LUTs)

    if LUTs < best_LUTs:
        best_LUTs = LUTs
        best_FFs = FFs
        best_BRAMs = BRAMs
        best_w = w
        best_wns = wns

print(f"Trial with best width = {best_w}")
print("LUTs, FFs, BRAMs, WNS")
print(best_LUTs, best_FFs, best_BRAMs, best_wns)