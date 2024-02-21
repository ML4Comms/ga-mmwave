import pandas as pd
import os
import pdb

current_directory = os.getcwd()

def load_datasets(mode):
    paths = ['AB', 'BA']
    datasets = {'AB':{},'BA':{}}
    for path in paths:
        if mode == 'small_scale_fading':
            if path == 'AB':
                SSF_mmdata_path = 'datasets/SSF_linear_mmdata_pathAB_2KHz'
                dir_path = os.path.join(current_directory, SSF_mmdata_path).replace("\\","/")
                datasets[path]['AP1'] = pd.read_csv(os.path.join(dir_path, 'pathAB_SSF_linear_AP1_downsampled2Khz_win100.txt'), sep=" ", header=None)
                datasets[path]['AP2'] = pd.read_csv(os.path.join(dir_path, 'pathAB_SSF_linear_AP2_downsampled2Khz_win100.txt'), sep=" ", header=None)
                datasets[path]['AP3'] = pd.read_csv(os.path.join(dir_path, 'pathAB_SSF_linear_AP3_downsampled2Khz_win100.txt'), sep=" ", header=None)
                datasets[path]['AP4'] = pd.read_csv(os.path.join(dir_path, 'pathAB_SSF_linear_AP4_downsampled2Khz_win100.txt'), sep=" ", header=None)
                datasets[path]['AP5'] = pd.read_csv(os.path.join(dir_path, 'pathAB_SSF_linear_AP5_downsampled2Khz_win100.txt'), sep=" ", header=None)
                datasets[path]['AP6'] = pd.read_csv(os.path.join(dir_path, 'pathAB_SSF_linear_AP6_downsampled2Khz_win100.txt'), sep=" ", header=None)
                datasets[path]['AP7'] = pd.read_csv(os.path.join(dir_path, 'pathAB_SSF_linear_AP7_downsampled2Khz_win100.txt'), sep=" ", header=None)
                datasets[path]['AP8'] = pd.read_csv(os.path.join(dir_path, 'pathAB_SSF_linear_AP8_downsampled2Khz_win100.txt'), sep=" ", header=None)
                datasets[path]['AP9'] = pd.read_csv(os.path.join(dir_path, 'pathAB_SSF_linear_AP9_downsampled2Khz_win100.txt'), sep=" ", header=None)
            elif path == 'BA':
                SSF_mmdata_path = 'datasets/SSF_linear_mmdata_pathBA_2KHz'
                dir_path = os.path.join(current_directory, SSF_mmdata_path).replace("\\","/")
                datasets[path]['AP1'] = pd.read_csv(os.path.join(dir_path, 'pathBA_SSF_linear_AP1_downsampled2Khz_win100.txt'), sep=" ", header=None)
                datasets[path]['AP2'] = pd.read_csv(os.path.join(dir_path, 'pathBA_SSF_linear_AP2_downsampled2Khz_win100.txt'), sep=" ", header=None)
                datasets[path]['AP3'] = pd.read_csv(os.path.join(dir_path, 'pathBA_SSF_linear_AP3_downsampled2Khz_win100.txt'), sep=" ", header=None)
                datasets[path]['AP4'] = pd.read_csv(os.path.join(dir_path, 'pathBA_SSF_linear_AP4_downsampled2Khz_win100.txt'), sep=" ", header=None)
                datasets[path]['AP5'] = pd.read_csv(os.path.join(dir_path, 'pathBA_SSF_linear_AP5_downsampled2Khz_win100.txt'), sep=" ", header=None)
                datasets[path]['AP6'] = pd.read_csv(os.path.join(dir_path, 'pathBA_SSF_linear_AP6_downsampled2Khz_win100.txt'), sep=" ", header=None)
                datasets[path]['AP7'] = pd.read_csv(os.path.join(dir_path, 'pathBA_SSF_linear_AP7_downsampled2Khz_win100.txt'), sep=" ", header=None)
                datasets[path]['AP8'] = pd.read_csv(os.path.join(dir_path, 'pathBA_SSF_linear_AP8_downsampled2Khz_win100.txt'), sep=" ", header=None)
                datasets[path]['AP9'] = pd.read_csv(os.path.join(dir_path, 'pathBA_SSF_linear_AP9_downsampled2Khz_win100.txt'), sep=" ", header=None)

        elif mode == 'composite_fading':
            if path == 'AB':
                composite_mmdata_path = 'datasets/Composite_linear_mmdata_pathAB_2KHz'
                dir_path = os.path.join(current_directory, composite_mmdata_path).replace("\\","/")
                datasets[path]['AP1'] = pd.read_csv(os.path.join(dir_path, 'pathAB_composite_linear_AP1_downsampled2Khz_win100.txt'), sep=" ", header=None)
                datasets[path]['AP2'] = pd.read_csv(os.path.join(dir_path, 'pathAB_composite_linear_AP2_downsampled2Khz_win100.txt'), sep=" ", header=None)
                datasets[path]['AP3'] = pd.read_csv(os.path.join(dir_path, 'pathAB_composite_linear_AP3_downsampled2Khz_win100.txt'), sep=" ", header=None)
                datasets[path]['AP4'] = pd.read_csv(os.path.join(dir_path, 'pathAB_composite_linear_AP4_downsampled2Khz_win100.txt'), sep=" ", header=None)
                datasets[path]['AP5'] = pd.read_csv(os.path.join(dir_path, 'pathAB_composite_linear_AP5_downsampled2Khz_win100.txt'), sep=" ", header=None)
                datasets[path]['AP6'] = pd.read_csv(os.path.join(dir_path, 'pathAB_composite_linear_AP6_downsampled2Khz_win100.txt'), sep=" ", header=None)
                datasets[path]['AP7'] = pd.read_csv(os.path.join(dir_path, 'pathAB_composite_linear_AP7_downsampled2Khz_win100.txt'), sep=" ", header=None)
                datasets[path]['AP8'] = pd.read_csv(os.path.join(dir_path, 'pathAB_composite_linear_AP8_downsampled2Khz_win100.txt'), sep=" ", header=None)
                datasets[path]['AP9'] = pd.read_csv(os.path.join(dir_path, 'pathAB_composite_linear_AP9_downsampled2Khz_win100.txt'), sep=" ", header=None)
            elif path == 'BA':
                composite_mmdata_path = 'datasets/Composite_linear_mmdata_pathBA_2KHz'
                dir_path = os.path.join(current_directory, composite_mmdata_path).replace("\\","/")
                datasets[path]['AP1'] = pd.read_csv(os.path.join(dir_path, 'pathBA_composite_linear_AP1_downsampled2Khz_win100.txt'), sep=" ", header=None)
                datasets[path]['AP2'] = pd.read_csv(os.path.join(dir_path, 'pathBA_composite_linear_AP2_downsampled2Khz_win100.txt'), sep=" ", header=None)
                datasets[path]['AP3'] = pd.read_csv(os.path.join(dir_path, 'pathBA_composite_linear_AP3_downsampled2Khz_win100.txt'), sep=" ", header=None)
                datasets[path]['AP4'] = pd.read_csv(os.path.join(dir_path, 'pathBA_composite_linear_AP4_downsampled2Khz_win100.txt'), sep=" ", header=None)
                datasets[path]['AP5'] = pd.read_csv(os.path.join(dir_path, 'pathBA_composite_linear_AP5_downsampled2Khz_win100.txt'), sep=" ", header=None)
                datasets[path]['AP6'] = pd.read_csv(os.path.join(dir_path, 'pathBA_composite_linear_AP6_downsampled2Khz_win100.txt'), sep=" ", header=None)
                datasets[path]['AP7'] = pd.read_csv(os.path.join(dir_path, 'pathBA_composite_linear_AP7_downsampled2Khz_win100.txt'), sep=" ", header=None)
                datasets[path]['AP8'] = pd.read_csv(os.path.join(dir_path, 'pathBA_composite_linear_AP8_downsampled2Khz_win100.txt'), sep=" ", header=None)
                datasets[path]['AP9'] = pd.read_csv(os.path.join(dir_path, 'pathBA_composite_linear_AP9_downsampled2Khz_win100.txt'), sep=" ", header=None)

    return datasets

