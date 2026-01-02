import pandas as pd
import os
import re

def parse_vivado_utilization(file_path, target_module):
    """
    Parse Vivado utilization report and extract metrics for a specific module.
    
    Args:
        file_path: Path to the utilization report file
        target_module: Name of the module to search for
    
    Returns:
        Dictionary with Total LUTs, FFs, and RAMB36 values
    """
    if not os.path.exists(file_path):
        return None
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the utilization table section
    lines = content.split('\n')
    
    # Look for the table rows
    in_table = False
    for i, line in enumerate(lines):
        # Detect table start (header with dashes)
        if '+---' in line and 'Instance' in lines[i-1] if i > 0 else False:
            in_table = True
            continue
        
        if in_table and target_module in line:
            # Split by '|' and clean up whitespace
            parts = [p.strip() for p in line.split('|')]
            
            # parts[1] is Module name
            # parts[2] is Total LUTs
            # parts[6] is FFs
            # parts[7] is RAMB36
            
            if len(parts) >= 8:
                module_name = parts[2]
                total_luts = parts[3]
                ffs = parts[7]
                ramb36 = parts[8]
                
                return {
                    'Module': module_name,
                    'Total LUTs': int(total_luts),
                    'FFs': int(ffs),
                    'RAMB36': int(ramb36)
                }
    
    return None

def parse_timing_slack(file_path):
    """
    Parse Vivado timing report and extract the Slack value.
    
    Args:
        file_path: Path to the timing report file
    
    Returns:
        Float value of the slack in nanoseconds, or None if not found
    """
    if not os.path.exists(file_path):
        return None
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Search for the Slack line using regex
    # Pattern matches "Slack (MET) :" or "Slack (VIOLATED) :" followed by the value
    slack_pattern = r'Slack\s+\((MET|VIOLATED)\)\s*:\s*([-+]?\d+\.?\d*)ns'
    
    match = re.search(slack_pattern, content)
    
    if match:
        slack_status = match.group(1)  # MET or VIOLATED
        slack_value = float(match.group(2))
        return {
            'status': slack_status,
            'slack_ns': slack_value
        }
    
    return None


def parse_parameter_string(param_string):
    """
    Parse a parameter string and extract all numeric values.
    
    Example input: "DeployParametricDepth8Trees10Frq166Paths3Attr3Sets3"
    
    Returns:
        Dictionary with parameter names and their values
    """
    # Define patterns for each parameter
    patterns = {
        'depth': r'Depth(\d+)',
        'trees': r'Trees(\d+)',
        'frequency': r'Frq(\d+)',
        'paths': r'Paths(\d+)',
        'attr': r'Attr(\d+)',
        'sets': r'Sets(\d+)'
    }
    
    result = {}
    
    for param_name, pattern in patterns.items():
        match = re.search(pattern, param_string)
        if match:
            result[param_name] = int(match.group(1))
    
    return result

# Example usage
if __name__ == "__main__":
    deploys = os.listdir("../Deploys")
    deploys.sort()
    dic = {}
    dic['pes_per_path'] = []
    dic['n_paths'] = []
    dic['sample_dim'] = []
    dic['LUTs'] = []
    dic['FFs'] = []
    dic['BRAMs'] = []
    dic['Ps8LUTs'] = []
    dic['Ps8FFs'] = []
    dic['WNS'] = []
    dic['Outlier'] = []
    for deploy in deploys:
        params = parse_parameter_string(deploy)
        if 'sets' in params.keys():
            pes_per_path = int(params['depth']*(params['sets']/params['paths']))
        else:
            pes_per_path = int(params['depth'])
        n_paths = int(params['paths'])
        sample_dim = 224 + params['attr']*32
        resource_file_path = f"../Deploys/{deploy}/utilization_report.txt"
        timing_file_path = f"../Deploys/{deploy}/timing_report.txt"
        target_module = "design_2_TreePEsWrapper_0_0_TreePEsWrapper"
        target_module2 = "design_2_ps8_0_axi_periph_0"
        
        result_tw = parse_vivado_utilization(resource_file_path, target_module)
        result_ps = parse_vivado_utilization(resource_file_path, target_module2)
        result_timing = parse_timing_slack(timing_file_path)
        
        if result_tw and result_ps and result_timing:
            dic['pes_per_path'].append(pes_per_path)
            dic['n_paths'].append(n_paths)
            dic['sample_dim'].append(sample_dim)
            dic['LUTs'].append(result_tw['Total LUTs'])
            dic['FFs'].append(result_tw['FFs'])
            dic['BRAMs'].append(result_tw['RAMB36'])
            dic['Ps8LUTs'].append(result_ps['Total LUTs'])
            dic['Ps8FFs'].append(result_ps['FFs'])
            dic['WNS'].append(int(result_timing['slack_ns']*1000))
            dic['Outlier'].append(False)
        else:
            print(f"Reports not found in {deploy}.")

    df = pd.DataFrame.from_dict(dic)
    df.to_csv("resources_consumption_dataset_et.csv",index=False)