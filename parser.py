import os
import numpy as np

def parse_H(file_path):
    if not os.path.isfile(file_path):
        print(f"File {file_path} does not exist.")
        return None
    
    data_list = []
    
    with open(file_path, 'r') as file:
        for line in file:
            if not line.strip():
                continue
            data_list.append(line.split(', '))
 
    if not data_list:
        return np.array([])
        
    H = np.array(data_list, dtype=float)
    
    return H