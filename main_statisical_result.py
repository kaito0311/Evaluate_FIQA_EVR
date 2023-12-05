import os 

import numpy as np 
import pandas as pd



list_fnmr = os.listdir("/data/disk2/tanminh/Evaluate_FIQA_EVR/output_dir_change_range/fnmr")

results = [np.arange(0, 0.2, 0.01)] 
for name in list_fnmr:
    print()
    path = os.path.join("/data/disk2/tanminh/Evaluate_FIQA_EVR/output_dir_change_range/fnmr", name)
    results.append(np.load(path)) 
results = np.array(results).T 

df = pd.DataFrame(results, columns=["FNMR"] + list_fnmr)
df.to_csv("statisical.csv")


