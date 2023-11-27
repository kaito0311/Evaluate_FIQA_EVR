import os 

import numpy as np 

data= np.load("/data/disk2/tanminh/Evaluate_FIQA_EVR/dict_score.npy", allow_pickle= True)

file_output= open("FaceQAN_XQLFW_r160.txt", "w") 

for key in (data.item().keys()): 
    file_output.write(
        os.path.basename(key) + " " + str(data.item()[key]) + "\n"
    )
file_output.close()