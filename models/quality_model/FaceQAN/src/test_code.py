import os 

import numpy as np 

data= np.load("dict_score.npy", allow_pickle= True)

file_output= open("FaceQAN_XQLFW.txt", "w") 

for key in (data.item().keys()): 
    file_output.write(
        os.path.basename(key) + " " + str(data.item()[key]) + "\n"
    )
file_output.close()