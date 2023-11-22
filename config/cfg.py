import os 

class config:
    name_dataset = "XQLFW" 
    name_model = "model"
    
    data_folder = "./data"
    embeding_folder = os.path.join(data_folder, f"{name_model}_embedding")

    pair_list_path = os.path.join(data_folder, name_dataset, "pair_list.txt") # file contains name of two faces and value 1 (genuie)  or 0 (imposter)
    path_score = os.path.join(data_folder, name_dataset, name_model, f"scores.txt") # file contains the name of a face and the similarlity score
     

    FMR = 1e-2


    pass 
