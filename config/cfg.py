import os 

class config:
    name_dataset = "XQLFW" 
    # name_quality_model = "DiffFIQA"
    name_quality_model = "ViT_FIQA_FLIP"
    # name_face_recog_model = "Elastic"
    name_face_recog_model = "Imintv5"
    # name_face_recog_model = "IMINT_V5_PLUS"

    # dataset infor
    # dataset_original =  "/data/disk2/tanminh/CR-FIQA/data/XQLFW/xqlfw_aligned_112"
    # pairs_list_path_original =  "/data/disk2/tanminh/CR-FIQA/data/XQLFW/xqlfw_pairs.txt"

    # size image 
    image_size_quality = 224

    # save process dataset infor
    data_folder = f"./data/processed_{name_dataset}"
    # os.makedirs(data_folder)
    output_path_dir_images = os.path.join(data_folder, "images")

    image_path_list = os.path.join(data_folder, "image_path_list.txt")
    pair_list_path = os.path.join(data_folder,"pair_list.txt") # file contains name of two faces and value 1 (genuie)  or 0 (imposter)
    path_score = os.path.join(data_folder, f"{name_quality_model}_scores.txt") # file contains the name of a face and the similarlity score
    
    # Model extract features 
    pretrained_face_recog = "/data/disk2/tanminh/CR-FIQA/pretrained/295672backbone.pth" 
    embeding_folder = os.path.join(data_folder, f"{name_face_recog_model}_embedding")


    # Model Quality CR-FIQA
    # pretrain_quality_model = "/data/disk2/tanminh/CR-FIQA/pretrained/32572backbone.pth"
    pretrain_quality_model = "pretrained/181952backbone.pth"


    # FMR
    FMR = 1e-3


    pass 
