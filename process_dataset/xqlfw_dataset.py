import os 
import cv2 

from config.cfg import config 

def copy_img(output_path, dataset_folder, name_person, img_id):
    img_name = f"{name_person}_{str(img_id).zfill(4)}.jpg"
    tmp_path = os.path.join(dataset_folder, name_person, img_name)
    img = cv2.imread(tmp_path)
    os.makedirs(output_path, exist_ok= True)
    cv2.imwrite(
        os.path.join(output_path, img_name), 
        img
    )
    return img_name, os.path.abspath(tmp_path)

def process_xqlf_pairs(output_path_dir, image_path_list, pair_list, dataset_folder, pairs_list_path_original): 
    '''
    
    Argument: 
        output_path_dir: (str) dir save all images
        image_path_list: (str) file contains many paths that link to image 
        pair_list: (str) contain two name file and 1(if same person) / 0 (if different)
        dataset_folder: (str) contains many folders whose name is a name person and contains image
        pairs_list_path: (str) has two situation, the first is contain one name and two id, the second is two name and two id
    
    '''

    f = open(pairs_list_path_original, "r")
    os.makedirs(os.path.dirname(image_path_list), exist_ok= True)
    os.makedirs(os.path.dirname(pair_list), exist_ok= True)
    img_path_file = open(image_path_list, "w") 
    pair_list_file = open(pair_list, "w")

    for line in f.readlines()[1:]:
        pair = line.strip().split() 

        if len(pair) == 3:
            img_name1, absolute_path1 = copy_img(
                output_path= output_path_dir, 
                dataset_folder= dataset_folder, 
                name_person= pair[0], 
                img_id= pair[1]
            )
            img_name2, absolute_path2 = copy_img(
                output_path= output_path_dir, 
                dataset_folder= dataset_folder,
                name_person= pair[0],
                img_id= pair[2],
            )
        else:
            img_name1, absolute_path1 = copy_img(
                output_path= output_path_dir,
                dataset_folder= dataset_folder,
                name_person= pair[0], 
                img_id= pair[1]
            )
            img_name2, absolute_path2 = copy_img(
                output_path= output_path_dir,
                dataset_folder= dataset_folder,
                name_person= pair[2],
                img_id= pair[3]
            )

        img_path_file.write(absolute_path1 + "\n")
        img_path_file.write(absolute_path2 + "\n")
        pair_list_file.write(f"{img_name1} {img_name2} {int(len(pair)== 3)}\n")
    
    img_path_file.close() 
    pair_list_file.close() 
