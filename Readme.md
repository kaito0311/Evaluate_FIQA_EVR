This source is created for evaluate Face Image Quality Assessment (FIQA) model by using EVR metric 
## Prepare 
1. Prepare dataset must contain following file and structure 
```
|+---- XQLFW_folder/
|      + ---- align_images/
|               + ---- Acamal_fmaklem/
                        + --- Acamal_fmaklem_0001.jpg
                        + --- Acamal_fmaklem_0002.jpg
                + ---- Abmalkdf_dafalm/
                        + --- Abmalkdf_dafalm_0001.jpg
                        + --- Abmalkdf_dafalm_002.jpg
                ...
        + ---- pair_image.txt
```
where format ```pair_image.txt``` is: 
```
Name_person id_1 id_2 
Name_person_1 id_1 Name_person_2 id_2
...   
```

## Step by step
0. Preprocess dataset [process_data](./process_data.py)
2. Run recognition model [main_extract_embed_face_recog.py](./main_extract_embed_face_recog.py) (pretrained) -> embed 
1. Run quality model [main_quality_score.py](./main_quality_score.py) (your own and modify output in file  [extract_score.py](./models/quality_model/extract_score.py))-> list quality.txt
3. Run Evaluate [main_visulize_results.py](./main_visulize_results.py) to visualize data 

## Structure folder (XQLFW)
1. Structure dataset after processed:
```
+-- data/process_XQLFW
|   +-- embedding_dir
|   +-- images/
|   image_path_list.txt
|   model_scores.txt
|   pair_list.txt
```
