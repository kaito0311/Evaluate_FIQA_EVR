from config.cfg import config
from process_dataset.CALFW.calfw import CALFW_Dataset
from process_dataset.CPLFW.cplfw import CPLFW_Dataset
from process_dataset.LFW.lfw import LFW_Dataset


# dataset = CALFW_Dataset("/data/disk2/tanminh/Evaluate_FIQA_EVR/dataset_evaluate/cplfw.bin")

# dataset.process(
#     dir_save_images= config.output_path_dir_images,
#     path_file_image_list= config.image_path_list,
#     path_file_image_pair= config.pair_list_path
# )
print("[INFO] Process name dataset: ", config.name_dataset.lower())
dataset = None
if config.name_dataset.lower() == "lfw":
    dataset = LFW_Dataset(
        "/data/disk2/tanminh/Evaluate_FIQA_EVR/dataset_evaluate/cfp_fp.bin"
    )
# elif config.name_dataset.lower() == 'xqlfw':
#     dataset = ...
elif config.name_dataset.lower() == "cplfw":
    dataset = CPLFW_Dataset(
        "dataset_bin/cplfw.bin",
    )
# elif config.name_dataset.lower() == "calfw":
#     ...
# elif config.name_dataset.lower() == "cfp_fp":
#     ...
else:
    raise ValueError("Please config true name dataset: ")

dataset.process(
    dir_save_images=config.output_path_dir_images,
    path_file_image_list=config.image_path_list,
    path_file_image_pair=config.pair_list_path,
)
