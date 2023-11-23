from config.cfg import config
from process_dataset.xqlfw_dataset import process_xqlf_pairs

# process_xqlf_pairs(
#     output_path_dir="data/processed_XQLFW/images/",
#     image_path_list="data/processed_XQLFW/image_path_list.txt",
#     pair_list= "data/processed_XQLFW/pairs_image_list.txt",
#     dataset_folder= "/data/disk2/tanminh/CR-FIQA/data/XQLFW/xqlfw_aligned_112",
#     pairs_list_path= "/data/disk2/tanminh/CR-FIQA/data/XQLFW/xqlfw_pairs.txt"
# )


process_xqlf_pairs(
    output_path_dir= config.output_path_dir_images,
    image_path_list= config.image_path_list, 
    pair_list= config.pair_list_path,
    dataset_folder= config.dataset_original,
    pairs_list_path_original= config.pairs_list_path_original
)

