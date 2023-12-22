import torch
import glob
from models.quality_model.CRFIQA.crfiqa import CR_FIQA_Model
from models.quality_model.extract_score import extract_scores, save_scores
from config.cfg import config

# ====================== Model CR-FIQA
print("[INFO] Using dataset: ", config.name_dataset.lower())
print("[INFO] Using quality model: ", config.name_quality_model.lower())
if config.name_quality_model.lower() == "cr-fiqa":
    model = CR_FIQA_Model(
        pretrained=config.pretrain_quality_model, arch="r100")

elif config.name_quality_model.lower() == "fiq_imint":
    # ====================== Model FIQ IMINT ========================
    from models.quality_model.Imintv5.imint import ONNX_FIQ_IMINT

    model = ONNX_FIQ_IMINT(
        path="/data/disk2/tanminh/Evaluate_FIQA_EVR/pretrained/model_fiq_imintv5.onnx"
    )
elif config.name_quality_model.lower() == "DiffFIQA".lower():
    # ====================== Model FIQ IMINT ========================
    from models.quality_model.DiffFIQA.DiffFIQA.diffiqa_r.inference_wrap import InferenceDiffFiQA

    model = InferenceDiffFiQA(
        file_config="models/quality_model/DiffFIQA/DiffFIQA/diffiqa_r/configs/inference_config.yaml"
    )

elif config.name_quality_model.lower() == "cr_fiqa_ontop":
    # ====================== Model CR-FIQA-ONTOP ========================
    from models.quality_model.Custom.crfiqa_ontop import CRFIQA_ontop

    model = CRFIQA_ontop(
        pretrained_backbone="/data/disk2/tanminh/Evaluate_FIQA_EVR/pretrained/2000backbone.pth",
        pretrained_head="/data/disk2/tanminh/Evaluate_FIQA_EVR/pretrained/2000header.pth",
        device="cuda",
    )
elif config.name_quality_model.lower() == str("ViT_FIQA_NO_FLIP").lower():
    # ====================== Model CR-FIQA-ONTOP ========================
    from models.quality_model.ViT.vit_fiqa import VIT_FIQA

    model = VIT_FIQA()
    model.load_state_dict(torch.load("/home/data2/tanminh/FIQA/FIQA/output/l2r_vit_flip_from_scratch_5k_id/70491header.pth"))
    model.eval()
elif config.name_quality_model.lower() == str("ViT_FIQA_FLIP").lower():
    # ====================== Model CR-FIQA-ONTOP ========================
    from models.quality_model.ViT.vit_fiqa import VIT_FIQA

    model = VIT_FIQA()
    model.load_state_dict(torch.load("/home/data2/tanminh/FIQA/FIQA/output/l2r_vit_flip_from_scratch_10k_id/79690header.pth"))
    model.eval()


else:
    raise ValueError("Please assign true name quality model")


if __name__ == "__main__":
    image_path_list = glob.glob(config.output_path_dir_images + "/*.jpg")

    output_scores = extract_scores(
        model=model, image_path_list=image_path_list, batch_size=2, device="cuda"
    )

    print(output_scores)

    save_scores(
        output_file=config.path_score, ls_scores=output_scores, list_name=image_path_list
    )
