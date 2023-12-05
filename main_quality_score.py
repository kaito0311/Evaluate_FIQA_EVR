import torch
import glob
from models.quality_model.CRFIQA.crfiqa import CR_FIQA_Model
from models.quality_model.extract_score import extract_scores, save_scores
from config.cfg import config

# ====================== Model CR-FIQA
print("[INFO] Using quality model: ", config.name_quality_model.lower())
if config.name_quality_model.lower() == "cf-fiqa":
    model = CR_FIQA_Model(
        pretrained=config.pretrain_quality_model, arch="r100")

elif config.name_quality_model.lower() == "fiq_imint":
    # ====================== Model FIQ IMINT ========================
    from models.quality_model.Imintv5.imint import ONNX_FIQ_IMINT

    model = ONNX_FIQ_IMINT(
        path="/data/disk2/tanminh/Evaluate_FIQA_EVR/pretrained/model_fiq_imintv5.onnx"
    )

elif config.name_quality_model.lower() == "cr_fiqa_ontop":
    # ====================== Model CR-FIQA-ONTOP ========================
    from models.quality_model.Custom.crfiqa_ontop import CRFIQA_ontop

    model = CRFIQA_ontop(
        pretrained_backbone="/data/disk2/tanminh/Evaluate_FIQA_EVR/pretrained/2000backbone.pth",
        pretrained_head="/data/disk2/tanminh/Evaluate_FIQA_EVR/pretrained/2000header.pth",
        device="cuda",
    )
elif config.name_quality_model.lower() == "cr_fiqa_ontop_4k":
    # ====================== Model CR-FIQA-ONTOP ========================
    from models.quality_model.Custom.crfiqa_ontop import CRFIQA_ontop

    model = CRFIQA_ontop(
        pretrained_backbone="/data/disk2/tanminh/Evaluate_FIQA_EVR/pretrained/4000backbone.pth",
        pretrained_head="/data/disk2/tanminh/Evaluate_FIQA_EVR/pretrained/4000header.pth",
        device="cuda",
    )

elif config.name_quality_model.lower() == "new_cr_fiqa":
    from models.quality_model.Custom.crfiqa_ontop import CRFIQA_ontop
    model = CRFIQA_ontop(
        pretrained_backbone=None,
        pretrained_head="/data/disk2/tanminh/Evaluate_FIQA_EVR/pretrained/ori_15030header",
        device="cuda",
    )

elif config.name_quality_model.lower() == "new_cr_fiqa_ori":
    from models.quality_model.Custom.crfiqa_ontop import CRFIQA_ontop
    model = CRFIQA_ontop(
        pretrained_backbone=None,
        pretrained_head="/data/disk2/tanminh/Evaluate_FIQA_EVR/pretrained/3340header.pth",
        device="cuda",
    )
elif config.name_quality_model.lower() == "new_cr_fiqa_ori_fi":
    from models.quality_model.Custom.crfiqa_ontop import CRFIQA_ontop
    model = CRFIQA_ontop(
        pretrained_backbone=None,
        pretrained_head="/data/disk2/tanminh/Evaluate_FIQA_EVR/pretrained/10k_4000header.pth",
        device="cuda",
    )
elif config.name_quality_model.lower() == str("CR_ONTOP_10K_9KITER").lower():
    from models.quality_model.Custom.crfiqa_ontop import CRFIQA_ontop
    model = CRFIQA_ontop(
        pretrained_backbone=None,
        pretrained_head="/data/disk2/tanminh/Evaluate_FIQA_EVR/pretrained/100k_ cosine_10000header.pth",
        device="cuda",
    )
elif config.name_quality_model.lower() == str("CR_ONTOP_10K_12KITER").lower():
    from models.quality_model.Custom.crfiqa_ontop import CRFIQA_ontop
    model = CRFIQA_ontop(
        pretrained_backbone=None,
        pretrained_head="/data/disk2/tanminh/Evaluate_FIQA_EVR/pretrained/100k_12000header.pth",
        device="cuda",
    )
elif config.name_quality_model.lower() == str("CR_ONTOP_10K_14KITER").lower():
    from models.quality_model.Custom.crfiqa_ontop import CRFIQA_ontop
    model = CRFIQA_ontop(
        pretrained_backbone=None,
        pretrained_head="/data/disk2/tanminh/Evaluate_FIQA_EVR/pretrained/100k_17000header.pth",
        device="cuda",
    )


else:
    raise ValueError("Please assign true name quality model")


image_path_list = glob.glob(config.output_path_dir_images + "/*.jpg")

output_scores = extract_scores(
    model=model, image_path_list=image_path_list, batch_size= 16, device="cuda"
)

print(output_scores)

save_scores(
    output_file=config.path_score, ls_scores=output_scores, list_name=image_path_list
)
