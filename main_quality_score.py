import torch
import glob
from models.quality_model.CRFIQA.crfiqa import CR_FIQA_Model
from models.quality_model.extract_score import extract_scores, save_scores
from config.cfg import config

# ====================== Model CR-FIQA
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

elif config.name_quality_model.lower() == "cr_fiqa_ontop":
    # ====================== Model CR-FIQA-ONTOP ========================
    from models.quality_model.Custom.crfiqa_ontop import CRFIQA_ontop

    model = CRFIQA_ontop(
        pretrained_backbone="/data/disk2/tanminh/Evaluate_FIQA_EVR/pretrained/2000backbone.pth",
        pretrained_head="/data/disk2/tanminh/Evaluate_FIQA_EVR/pretrained/2000header.pth",
        device="cuda",
    )
    # ====================== Model CR-FIQA-ONTOP ========================
elif config.name_quality_model.lower() == str("CR_ONTOP_10K_MULTI_FC_KL").lower():
    from models.quality_model.Custom.crfiqa_ontop import CRFIQA_ontop_multi_fc
    model = CRFIQA_ontop_multi_fc(
        pretrained_backbone=None,
        # pretrained_head="pretrained/pure_l1_multi_fc_kl_div_280000header.pth",
        pretrained_head="/data/disk2/tanminh/Evaluate_FIQA_EVR/pretrained/norm_cosine_79000header.pth",
        device="cuda",
    )
    model.eval() 
elif config.name_quality_model.lower() == str("CR_ONTOP_10K_MULTI_FC_KL_NORM").lower():
    from models.quality_model.Custom.crfiqa_ontop import CRFIQA_ontop_multi_fc
    model = CRFIQA_ontop_multi_fc(
        pretrained_backbone=None,
        # pretrained_head="pretrained/pure_l1_multi_fc_kl_div_280000header.pth",
        pretrained_head="/data/disk2/tanminh/Evaluate_FIQA_EVR/pretrained/norm_cosine_266000header.pth",
        device="cuda",
    )
    model.eval() 
elif config.name_quality_model.lower() == str("CR_ONTOP_10K_MULTI_FC_KL_FIXBUG").lower():
    from models.quality_model.Custom.crfiqa_ontop import CRFIQA_ontop_multi_fc
    model = CRFIQA_ontop_multi_fc(
        pretrained_backbone=None,
        # pretrained_head="pretrained/pure_l1_multi_fc_kl_div_280000header.pth",
        pretrained_head="/data/disk2/tanminh/Evaluate_FIQA_EVR/pretrained/norm_cosine_58000header_truth.pth",
        device="cuda",
    )
    model.eval() 
elif config.name_quality_model.lower() == str("CR_ONTOP_SUB_NNCCS").lower():
    from models.quality_model.Custom.crfiqa_ontop import CRFIQA_ontop_multi_fc
    model = CRFIQA_ontop_multi_fc(
        pretrained_backbone=None,
        # pretrained_head="pretrained/pure_l1_multi_fc_kl_div_280000header.pth",
        pretrained_head="/data/disk2/tanminh/Evaluate_FIQA_EVR/pretrained/norm_cosine_298000header_tadd_loss_regress.pth",
        device="cuda",
    )
    model.eval() 

elif config.name_quality_model.lower() == str("ViT_FIQA").lower():
    from models.quality_model.ViT.vit_fiqa import VIT_FIQA
    model = VIT_FIQA()
    model.load_state_dict(torch.load("pretrained/vit_2100header.pth"))
    model.eval() 


else:
    raise ValueError("Please assign true name quality model")


image_path_list = glob.glob(config.output_path_dir_images + "/*.jpg")

output_scores = extract_scores(
    model=model, image_path_list=image_path_list, batch_size= 4, device="cuda"
)

print(output_scores)

save_scores(
    output_file=config.path_score, ls_scores=output_scores, list_name=image_path_list
)
