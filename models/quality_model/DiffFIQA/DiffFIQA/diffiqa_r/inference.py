
import os
import pickle
import argparse
from typing import Any

import torch 
from torch.utils.data import DataLoader

from tqdm import tqdm

from utils import *
from dataset import ImageDataset


@torch.no_grad()
def inference(args):

    # Seed all libraries to ensure consistency between runs.
    seed_all(args.base.seed)

    # Load the training FR model and construct the transformation
    model, trans = construct_full_model(args.model.config)
    model.load_state_dict(torch.load(args.model.weights))
    model.to(args.base.device).eval()

    # Construct the Image Dataloader 
    dataset = ImageDataset(args.dataset.loc, trans)
    dataloader = DataLoader(dataset, **args_to_dict(args.dataloader.params, {}))

    # Predict quality scores 
    d_res = {}
    for (name_batch, img_batch) in tqdm(dataloader, 
                                        desc=" Inference ", 
                                        disable=not args.base.verbose):
        print(img_batch.shape)
        img_batch = img_batch.to(args.base.device)
        preds = model(img_batch).detach().squeeze().cpu().numpy()
        d_res.update(dict(zip(name_batch, preds)))

    print(d_res)

    np.save("result.npy", d_res)

class InferenceDiffFiQA:
    def __init__(self, file_config):
        self.arguments =  parse_config_file(file_config)
        # Seed all libraries to ensure consistency between runs.
        seed_all(self.arguments.base.seed)
        # Load the training FR model and construct the transformation
        self.model, self.trans = construct_full_model(self.arguments.model.config)
        self.model.load_state_dict(torch.load(self.arguments.model.weights))
        self.model.to(self.arguments.base.device).eval()
    
    def __call__(self,batch_image, *args: Any, **kwds: Any) -> Any:
        preds = self.model(batch_image) 
        return preds
    def eval(self):
        return self.model.eval() 

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", 
        "--config",
        type=str,
        default= "configs/inference_config.yaml",
        help=' Location of the DifFIQA(R) inference configuration. '
    )
    args = parser.parse_args()
    arguments = parse_config_file(args.config)

    inference(arguments)