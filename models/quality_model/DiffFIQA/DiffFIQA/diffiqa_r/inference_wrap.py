from typing import Any 

from models.quality_model.DiffFIQA.DiffFIQA.diffiqa_r.utils import * 

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