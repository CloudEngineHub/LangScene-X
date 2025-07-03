import logging
import os
import random
import warnings
from random import randint

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from field_construction.pipeline import FieldConstructionPipeline

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


@hydra.main(config_path="configs", config_name="field_construction", version_base=None)
def main(cfg: DictConfig):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler]
    )
    # ignore pil debug message.
    pil_logger = logging.getLogger("PIL")
    pil_logger.setLevel(logging.WARNING)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    warnings.filterwarnings("ignore", category=FutureWarning)

    setup_seed(42)
    
    pipeline = FieldConstructionPipeline(cfg)
    if cfg.pipeline.mode == "train":
        pipeline.construct_field()
    elif cfg.pipeline.mode == "render":
        pipeline.render_result()
    elif cfg.pipeline.mode == "eval":
        pipeline.eval()
    else:
        raise NotImplementedError

if __name__ == "__main__":
    main()
