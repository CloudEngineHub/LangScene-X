
from random import randint

from .gaussian_field import GaussianField
from .preprocessor import Preprocessor


class FieldConstructionPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        if cfg.pipeline.mode == "train":
            self.preprocessor = Preprocessor(cfg)
        else:
            self.preprocessor = None
    
    def construct_field(self):
        self.preprocessor.preprocess()
        del self.preprocessor
        self.preprocessor = None

        self.gaussian_field = GaussianField(self.cfg)
        self.gaussian_field.train()

    
    def render_result(self):
        self.gaussian_field = GaussianField(self.cfg)
        self.gaussian_field.render()
        
    def eval(self):
        self.gaussian_field = GaussianField(self.cfg)
        self.gaussian_field.eval()

