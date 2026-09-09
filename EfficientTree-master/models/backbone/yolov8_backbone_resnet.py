import argparse
import sys
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = ROOT.relative_to(Path.cwd())  # relative

from ..backbone.common import *
from ..backbone.experimental import *
from utils.general import make_divisible
from ..loss.loss import *
from ..backbone.resnet import resnet34

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

LOGGER = logging.getLogger(__name__)

class YoloV8BackBone_resnet(nn.Module):
    def __init__(self, cfg):
        super(YoloV8BackBone_resnet, self).__init__()

        self.gd = cfg.Model.depth_multiple
        self.gw = cfg.Model.width_multiple

        # yolov8n/s
        if cfg.Model.depth_multiple == 0.33:
            self.channels_out = {'stage1': 64, 'stage2_1': 128, 'stage2_2': 128, 'stage3_1': 256, 'stage3_2': 256,
                                 'stage4_1': 512, 'stage4_2': 512, 'stage5': 1024, 'spp': 1024, 'csp1': 1024}
        # yolov8m
        elif cfg.Model.depth_multiple == 0.67:
            self.channels_out = {'stage1': 64, 'stage2_1': 128, 'stage2_2': 128, 'stage3_1': 256, 'stage3_2': 256,
                                 'stage4_1': 512, 'stage4_2': 512, 'stage5': 768, 'spp': 768, 'csp1': 768}
        # yolov8l/x
        else:
            self.channels_out = {'stage1': 64, 'stage2_1': 128, 'stage2_2': 128, 'stage3_1': 256, 'stage3_2': 256,
                                 'stage4_1': 512, 'stage4_2': 512, 'stage5': 512, 'spp': 512, 'csp1': 512}
        self.re_channels_out()

        if cfg.Model.Backbone.activation == 'SiLU': 
            CONV_ACT = 'silu'
            C_ACT = 'silu'
        elif cfg.Model.Backbone.activation == 'ReLU': 
            CONV_ACT = 'relu'
            C_ACT = 'relu'
        else:
            CONV_ACT = 'hard_swish'
            C_ACT = 'hard_swish'

        self.resnet = resnet34()
        self.sppf = SPPF(self.channels_out['csp1'], self.channels_out['spp'], 5, CONV_ACT)
        self.out_shape = {'C3_size': self.channels_out['stage3_2'],
                          'C4_size': self.channels_out['stage4_2'],
                          'C5_size': self.channels_out['spp']}

    def forward(self, x):

        output = self.resnet(x)
        sppf = self.sppf(output[3])
        return output[1], output[2], sppf

    def get_depth(self, n):
        return max(round(n * self.gd), 1) if n > 1 else n

    def get_width(self, n):
        return make_divisible(n * self.gw, 8)

    def re_channels_out(self):
        for k, v in self.channels_out.items():
            self.channels_out[k] = self.get_width(v)
