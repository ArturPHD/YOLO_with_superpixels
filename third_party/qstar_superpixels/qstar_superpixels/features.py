import torch.nn as nn

class SPCN(nn.Module):
    def __init__(self, in_ch, hid_ch, ratio=4.0, bn:bool=False):
        super().__init__()
        r = int(round(ratio))
        self.ds = nn.Sequential(
            nn.Conv2d(in_ch, hid_ch, 1, 1, 0),
        )
        self.cv = nn.Sequential(
            nn.Conv2d(in_ch, r*hid_ch, 3, 2, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(r*hid_ch, hid_ch, 3, 2, 1),
            nn.UpsamplingBilinear2d(scale_factor=4)
        )
        self.act = nn.Sequential(
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.act(
            self.ds(x) + self.cv(x)
        )
