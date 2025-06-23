import torch
import torch.nn as nn

__all__ = ['TemporalShift', 'Res2TSMBlock']

class TemporalShift(nn.Module):
    def __init__(self, channels, shift_div=8):
        super().__init__()
        self.fold = channels // shift_div

    def forward(self, x):
        B, C, T, F = x.size()
        t = x.permute(0, 2, 1, 3).contiguous()      # B×T×C×F
        out = torch.zeros_like(t)
        out[:, :-1, :self.fold, :]            = t[:, 1:, :self.fold, :]
        out[:, 1:,  self.fold:2*self.fold, :] = t[:, :-1, self.fold:2*self.fold, :]
        out[:, :, 2*self.fold:, :]            = t[:, :, 2*self.fold:, :]
        return out.permute(0, 2, 1, 3)         # B×C×T×F

class Res2TSMBlock(nn.Module):
    def __init__(self, channels, scale=4, shift_div=8):
        super().__init__()
        assert channels % scale == 0, "channels must be divisible by scale"
        self.scale          = scale
        self.width          = channels // scale
        self.temporal_shift = TemporalShift(channels, shift_div)
        # one depthwise conv per branch except the first
        self.convs = nn.ModuleList([
            nn.Conv2d(self.width, self.width,
                      kernel_size=(3,1), padding=(1,0),
                      groups=self.width, bias=False)
            for _ in range(scale-1)
        ])
        self.bn  = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # 1) global TSM
        x = self.temporal_shift(x)            # B×C×T×F

        # 2) Res2Net-style split + fuse
        splits = torch.split(x, self.width, dim=1)
        y = splits[0]
        outs = [y]
        for i in range(1, self.scale):
            sp = splits[i]
            sp = sp + y                       # add only the last output
            sp = self.convs[i-1](sp)          # depthwise conv
            y = sp                            # update for next branch
            outs.append(sp)

        out = torch.cat(outs, dim=1)         # B×C×T×F
        out = self.bn(out)
        return self.act(out)
