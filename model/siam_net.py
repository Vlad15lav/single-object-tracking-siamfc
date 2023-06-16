import torch.nn.functional as F

from torch import nn

class SiamFC(nn.Module):
    def __init__(self):
        super(SiamFC, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=2, bias=True),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5, stride=1, groups=2, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=1),
            
            nn.Conv2d(256, 384, kernel_size=3, stride=1, groups=1, bias=True),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            
            nn.Conv2d(384, 384, kernel_size=3, stride=1, groups=2, bias=True),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            
            nn.Conv2d(384, 256, kernel_size=3, stride=1, groups=2, bias=True),
        )

    def get_corr(self, embed_z, embed_x):
        b, c, h, w = embed_x.shape
        match_map = F.conv2d(embed_x.view(-1, b * c, h, w), embed_z, groups=b)
        match_map.view(b, -1, match_map.size(-2), match_map.size(-1))
        
        return match_map

    def forward(self, z, x):
        embed_z = self.feature_extractor(z)
        embed_x = self.feature_extractor(x)

        return self.get_corr(embed_z, embed_x)