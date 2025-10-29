"""
Simple and stable conditional U-Net model.
"""
import torch
import torch.nn as nn
from .embeddings import SinusoidalPositionEmbeddings


class ResidualBlock(nn.Module):
    """Simple Residual Block"""
    def __init__(self, in_channels, out_channels, time_dim, cond_dim):
        super().__init__()
        
        self.time_mlp = nn.Linear(time_dim, out_channels)
        self.cond_mlp = nn.Linear(cond_dim, out_channels)
        
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, t, c):
        # Time and condition embedding
        t_emb = self.time_mlp(t)[:, :, None, None]
        c_emb = self.cond_mlp(c)[:, :, None, None]
        
        # Debug: Check tensor shapes
        if t_emb.shape[0] != c_emb.shape[0]:
            print(f"Batch size mismatch: t_emb={t_emb.shape}, c_emb={c_emb.shape}")
            print(f"t.shape={t.shape}, c.shape={c.shape}")
            print(f"x.shape={x.shape}")
        
        h = self.block1(x)
        h = h + t_emb + c_emb
        h = self.block2(h)
        
        return h + self.shortcut(x)


class DownBlock(nn.Module):
    """Downsampling block"""
    def __init__(self, in_channels, out_channels, time_dim, cond_dim):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_dim, cond_dim)
        self.down = nn.Conv2d(out_channels, out_channels, 4, stride=2, padding=1)
    
    def forward(self, x, t, c):
        x = self.res(x, t, c)
        return self.down(x), x  # return both downsampled and skip


class UpBlock(nn.Module):
    """Upsampling block"""
    def __init__(self, in_channels, skip_channels, out_channels, time_dim, cond_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1)
        self.res = ResidualBlock(in_channels + skip_channels, out_channels, time_dim, cond_dim)
    
    def forward(self, x, skip, t, c):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.res(x, t, c)


class SimpleUNet(nn.Module):
    """
    Simple and stable conditional U-Net
    """
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        cond_dim=6,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        time_dim=256,
    ):
        super().__init__()
        
        self.time_dim = time_dim
        self.cond_dim = cond_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # Condition embedding
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # Initial projection
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Downsampling
        self.downs = nn.ModuleList()
        channels = [base_channels]
        in_ch = base_channels
        
        for mult in channel_mults:
            out_ch = base_channels * mult
            self.downs.append(DownBlock(in_ch, out_ch, time_dim, time_dim))
            channels.append(out_ch)
            in_ch = out_ch
        
        # Middle
        self.mid = ResidualBlock(in_ch, in_ch, time_dim, time_dim)
        
        # Upsampling
        self.ups = nn.ModuleList()
        
        for mult in reversed(channel_mults):
            out_ch = base_channels * mult
            skip_ch = channels.pop()  # Get corresponding skip channel
            self.ups.append(UpBlock(in_ch, skip_ch, out_ch, time_dim, time_dim))
            in_ch = out_ch
        
        # Final projection
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, out_channels, 1)
        )
    
    def forward(self, x, t, cond):
        """
        Args:
            x: (B, 1, H, W) noisy image
            t: (B,) timestep
            cond: (B, cond_dim) conditions
        """
        # Embeddings
        t_emb = self.time_mlp(t)
        c_emb = self.cond_mlp(cond)
        
        # Initial
        x = self.init_conv(x)
        
        # Downsampling
        skips = []
        for down in self.downs:
            x, skip = down(x, t_emb, c_emb)
            skips.append(skip)
        
        # Middle
        x = self.mid(x, t_emb, c_emb)
        
        # Upsampling
        for up in self.ups:
            skip = skips.pop()
            x = up(x, skip, t_emb, c_emb)
        
        # Final
        return self.final_conv(x)

