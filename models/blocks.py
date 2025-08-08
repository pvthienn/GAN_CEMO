import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


class CondSequential(nn.Sequential):
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, CondResBlock) or isinstance(layer, CondAttentionBlock):
                x = layer(x, emb)
                ###print('***')
            else:
                x = layer(x)
                ###print('only x')
        return x


class Upsample(nn.Module):
    def __init__(self, scale_factor=2):
        super(Upsample, self).__init__()

        self.scale_factor = scale_factor

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        return x

    
class Downsample(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super(Downsample, self).__init__()

        self.avg_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        return self.avg_pool(x)


 


class ResBlock(nn.Module):
    def __init__(self, channels, dropout, out_channels=None, up=False, down=False):
        super(ResBlock, self).__init__()
        
        self.channels = channels
        self.dropout = dropout
        self.out_channels = out_channels or channels

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample()
            self.x_upd = Upsample()
        elif down:
            self.h_upd = Downsample()
            self.x_upd = Downsample()
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1))
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

    def forward(self, x):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        h = self.out_layers(h)

        return self.skip_connection(x) + h


class CondResBlock(nn.Module):
    def __init__(self, channels, dropout, out_channels=None, up=False, down=False, audio_condition_type=None):
        super(CondResBlock, self).__init__()
        
        self.channels = channels
        
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.audio_condition_type = audio_condition_type

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample()
            self.x_upd = Upsample()
        elif down:
            self.h_upd = Downsample()
            self.x_upd = Downsample()
        else:
            self.h_upd = self.x_upd = nn.Identity()

       

        
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1))
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

    def forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        
        if self.audio_condition_type in ['attention']:
            ae_emb = emb
        else:
            print('error')
    

        
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1, num_head_channels=-1):
        super(AttentionBlock, self).__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (channels % num_head_channels == 0), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)

        self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class CondAttentionBlock(nn.Module):
    def __init__(self, channels, audio_dim, spatial_dim, num_heads=1, num_head_channels=-1):
        super(CondAttentionBlock, self).__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (channels % num_head_channels == 0), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        
        #self.audio_emb_proj = nn.Linear(audio_dim, spatial_dim**2)
        #self.audio_emb_proj = nn.Linear(audio_dim, spatial_dim**2)
        self.norm_x = nn.GroupNorm(32, channels)
        self.norm_emb = nn.GroupNorm(32, channels)
        self.q = nn.Conv1d(channels, channels, 1)
        self.kv = nn.Conv1d(channels, channels * 2, 1)

        self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x, emb):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        #print('c', c)
        
        ae_emb = emb.reshape(b, -1)
        #print('emb', ae_emb.shape)
        #ae_emb = self.audio_emb_proj(ae_emb)
        ae_emb = ae_emb.unsqueeze(1).expand(-1, c, -1)

        q = self.q(self.norm_x(x))
        kv = self.kv(self.norm_emb(ae_emb))
        qkv = torch.cat([q, kv], dim=1)

        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttentionLegacy(nn.Module):
    def __init__(self, n_heads):
        super(QKVAttentionLegacy, self).__init__()
        self.n_heads = n_heads

    def forward(self, qkv, dtype=torch.float32):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = torch.softmax(weight.type(dtype), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)
