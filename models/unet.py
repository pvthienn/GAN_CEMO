"""
based-Unet architecture of https://github.com/MStypulkowski/diffused-heads
audio-encoder refers https://github.com/Rudrabha/Wav2Lip
unet-discriminator refers at https://github.com/boschresearch/unetgan?tab=readme-ov-file
"""
import torch
import torch.nn as nn

from models.blocks import (
    zero_module, CondResBlock, ResBlock, CondSequential, 
    AttentionBlock, CondAttentionBlock, Upsample, Downsample
) #models.
from functools import partial
from math import floor, log2
from linear_attention_transformer import ImageLinearAttention
class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.SiLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)

class nonorm_Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            )
        self.act = LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)

class Conv2dTranspose(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, output_padding=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.SiLU()

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)
class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        
        if noise is None:
             batch, _, height, width = image.shape
             noise = image.new_empty(batch, 1, height, width).normal_()

        return self.weight * noise
class GENUNet(nn.Module):
    def __init__(self, image_size, in_channels, model_channels, out_channels, num_res_blocks, attention_resolutions,
                dropout=0, channel_mult=(1, 2, 3), num_heads=1, num_head_channels=-1, resblock_updown=False,
                audio_condition_type='attention', precision=32, n_motion_frames=0, grayscale_motion=False,
                n_audio_motion_embs=0):
        super(GENUNet, self).__init__()

        self.image_size = image_size
     
        self.audio_condition_type = audio_condition_type
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = [image_size // res for res in attention_resolutions]
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.dtype = torch.float32 if precision == 32 else torch.float16
        self.n_audio_motion_embs = n_audio_motion_embs
        self.img_channels = in_channels

        self.motion_channels = 1 if grayscale_motion else 3
        self.in_channels = in_channels
        self.noise=NoiseInjection()
        self.emo_emb= nn.Sequential(
        nn.Linear(6, 8*model_channels),
        nn.SiLU(),
        nn.Linear(8*model_channels, 8*model_channels),
        nn.SiLU())        
        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [CondSequential(nn.Conv2d(self.in_channels, ch, 3, padding=1))]
        )
        
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [CondResBlock(ch, dropout, out_channels=int(mult * model_channels), audio_condition_type=audio_condition_type)]

                ch = int(mult * model_channels)
                if ds in self.attention_resolutions:
                    if audio_condition_type == 'attention':
                        layers.append(CondAttentionBlock(ch, 4 * model_channels, image_size // ds, num_heads=num_heads, num_head_channels=num_head_channels))
                    else:
                        layers.append(AttentionBlock(ch, num_heads=num_heads, num_head_channels=num_head_channels))

                self.input_blocks.append(CondSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    CondSequential(CondResBlock(ch,dropout, out_channels=out_ch, down=True, audio_condition_type=audio_condition_type))
                        if resblock_updown
                        else Downsample()
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch
        
        if audio_condition_type == 'attention':
            attention_layer = CondAttentionBlock(ch, 4 * model_channels, image_size // ds, num_heads=num_heads, num_head_channels=num_head_channels)
        else:
            attention_layer = AttentionBlock(ch, num_heads=num_heads, num_head_channels=num_head_channels)
        self.middle_block = CondSequential(
            CondResBlock(ch,dropout, audio_condition_type=audio_condition_type),
            attention_layer,
            CondResBlock(ch, dropout, audio_condition_type=audio_condition_type)
        )

        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    CondResBlock(ch + ich, dropout, out_channels=int(model_channels * mult), audio_condition_type=audio_condition_type)
                ]
                ch = int(model_channels * mult)
                if ds in self.attention_resolutions:
                    if audio_condition_type == 'attention':
                        layers.append(CondAttentionBlock(ch, 4 * model_channels, image_size // ds, num_heads=num_heads, num_head_channels=num_head_channels))
                    else:
                        layers.append(AttentionBlock(ch, num_heads=num_heads, num_head_channels=num_head_channels))
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        CondResBlock(ch, dropout, out_channels=out_ch, up=True, audio_condition_type=audio_condition_type)
                        if resblock_updown
                        else Upsample()
                    )
                    ds //= 2
                self.output_blocks.append(CondSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(input_ch, out_channels, 3, padding=1)),
        )
        self.ae_dim= nn.Conv2d(1024, 256, 1, padding=0)
        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
        )
    def forward(self, x, emo_emb, audio_emb):
        B = audio_emb.size(0)
        emo_emb = emo_emb.unsqueeze(1).repeat(1, 5, 1) #(B, T, 6) 
        input_dim_size = len(x.size())
        a=audio_emb
        e=emo_emb
       
        if input_dim_size > 4:
            a = torch.cat([a[:, i] for i in range(a.size(1))], dim=0)
            e = torch.cat([e[:, i] for i in range(e.size(1))], dim=0) #(B*T, 6)
            x = torch.cat([x[:, :, i] for i in range(x.size(2))], dim=0)
        audio_em = self.audio_encoder(a) # B*T, 512, 1, 1
        emo_em = self.emo_emb(e)
        emo_em = emo_em.reshape(-1, 512, 1, 1)
        
        ae_emb= torch.cat([audio_em, emo_em], dim=1)
        noise=self.noise(x)
        x = x+noise
        
        if self.audio_condition_type in ['attention'] and audio_emb is not None:
            emb = ae_emb
        else:
            print('error')

        hs = []
        h = x
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)

        h = self.middle_block(h, emb)

        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        out=self.out(h)
        if input_dim_size>4:
            outputs = torch.split(out, B, dim=0) # [(B, C, H, W)]
            outputs = torch.stack(outputs, dim=2) # (B, C, T, H, W)
            outputs = outputs.clamp(0, 1)
        else:
            outputs=out
            outputs = outputs.clamp(0, 1)
        return outputs
    def generate(self, x, emo_emb, audio_emb):
        B = audio_emb.size(0)
        input_dim_size = len(x.size())
        a=audio_emb
        e=emo_emb
        if input_dim_size > 4:
            a = torch.cat([a[:, i] for i in range(a.size(1))], dim=0)
            e = torch.cat([e[:, i] for i in range(e.size(1))], dim=0) #(B*T, 6)
            x = torch.cat([x[:, :, i] for i in range(x.size(2))], dim=0)
        audio_em = self.audio_encoder(a) # B*T, 512, 1, 1
        emo_em = self.emo_emb(e)
        emo_em = emo_em.reshape(-1, 512, 1, 1)
        ae_emb= torch.cat([audio_em, emo_em], dim=1)
        noise=self.noise(x)
        x = x+noise
        
        if self.audio_condition_type in ['attention'] and audio_emb is not None:
            emb = ae_emb
        else:
            print('error')

        hs = []
        h = x
        ###print('channel', self.in_channels)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)

        h = self.middle_block(h, emb)

        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        out=self.out(h)
        if input_dim_size>4:
            outputs = torch.split(out, B, dim=0) # [(B, C, H, W)]
            outputs = torch.stack(outputs, dim=2) # (B, C, T, H, W)
            outputs = outputs.clamp(0, 1)
        else:
            outputs=out
            outputs = outputs.clamp(0, 1)
        return outputs
class Rezero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        return self.fn(x) * self.g
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return self.fn(x) + x
attn_and_ff = lambda chan: nn.Sequential(*[
    Residual(Rezero(ImageLinearAttention(chan, norm_queries = True))),
    Residual(Rezero(nn.Sequential(nn.Conv2d(chan, chan * 2, 1), nn.SiLU(), nn.Conv2d(chan * 2, chan, 1))))
])

def double_conv(chan_in, chan_out):
    return nn.Sequential(
        nn.Conv2d(chan_in, chan_out, 3, padding=1),
        nn.SiLU(),
        nn.Conv2d(chan_out, chan_out, 3, padding=1),
        nn.SiLU()
    )

class DownBlock(nn.Module):
    def __init__(self, input_channels, filters, downsample=True):
        super().__init__()
        self.conv_res = nn.Conv2d(input_channels, filters, 1, stride = (2 if downsample else 1))

        self.net = double_conv(input_channels, filters)
        self.down = nn.Conv2d(filters, filters, 3, padding = 1, stride = 2) if downsample else None

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        unet_res = x

        if self.down is not None:
            x = self.down(x)

        x = x + res
        return x, unet_res
class UpBlock(nn.Module):
    def __init__(self, input_channels, filters):
        super().__init__()
        self.conv_res = nn.ConvTranspose2d(input_channels // 2, filters, 1, stride = 2)
        self.net = double_conv(input_channels, filters)
        self.up = nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=False)
        self.input_channels = input_channels
        self.filters = filters

    def forward(self, x, res):
        *_, h, w = x.shape
        conv_res = self.conv_res(x, output_size = (h * 2, w * 2))
        x = self.up(x)
        x = torch.cat((x, res), dim=1)
        x = self.net(x)
        x = x + conv_res
        return x

class Discriminator(nn.Module):
    def __init__(self, image_size=128, network_capacity = 16, transparent = False, fmap_max = 512):
        super().__init__()
        num_layers = int(log2(image_size) - 3)
        num_init_filters = 3 if not transparent else 4

        blocks = []
        filters = [num_init_filters] + [(network_capacity) * (2 ** i) for i in range(num_layers + 1)]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        filters[-1] = filters[-2]

        chan_in_out = list(zip(filters[:-1], filters[1:]))
        chan_in_out = list(map(list, chan_in_out))

        down_blocks = []
        attn_blocks = []

        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            num_layer = ind + 1
            is_not_last = ind != (len(chan_in_out) - 1)

            block = DownBlock(in_chan, out_chan, downsample = is_not_last)
            down_blocks.append(block)

            attn_fn = attn_and_ff(out_chan)
            attn_blocks.append(attn_fn)

        self.down_blocks = nn.ModuleList(down_blocks)
        self.attn_blocks = nn.ModuleList(attn_blocks)

        last_chan = filters[-1]

        self.to_logit = nn.Sequential(
            nn.SiLU(),
            nn.AvgPool2d(image_size // (2 ** num_layers)),
            nn.Flatten(1),
            nn.Linear(last_chan, 1)
        )

        self.conv = double_conv(last_chan, last_chan)

        dec_chan_in_out = chan_in_out[:-1][::-1]
        self.up_blocks = nn.ModuleList(list(map(lambda c: UpBlock(c[1] * 2, c[0]), dec_chan_in_out)))
        self.conv_out = nn.Conv2d(3, 1, 1)

    def forward(self, x):
        b, *_ = x.shape

        residuals = []

        for (down_block, attn_block) in zip(self.down_blocks, self.attn_blocks):
            x, unet_res = down_block(x)
            residuals.append(unet_res)

            if attn_block is not None:
                x = attn_block(x)

        x = self.conv(x) + x
        enc_out = self.to_logit(x)

        for (up_block, res) in zip(self.up_blocks, residuals[:-1][::-1]):
            x = up_block(x, res)

        dec_out = self.conv_out(x)
        return enc_out.squeeze(), dec_out
