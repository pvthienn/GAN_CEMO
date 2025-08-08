"""
Code is a simplified version of https://github.com/openai/guided-diffusion
"""
import torch
import torch.nn as nn

from models.blocks import (
    zero_module, CondResBlock, ResBlock, CondSequential, 
    AttentionBlock, CondAttentionBlock, TimestepEmbedding, Upsample, Downsample
)
class emotion_encoder(nn.Module):
    def __init__(self, ):
        super(emotion_encoder, self).__init__()
        self.emotion_encoder = nn.Sequential(
        nn.Linear(6, 4*512),
        nn.ReLU(inplace=True),
        nn.Linear(4*512, 4*512), 
        nn.ReLU(inplace=True))
    def forward(self, emotion):
            emotions=self.emotion_encoder(emotion)
            return emotions
class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()
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
        self.act = nn.LeakyReLU(0.01, inplace=True)

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
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)
class UNet(nn.Module):
    def __init__(self, image_size, in_channels, model_channels, out_channels, num_res_blocks, attention_resolutions,
                dropout=0, channel_mult=(1, 2, 4, 8), num_heads=1, num_head_channels=-1, resblock_updown=False, 
                id_condition_type='frame', audio_condition_type='pre_gn', precision=32, n_motion_frames=0, grayscale_motion=False,
                n_audio_motion_embs=0):
        super(UNet, self).__init__()

        self.image_size = image_size
        self.id_condition_type = id_condition_type
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
        self.n_motion_frames = n_motion_frames
        self.n_audio_motion_embs = n_audio_motion_embs
        self.img_channels = in_channels

        self.motion_channels = 1 if grayscale_motion else 3
        self.in_channels = in_channels #+ self.motion_channels #* n_motion_frames
        #if id_condition_type == 'frame':
        #    self.in_channels += 1 #in_channels
        
        time_embed_dim = model_channels * 4
        self.identity_encoder = None
        #if id_condition_type != 'frame':
        #    self.identity_encoder = IdentityEncoder(
        #        image_size, in_channels, model_channels, time_embed_dim, num_res_blocks, self.attention_resolutions, 
        #        dropout=dropout, channel_mult=channel_mult, num_heads=num_heads, num_head_channels=num_head_channels,
        #        resblock_updown=resblock_updown, precision=precision
        #    )
        
        self.time_embed = TimestepEmbedding(model_channels)
        
        self.audio_embed = nn.Sequential(
            nn.Linear(model_channels * (2 * n_audio_motion_embs + 1), 4 * model_channels),
            nn.SiLU(),
            nn.Linear(4 * model_channels, 4 * model_channels),
        )
        self.emo_emb= nn.Sequential(
        nn.Linear(6, 4*512),
        nn.SiLU(),
        nn.Linear(4*512, 4*512), 
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
                layers = [CondResBlock(ch, time_embed_dim, dropout, out_channels=int(mult * model_channels), id_condition_type=id_condition_type, audio_condition_type=audio_condition_type)]

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
                    CondSequential(CondResBlock(ch, time_embed_dim, dropout, out_channels=out_ch, down=True, id_condition_type=id_condition_type, audio_condition_type=audio_condition_type))
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
            CondResBlock(ch, time_embed_dim, dropout, id_condition_type=id_condition_type, audio_condition_type=audio_condition_type),
            attention_layer,
            CondResBlock(ch, time_embed_dim, dropout, id_condition_type=id_condition_type, audio_condition_type=audio_condition_type)
        )

        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    CondResBlock(ch + ich, time_embed_dim, dropout, out_channels=int(model_channels * mult), id_condition_type=id_condition_type, audio_condition_type=audio_condition_type)
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
                        CondResBlock(ch, time_embed_dim, dropout, out_channels=out_ch, up=True, id_condition_type=id_condition_type, audio_condition_type=audio_condition_type)
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
        self.audio_less= nn.Conv2d(512, 256, 1, padding=0)
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
            #thien add
            #nn.Linear(model_channels * (2 * n_audio_motion_embs + 1), 4 * model_channels),
            #nn.SiLU(),
            #nn.Linear(4 * model_channels, 4 * model_channels),
        )
    def forward(self, x, timesteps, emo_emb, motion_frames=None, audio_emb=None):
        ###print(emo_emb)
        ##thien add
        B = audio_emb.size(0)
        #print('thien test e 0', e.shape)
        #emotion = (B, 6)
        # repeating the same emotion for every frame
        emo_emb = emo_emb.unsqueeze(1).repeat(1, 5, 1) #(B, T, 6) 
        #print('thien test e 0', e.shape)
        input_dim_size = len(motion_frames.size())
        #print('thien test a 0', audio_emb.shape)
        a=audio_emb
        e=emo_emb
        m=motion_frames
        #print(a)
        if input_dim_size > 4:
            a = torch.cat([a[:, i] for i in range(a.size(1))], dim=0)
            #print(a.shape)
            e = torch.cat([e[:, i] for i in range(e.size(1))], dim=0) #(B*T, 6)
            m = torch.cat([m[:, :, i] for i in range(m.size(2))], dim=0)
            #icond = torch.cat([icond[:, :, i] for i in range(icond.size(2))], dim=0)
        #print('thien test e 0', emo_emb.shape)
        #print('thien test a 0', audio_emb.shape)
        audio_em = self.audio_encoder(a) # B*T, 512, 1, 1
        emo_em = self.emo_emb(e)
        # ee_needed =  torch.mean(emotion_embedding,0).unsqueeze(0)
        
        #emotion_embedding = emotion_embedding.view(-1,512,1,1)
        
        
        #
        t_emb = self.time_embed(timesteps, dtype=x.dtype)
        #emo_emb = self.emo_emb(emo_emb)
        if audio_emb is not None:
            #print('unet', audio_emb.shape)
            a_emb = audio_em.reshape(audio_em.shape[0], -1)
            #print('unet', a_emb.shape)
            a_emb = self.audio_embed(a_emb)
        else:
            a_emb = 0
            ###print('without frames of audio features')

        if motion_frames is not None:
            x = torch.cat([x, m], dim=1)
            ###print('motion')
            ###print(x.shape)ssss

        if self.id_condition_type == 'frame':

            '''print('t',t_emb.shape)
            print('a', a_emb.shape)
            print('emo', emo_emb.shape)'''
            #x = torch.cat([x, x_cond], dim=1)
            #if self.audio_condition_type == 'pre_gn':
            emb = t_emb + a_emb +emo_em
            ###print('toal emb', emb.shape)
            #elif self.audio_condition_type in ['post_gn', 'double_pre_gn', 'attention'] and audio_emb is not None:
            #    emb = (t_emb, a_emb)
            #else:
            #    raise NotImplemented(self.audio_condition_type)

        #elif self.id_condition_type == 'post_gn':
        #    x_emb = self.identity_encoder(x_cond)
        #    if self.audio_condition_type == 'pre_gn':
        #        emb = (t_emb + a_emb, x_emb)
        #    elif self.audio_condition_type in ['post_gn', 'double_pre_gn', 'attention'] and audio_emb is not None:
        #        emb = (t_emb, a_emb, x_emb)
        #    else:
        #        raise NotImplementedError(self.audio_condition_type)

        #elif self.id_condition_type == 'pre_gn':
        #    x_emb = self.identity_encoder(x_cond)
         #   if self.audio_condition_type == 'pre_gn':
         #       emb = x_emb + t_emb + a_emb
         #   elif self.audio_condition_type in ['post_gn', 'double_pre_gn', 'attention'] and audio_emb is not None:
         #       emb = (x_emb + t_emb, a_emb)
         #   else:
         #       raise NotImplementedError(self.audio_condition_type)
                
        else:
            raise NotImplementedError(self.id_condition_type)

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
        print('out', out.shape)
        outputs = torch.split(out, B, dim=0) # [(B, C, H, W)]
        print('output', type(outputs))

        outputs = torch.stack(out, dim=2) # (B, C, T, H, W)
        return outputs, out #self.out(h)


'''class IdentityEncoder(nn.Module):
    def __init__(self, image_size, in_channels, model_channels, out_dim, num_res_blocks, attention_resolutions,
                dropout=0, channel_mult=(1, 2, 4, 8), num_heads=4, num_head_channels=-1, resblock_updown=False, 
                precision=32):
        super(IdentityEncoder, self).__init__()

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_dim = out_dim
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.dtype = torch.float32 if precision == 32 else torch.float16

        ch = int(channel_mult[0] * model_channels)
        
        self.input_blocks = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(self.in_channels, ch, 3, padding=1))]
        )

        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, dropout, out_channels=int(mult * model_channels))]

                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads, num_head_channels=num_head_channels))

                self.input_blocks.append(nn.Sequential(*layers))

            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    nn.Sequential(ResBlock(ch, dropout, out_channels=out_ch, down=True))
                        if resblock_updown
                        else Downsample()
                )
                ch = out_ch
                ds *= 2

        self.middle_block = nn.Sequential(
            ResBlock(ch, dropout),
            AttentionBlock(ch, num_heads=num_heads, num_head_channels=num_head_channels),
            ResBlock(ch, dropout)
        )

        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            zero_module(nn.Conv2d(ch, out_dim, 1)),
            nn.Flatten()
        )

    def forward(self, x):
        for module in self.input_blocks:
            x = module(x)

        x = self.middle_block(x)

        return self.out(x)'''


if __name__ == '__main__':

    image_size = 64
    in_channels = 3
    model_channels = 512
    out_channels = 3 # 3 or 6 if sigma learnable
    num_res_blocks = 1
    attention_resolutions = (8, 4, 2)
    dropout = 0.1
    channel_mult = (1, 2, 3)
    num_heads = 4
    num_head_channels = -1
    resblock_updown = True

    unet = UNet(image_size, in_channels, model_channels, out_channels, num_res_blocks, attention_resolutions,
                dropout=dropout, channel_mult=channel_mult, num_heads=num_heads, num_head_channels=-1, resblock_updown=True, 
                id_condition_type='frame', precision=32).to('cuda')
    #print(unet)
    x = torch.randn(5, 3, 64, 64).to('cuda')
    x_motion = torch.randn(5, 3, 64, 64).to('cuda')
    t = torch.randint(10, (5,)).to('cuda')
    a = torch.randn(5, 512, 1, 1).to('cuda')
    e = torch.randint(0, 6, (5,) ).to('cuda')
    e = torch.nn.functional.one_hot(e, num_classes=6).float()

    out = unet(x, t, e,x_motion, a)
    print(out.shape)
