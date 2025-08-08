import wandb
from tqdm import tqdm, trange
from math import ceil
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import Compose

from models.losses import gaussian_kl, discretized_gaussian_nll




class Diffusion(nn.Module):
    def __init__(
        self, nn_backbone, n_timesteps, in_channels, out_channels, image_size,
        precision=32, motion_transforms=None):
        super(Diffusion, self).__init__()

        self.nn_backbone = nn_backbone
        self.n_timesteps = n_timesteps
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.x_shape = (image_size, image_size)
        self.dtype = torch.float32 if precision == 32 else torch.float16

        self.motion_transforms = motion_transforms if motion_transforms else Compose([])

        self.timesteps = torch.arange(n_timesteps)
        self.beta = self.get_beta_schedule().type(self.dtype)
        self.set_params()

    def forward(self, x0, emo_emb, motion_frames=None, audio_emb=None):
        x0 = torch.cat([x0[:, :, i] for i in range(x0.size(2))], dim=0)
        x0, _= x0.chunk(2, 1)
        timesteps = torch.randint(self.n_timesteps, (x0.shape[0],)).to(x0.device)
        eps, xt = self.forward_diffusion(x0, timesteps)
        output, nn_out = self.nn_backbone(xt, timesteps, emo_emb, motion_frames=motion_frames, audio_emb=audio_emb)
        outp, nu = output.chunk(2, 1)
        losses = {}
        if self.out_channels == self.in_channels:
            eps_pred = nn_out
        else:
            eps_pred, nu = nn_out.chunk(2, 1)
            nn_out_frozen = torch.cat([eps_pred.detach(), nu], dim=1)
            losses['vlb'] = self.vlb_loss(x0, xt, timesteps, nn_out_frozen).mean()

        #if landmarks is not None:
        #    losses['lip'] = self.lip_loss(eps, eps_pred, landmarks)

        losses['simple'] = ((eps - eps_pred) ** 2).mean()
        return losses, outp, eps_pred #nn_out

    def forward_diffusion(self, x0, timesteps):
        timesteps = timesteps.to(torch.long)
        eps = torch.randn_like(x0)
        alpha_bar = self.alpha_bar.to(x0.device)
        sqrt_alpha_bar = torch.sqrt(alpha_bar[timesteps])
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar[timesteps])
        #print('self.broadcast)(sqrt_alpha_bar)', self.broadcast(sqrt_alpha_bar).shape)
        #print('x0', x0.shape)
        
        #print('x0', x0.shape)
        xt = self.broadcast(sqrt_alpha_bar) * x0 + self.broadcast(sqrt_one_minus_alpha_bar) * eps
        print('eps', eps.shape)
        print('xt', xt.shape)
        return eps, xt

    def sample(self, motion_frames, x_cond, audio_emb, n_audio_motion_embs=2, n_motion_frames=2, motion_channels=3):
        with torch.no_grad():
            n_frames = audio_emb.shape[0]

            xT = torch.randn(n_frames, self.in_channels, self.x_shape[0], self.x_shape[1]).to(x_cond.device) #x_cond.shape[0]
            print('thien test xt', xT.shape)
            print('thien test xt', audio_emb.shape)
            #audio_ids = [0] * n_audio_motion_embs
            #for i in range(n_audio_motion_embs + 1):
            #    audio_ids += [i]
            
            #motion_frames = [self.motion_transforms(x_cond) for _ in range(n_motion_frames)]
            #motion_frames = torch.cat(motion_frames, dim=1)

            samples = []
            for i in trange(n_frames, desc=f'Sampling'):
                print(xT[i].shape)
                sample_frame = self.sample_loop(xT[i].unsqueeze(0).to(x_cond.device), x_cond[i].unsqueeze(0), motion_frames[i].unsqueeze(0), audio_emb[i].unsqueeze(0))
                samples.append(sample_frame) #.unsqueeze(1))
                #motion_frames = torch.cat([motion_frames[:, motion_channels:, :], self.motion_transforms(sample_frame)], dim=1)
                #audio_ids = audio_emb[i] # audio_ids[1:] + [min(i + n_audio_motion_embs + 1, n_frames - 1)]
                #print(audio_ids)
                #if i<=2:
                #    print(audio_ids)
            sp=torch.cat(samples, dim=0) #1
            print(sp.shape)
        return sp #torch.cat(samples, dim=1)

    def sample_loop(self, xT, x_cond, motion_frames, audio_emb):
        xt = xT
        #print('sample', audio_emb.shape)
        for i, t in reversed(list(enumerate(self.timesteps))):
            timesteps = torch.tensor([t] * xT.shape[0]).to(xT.device)
            timesteps_ids = torch.tensor([i] * xT.shape[0]).to(xT.device)
            nn_out = self.nn_backbone(xt, timesteps, x_cond, motion_frames=motion_frames, audio_emb=audio_emb)
            mean, logvar = self.get_p_params(xt, timesteps_ids, nn_out)
            noise = torch.randn_like(xt) if t > 0 else torch.zeros_like(xt)
            xt = mean + noise * torch.exp(logvar / 2)
        print('xt', xt.shape)
        return xt

    def get_p_params(self, xt, timesteps, nn_out):
        timesteps = timesteps.to(xt.device)
        if self.in_channels == self.out_channels:
            eps_pred = nn_out
            p_logvar = self.broadcast(torch.log(self.beta.to(xt.device)[timesteps])).to(xt.device)
        else:
            eps_pred, nu = nn_out.chunk(2, 1)
            nu = (nu + 1) / 2
            p_logvar = (nu * self.broadcast(torch.log(self.beta.to(xt.device)[timesteps])) + 
                        (1 - nu) * self.broadcast(self.log_beta_tilde_clipped.to(xt.device)[timesteps])).to(xt.device)

        p_mean, _ = self.get_q_params(xt, timesteps, eps_pred=eps_pred)
        return p_mean, p_logvar

    def get_q_params(self, xt, timesteps,eps_pred=None, x0=None):
        #print('thien', x0)
        if x0 is None:
            coef1_x0 = self.broadcast(self.coef1_x0.to(xt.device)[timesteps]).to(xt.device)
            coef2_x0 = self.broadcast(self.coef2_x0.to(xt.device)[timesteps]).to(xt.device)
            print('coef1_x0', coef1_x0.shape)
            print('xt', xt.shape)
            print('eps_pred', eps_pred.shape)
            #xt, _ = xt.chunk(2, 1)
            x0 = coef1_x0 * xt - coef2_x0 * eps_pred
            x0 = x0.clamp(-1, 1)
        print('x0', x0.shape)
        print('xt', xt.shape)
        #print(' coef1_q',  coef1_q.shape)
        coef1_q = self.broadcast(self.coef1_q.to(xt.device)[timesteps]).to(xt.device)
        coef2_q = self.broadcast(self.coef2_q.to(xt.device)[timesteps]).to(xt.device)
        q_mean = coef1_q * x0 + coef2_q * xt

        q_logvar = self.broadcast(self.log_beta_tilde_clipped.to(xt.device)[timesteps]).to(xt.device)

        return q_mean, q_logvar

    def vlb_loss(self, x0, xt, timesteps, nn_out):
        #print('thien', x0)
        p_mean, p_logvar = self.get_p_params(xt, timesteps, nn_out)
        q_mean, q_logvar = self.get_q_params(xt, timesteps, x0=x0)

        kl = gaussian_kl(q_mean, q_logvar, p_mean, p_logvar)
        kl = kl.mean([1, 2, 3]) / np.log(2.0)

        decoder_nll = discretized_gaussian_nll(x0, means=p_mean, log_scales=0.5 * p_logvar)
        decoder_nll = decoder_nll.mean([1, 2, 3]) / np.log(2.0)

        loss = torch.where((timesteps == 0), decoder_nll, kl)
        return loss

    def lip_loss(self, eps, eps_pred, landmarks):
        loss = 0.
        for _eps, _eps_pred, _landmarks in zip(eps, eps_pred, landmarks):
            min_col, min_row = _landmarks.min(0)[0].floor().long()
            max_col, max_row = _landmarks.max(0)[0].ceil().long()
            eps_cropped = _eps[:, max(0, min_row - 5) : min(_eps.shape[1], max_row + 5), max(0, min_col - 5) : min(_eps.shape[1], max_col + 5)]
            eps_pred_cropped = _eps_pred[:, max(0, min_row - 5) : min(_eps.shape[1], max_row + 5), max(0, min_col - 5) : min(_eps.shape[1], max_col + 5)]
            loss += ((eps_cropped - eps_pred_cropped) ** 2).mean()
        return loss / eps.shape[0]

    def get_beta_schedule(self, max_beta=0.999):
        alpha_bar = lambda t: np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2
        betas = []
        for i in range(self.n_timesteps):
            t1 = i / self.n_timesteps
            t2 = (i + 1) / self.n_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return torch.tensor(betas)

    def set_params(self):
        self.alpha = 1 - self.beta#.to('cuda')
        print(type(self.beta))
        #self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.alpha_bar = torch.cumprod(self.alpha.to(torch.float32), dim=0).type(self.dtype)

        self.alpha_bar_prev = torch.cat([torch.ones(1,), self.alpha_bar[:-1]])

        self.beta_tilde = self.beta * (1.0 - self.alpha_bar_prev) / (1.0 - self.alpha_bar)
        self.log_beta_tilde_clipped = torch.log(torch.cat([self.beta_tilde[1, None], self.beta_tilde[1:]]))

        # to calculate x0 from eps_pred
        self.coef1_x0 = torch.sqrt(1.0 / self.alpha_bar)
        self.coef2_x0 = torch.sqrt(1.0 / self.alpha_bar - 1)

        # for q(x_{t-1} | x_t, x_0)
        self.coef1_q = self.beta * torch.sqrt(self.alpha_bar_prev) / (1.0 - self.alpha_bar)
        self.coef2_q = (1.0 - self.alpha_bar_prev) * torch.sqrt(self.alpha) / (1.0 - self.alpha_bar)

    def space(self, n_timesteps_new):
        self.timesteps = torch.tensor(self.space_timesteps(self.n_timesteps, n_timesteps_new), dtype=torch.long)
        self.n_timesteps = n_timesteps_new

        self.beta = self.get_spaced_beta()
        self.set_params()

    def space_timesteps(self, n_timesteps, target_timesteps):
        all_steps = []
        frac_stride = (n_timesteps - 1) / (target_timesteps - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(target_timesteps):
            taken_steps.append(round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        return all_steps

    def get_spaced_beta(self):
        last_alpha_cumprod = 1.0
        new_beta = []
        for i, alpha_cumprod in enumerate(self.alpha_bar):
            if i in self.timesteps:
                new_beta.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
        return torch.tensor(new_beta, dtype=self.dtype)

    def broadcast(self, arr, dim=4):
        while arr.dim() < dim:
            arr = arr[:, None]
        return arr.to(self.device)

    @property
    def device(self):
        return next(self.nn_backbone.parameters()).device

if __name__ == '__main__':
    from unet import UNet

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
        dropout=dropout, channel_mult=channel_mult, num_heads=num_heads, num_head_channels=num_head_channels, resblock_updown=resblock_updown).to('cuda')
    diffusion = Diffusion(unet, 10, in_channels, out_channels, 32) 

    x = torch.randn(5, 3, image_size, image_size).float().to('cuda')
    x_motion = torch.randn(5, 3, 64, 64).to('cuda')
    a = torch.randn(5, 512, 1, 1).float().to('cuda')
    e = torch.randint(0, 6, (5,) ).to('cuda')
    e = torch.nn.functional.one_hot(e, num_classes=6).float()
    out=diffusion(x, e, audio_emb=a, motion_frames=x_motion)
    print(out)
    print(diffusion.sample(5).shape)
