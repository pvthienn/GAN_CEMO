from gc import freeze
from os.path import dirname, join, basename, isfile, isdir, splitext
from tqdm.auto import tqdm
import torch.nn.functional as F
from models import SyncNet
from models import emo_disc
import audio
from PIL import Image
import torch
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np
from glob import glob
import os, random, cv2, argparse
import albumentations as A
import util
from params import hparams, get_image_list
import os
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.transforms
from models import unet as  UNET
from models.unet import  Discriminator
import argparse
import json
from tqdm import tqdm
import random as rn
import shutil
import logging
parser = argparse.ArgumentParser(description='Code to train the Wav2Lip model without the visual quality discriminator')

parser.add_argument("--data_root", default='preprocessed_datase',  help="Root folder of the preprocessed LRS2 dataset", required=None, type=str)

parser.add_argument('--checkpoint_dir', default='ckpt_againmnew', help='Save checkpoints to this directory', required=None, type=str)
parser.add_argument('--syncnet_checkpoint_path',default='checkpoint_step000042922.pth', help='Load the pre-trained Expert discriminator', required=None, type=str)
parser.add_argument('--emotion_disc_path', default='checkpoint/disc_emo_235.pth',help='Load the pre-trained emotion discriminator', required=None, type=str) 
parser.add_argument('--netg_disc_path', help='Load the pre-trained emotion discriminator', required=None, type=str) 
parser.add_argument('--num_epochs', type=int, default=1500)
parser.add_argument('--lr_emo', type=float, default=1e-04)
parser.add_argument("--gpu-no", type=str, help="select gpu", default='0')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--audio_emb_dir', type=str, default='/path/to/folder/with/audio_embs', help="Path to the folder with audio embeddings")
parser.add_argument('--img_resize', type=int, default=128, help="Image resize dimension")
parser.add_argument('--n_motion_frames', type=int, default=5, help="Number of additional conditioning frames to preserve motion") 
parser.add_argument('--n_audio_motion_embs', type=int, default=0, help="Number of additional audio embeddings before and after the current frame")
parser.add_argument('--grayscale_motion', type=bool, default=False, help="Use grayscale motion")
parser.add_argument('--motion_blur', type=bool, default=False, help="Use motion blur")
parser.add_argument('--img_size', type=int, default=128, help="Image size")
parser.add_argument('--in_channels', type=int, default=3, help="Number of input channels")
parser.add_argument('--model_channels', type=int, default=64, help="Number of model channels")
parser.add_argument('--out_channels', type=int, default=3, help="Number of output channels")
parser.add_argument('--num_res_blocks', type=int, default=4, help="Number of residual blocks") 
parser.add_argument('--attention_resolutions', type=int, nargs='+', default=[16], help="Where to add attention layers") #[16, 8, 4]
parser.add_argument('--dropout', type=float, default=0, help="Dropout rate")
parser.add_argument('--channel_mult', type=int, nargs='+', default=[1, 1, 2, 3], help="Channel multiplier") #[1, 2, 3]
parser.add_argument('--num_heads', type=int, default=4, help="Number of heads for attention") 
parser.add_argument('--num_head_channels', type=int, default=-1, help="Number of head channels for attention") 
parser.add_argument('--resblock_updown', type=bool, default=True, help="Use resblock updown")
parser.add_argument('--id_condition_type', type=str, choices=['frame', 'post_gn', 'pre_gn'], default='post_gn', help="ID condition type")
parser.add_argument('--audio_condition_type', type=str, choices=['pre_gn', 'post_gn', 'double_pre_gn', 'attention'], default='attention', help="Audio condition type")
parser.add_argument('--checkpoint', type=str, default=None, help="Path to the checkpoint for a new training")
parser.add_argument('--precision', type=int, default=32, help="Precision")
parser.add_argument('--n_workers', type=int, default=0, help="Number of workers")
parser.add_argument('--log_dir', type=str, default='/path/to/log/folder', help="Path to the log folder")
parser.add_argument('--debug', type=bool, default=False, help="Enable debug mode")
args = parser.parse_args()


global_step = 0
global_epoch = 0
os.environ['CUDA_VISIBLE_DEVICES']='0'
use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))

syncnet_T = 5 #5
syncnet_mel_step_size = 16
def warmup(start, end, max_steps, current_step):
    if current_step > max_steps:
        return end
    return (end - start) * (current_step / max_steps) + start
def to_categorical(y, num_classes=None, dtype='float32'):

    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y)
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

emotion_dict = {'ANG':0, 'DIS':1, 'FEA':2, 'HAP':3, 'NEU':4, 'SAD':5}
intensity_dict = {'XX':0, 'LO':1, 'MD':2, 'HI':3}
emonet_T = 5

class Dataset(object):
    def __init__(self, split, val=False):
        #self.all_videos = get_image_list(args.data_root, split)
        # self.all_videos = [join(args.data_root, f) for f in os.listdir(args.data_root) if isdir(join(args.data_root, f))]
        self.filelist = []
        self.all_videos = [f for f in os.listdir(args.data_root) if isdir(join(args.data_root, f))]
        #self.identity_dict = {}
        for filename in self.all_videos:
            #print(splitext(filename))
            labels = splitext(filename)[0].split('_')
            #print(labels)
            emotion = emotion_dict[labels[2]]

            emotion_intensity = intensity_dict[labels[3]]
            if val:
                if emotion_intensity != 3:
                    continue
            self.filelist.append((filename, emotion, emotion_intensity))

        self.filelist = np.array(self.filelist)
        target = {}
        for i in range(1, 2*emonet_T):
            target['image' + str(i)] = 'image'

        self.augments = A.Compose([
                        A.RandomBrightnessContrast(p=0.4),
                        A.RandomGamma(p=0.4),
                        A.CLAHE(p=0.4),
                        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=50, val_shift_limit=50, p=0.4),
                        A.ChannelShuffle(p=0.4),
                        A.RGBShift(p=0.4),
                        A.RandomBrightness(p=0.4),
                        A.RandomContrast(p=0.4),
                        A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
                    ], additional_targets=target, p=0.8)
    def augmentVideo(self, video):
        args = {}
        args['image'] = video[0, :, :, :]
        for i in range(1, 2*emonet_T):
            args['image' + str(i)] = video[i, :, :, :]
        result = self.augments(**args)
        video[0, :, :, :] = result['image']
        for i in range(1, 2*emonet_T):
            video[i, :, :, :] = result['image' + str(i)]
        return video

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def read_window(self, window_fnames):
        if window_fnames is None: return None
        window = []
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                return None
            try:
                img = cv2.resize(img, (hparams.img_size, hparams.img_size))
            except Exception as e:
                return None

            window.append(img)

        return window

    def crop_audio_window(self, spec, start_frame):
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame) # 0-indexing ---> 1-indexing
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))

        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx : end_idx, :]

    def get_segmented_mels(self, spec, start_frame):
        mels = []
        assert syncnet_T == 5 #5
        start_frame_num = self.get_frame_id(start_frame) + 1 # 0-indexing ---> 1-indexing
        if start_frame_num - 2 < 0: return None
        for i in range(start_frame_num, start_frame_num + syncnet_T):
            m = self.crop_audio_window(spec, i - 2)
            if m.shape[0] != syncnet_mel_step_size:
                return None
            mels.append(m.T)

        mels = np.asarray(mels)

        return mels

    def prepare_window(self, window):
        x = np.asarray(window) / 255.
        x = np.transpose(x, (3, 0, 1, 2))

        return x

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.filelist) - 1)
            filename = self.filelist[idx]
            vidname = filename[0]
            emotion = int(filename[1])
            emotion = to_categorical(emotion, num_classes=6)
            img_names = list(glob(join(args.data_root, vidname, '*.jpg')))
            if len(img_names) <= 3 * syncnet_T:
                continue
            img_name = random.choice(img_names)
            wrong_img_name = random.choice(img_names)
            while wrong_img_name == img_name:
                wrong_img_name = random.choice(img_names)

            window_fnames = self.get_window(img_name)
            wrong_window_fnames = self.get_window(wrong_img_name)
            if window_fnames is None or wrong_window_fnames is None:
                continue

            window = self.read_window(window_fnames)
            if window is None:
                continue
            wrong_window = self.read_window(wrong_window_fnames)
            if wrong_window is None:
                continue
            try:
                wavpath = join(args.data_root, vidname, "audio.wav")
                wav = audio.load_wav(wavpath, hparams.sample_rate)

                orig_mel = audio.melspectrogram(wav).T
            except Exception as e:
                continue
            mel = self.crop_audio_window(orig_mel.copy(), img_name)
            if (mel.shape[0] != syncnet_mel_step_size):
                continue
            indiv_mels = self.get_segmented_mels(orig_mel.copy(), img_name)
            if indiv_mels is None: continue
            window = np.asarray(window)
            y = window.copy()
            window[:, :, :] = 0.
            wrong_window = np.asarray(wrong_window)
            cond_window=np.tile(np.expand_dims(wrong_window[0], axis=0), (5, 1, 1, 1)) 
            conact_for_aug = np.concatenate([y, wrong_window, cond_window], axis=0)
            aug_results = self.augmentVideo(conact_for_aug)
            y, wrong_window, cond_window= np.split(aug_results, 3, axis=0)
            y = (np.transpose(y, (3, 0, 1, 2)) / 255)*2-1
            window = np.transpose(window, (3, 0, 1, 2)) 
            wrong_window = (np.transpose(wrong_window, (3, 0, 1, 2)) / 255)*2-1
            cond_window = (np.transpose(cond_window, (3, 0, 1, 2)) / 255) *2-1
            x =  wrong_window 
            x = torch.FloatTensor(x)
            cond = torch.FloatTensor(cond_window)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)
            indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1)
            y = torch.FloatTensor(y)
            return x, indiv_mels, mel, y, emotion, cond

logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss

def freezeNet(network):
    for p in network.parameters():
        p.requires_grad = False

def unfreezeNet(network):
    for p in network.parameters():
        p.requires_grad = True

device = torch.device("cuda" if use_cuda else "cpu")
device_ids = list(range(torch.cuda.device_count()))
def get_sync_loss(mel, g):
    g = g[:, :, :, g.size(3)//2:]
    g = torch.cat([g[:, :, i] for i in range(syncnet_T)], dim=1)
    a, v = syncnet(mel, g)
    y = torch.ones(g.size(0), 1).float().to(device)
    return cosine_loss(a, v, y)

def init_model(model, device):
    model = model.to(device)
    model = init_weights(model)
    return model
def set_requires_grad(model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad
syncnet = SyncNet().to(device)
freezeNet(syncnet)

disc_emo = emo_disc.DISCEMO().to(device)
disc_emo.load_state_dict(torch.load(args.emotion_disc_path)) #['state_dict'])
emo_loss_disc = nn.CrossEntropyLoss()
perceptual_loss = util.perceptionLoss(device)
L2_loss = nn.L1Loss() 
net_D = init_model(Discriminator(), device)
GANcriterion = GANLoss(gan_mode='vanilla').to(device)
opt_G = optim.Adam(net_D.parameters(), lr=1e-04, betas=(0.5,0.999))
def train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None):
    print(f'num_batches:{len(train_data_loader)}')
    global global_step, global_epoch
    resumed_step = global_step
    while global_epoch < nepochs:
        print('Starting Epoch: {}'.format(global_epoch))
        running_sync_loss, running_total_loss  = 0., 0.
        running_ploss, running_loss_de_c = 0., 0.
        running_loss_fake_c, running_loss_real_c = 0., 0.
        l2_losses=0.
        loss_simple=0.
        loss_p=0.
        dec_loss_coef = warmup(0, 1., 500000, global_step)
        prog_bar = tqdm(enumerate(train_data_loader))
        for step, (x, indiv_mels, mel, gt, emotion, cond) in prog_bar:

            model.train()
            disc_emo.train()
            net_D.train()
       
            freezeNet(disc_emo)
            x = x.to(device)
            cond = cond.to(device)
            mel = mel.to(device)
            indiv_mels = indiv_mels.to(device)
            gt = gt.to(device)
            emotion = emotion.to(device)
            g = model(x, emotion, indiv_mels)

            sp = torch.cat([gt[:, :, i] for i in range(gt.size(2))], dim=0)
            gless=torch.cat([g[:, :, i] for i in range(g.size(2))], dim=0)
            fake_enpreds, fake_depreds = net_D(gless.detach())
            real_enpreds, real_depreds = net_D(sp)
            enc_divergence = (F.relu(1 + real_enpreds) + F.relu(1 - fake_enpreds)).mean()
            dec_divergence = (F.relu(1 + real_depreds) + F.relu(1 - fake_depreds)).mean()
            loss_D = enc_divergence + dec_divergence * dec_loss_coef
            opt_G.zero_grad()
            loss_D.requires_grad_()
            loss_D.backward()
            opt_G.step()
            optimizer.zero_grad()
            fake_enpreds, fake_depreds = net_D(gless)
            loss_G_GAN = fake_enpreds.mean() + F.relu(1 + fake_depreds).mean()
            emotion_ = emotion.unsqueeze(1).repeat(1, 5, 1)
            emotion_ = torch.cat([emotion_[:, i] for i in range(emotion_.size(1))], dim=0)

            de_c= disc_emo.forward((g+1)/2)

            loss_de_c = emo_loss_disc(de_c, torch.argmax(emotion, dim=1))

            if hparams.syncnet_wt > 0.:
                sync_loss = get_sync_loss(mel, (g+1)/2)
            else:
                sync_loss = 0.

            ploss =  perceptual_loss.calculatePerceptionLoss((g+1)/2,(gt+1)/2)
            L2loss = L2_loss(g, gt)
            loss = 0.03* ploss + params.syncnet_wt * sync_loss + 0.01 * loss_G_GAN + 0.01 * loss_de_c + (1-0.03-0.01*2-hparams.syncnet_wt)* L2loss
            loss_p+=ploss #los['vlb']
            l2_losses+=L2loss
            loss.backward()
            optimizer.step()
            unfreezeNet(disc_emo)
            if params.syncnet_wt > 0.:
                running_sync_loss += sync_loss.item()
            else:
                running_sync_loss += 0.
            running_total_loss += loss.item()
            running_loss_de_c += loss_de_c.item()
            disc_emo.opt.zero_grad()
            g = g.detach()
            class_real = disc_emo((gt+1)/2) # for ground-truth

            loss_real_c = emo_loss_disc(class_real, torch.argmax(emotion, dim=1))
            loss_real_c.backward()
            disc_emo.opt.step()

            running_loss_real_c += loss_real_c.item()
            global_step += 1
            cur_session_steps = global_step - resumed_step

            if global_step == 1 or global_step % checkpoint_interval == 0:
                save_checkpoint(model, optimizer, global_step, checkpoint_dir, global_epoch)
                save_checkpoint(disc_emo, disc_emo.opt, global_step,checkpoint_dir, global_epoch, prefix='disc_emo_')
                save_checkpoint(net_D, opt_G, global_step,checkpoint_dir, global_epoch, prefix='net_D_')

            if global_step == 1 or global_step % hparams.eval_interval == 0:
                with torch.no_grad():
                    average_sync_loss = eval_model(test_data_loader, global_step, device, model, checkpoint_dir)

                    if average_sync_loss < 1.0:
                        params.set_hparam('syncnet_wt', 0.1) #15) #0.03) # without image GAN a lesser weight is sufficient

            prog_bar.set_description('Loss_total: {:.4f}, L2Loss: {:.4f}, pLoss: {:.4f}, Sync Loss: {:.4f},  dec Loss: {:.4f},  realdec Loss: {:.4f},'.format(running_total_loss / (step + 1), l2_losses / (step + 1), loss_p / (step + 1),
                                                                    running_sync_loss / (step + 1),running_loss_de_c / (step + 1),running_loss_real_c / (step + 1),
                                                                    ))
        global_epoch += 1

def eval_model(test_data_loader, global_step, device, model, checkpoint_dir):
    eval_steps = 50
    print('\nEvaluating for {} steps'.format(eval_steps))
    sync_losses, recon_losses, losses_de_c, p_losses = [], [], [], []
    losses_real_c = []
    pclos=[]
    l2losses=[]
    losses_simple=[]
    step = 0
    while 1:
        for x, indiv_mels, mel, gt, emotion, cond in test_data_loader:
            step += 1
            model.eval()
            disc_emo.eval()
            x = x.to(device)
            cond = cond.to(device)

            gt = gt.to(device)
            indiv_mels = indiv_mels.to(device)
            mel = mel.to(device)
            emotion = emotion.to(device)

            g= model(x, emotion, indiv_mels)
            l2loss = 0 #L2_loss(g, gt)
            sp = torch.cat([gt[:, :, i] for i in range(gt.size(2))], dim=0)
            plos=perceptual_loss.calculatePerceptionLoss((g+1)/2,(gt+1)/2)
            sync_loss = get_sync_loss(mel, (g+1)/2)
            de_c = disc_emo.forward((g+1)/2)
            emotion_ = emotion.unsqueeze(1).repeat(1, 5, 1)
            emotion_ = torch.cat([emotion_[:, i] for i in range(emotion_.size(1))], dim=0)
            loss_de_c = emo_loss_disc(de_c, torch.argmax(emotion, dim=1))
            class_real = disc_emo((gt+1)/2) # for ground-truth
            loss_real_c = emo_loss_disc(class_real, torch.argmax(emotion, dim=1))
            losss=0.03 * sync_loss + 0.01 * plos+ (1-0.01-0.03) * l2loss # ++ 0.001 * plos+ 0.001 * loss_de_c #+ 0.0001 * los['vlb']
            pclos.append(plos)
            l2losses.append(l2loss)
            sync_losses.append(sync_loss.item())
            p_losses.append(losss.item())

            if step > eval_steps:
                averaged_sync_loss = sum(sync_losses) / len(sync_losses)
                averaged_vlb = sum(pclos) / len(pclos)
                averaged_l2 = sum(l2losses) / len(l2losses)
                averaged_ploss =  sum(p_losses) / len(p_losses)
                print('total_loss: {:.4f}, l2loss: {:4f}, ploss: {:.4f}, Sync Loss: {:.4f}'.format(
                    averaged_ploss, averaged_l2, averaged_vlb, averaged_sync_loss))
                return averaged_sync_loss

def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch, prefix=''):
    checkpoint_path = join(
        checkpoint_dir, "{}checkpoint_step{:09d}.pth".format(prefix, global_step))
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)
def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_checkpoint(path, model, optimizer, reset_optimizer=False, overwrite_global_states=True):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    if overwrite_global_states:
        global_step = checkpoint["global_step"]
        global_epoch = checkpoint["global_epoch"]

    return model

if __name__ == "__main__":
    checkpoint_dir = args.checkpoint_dir

    full_dataset = Dataset('train')
    train_size = int(0.9 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=hparams.batch_size, shuffle=True,
        num_workers=hparams.num_workers, drop_last=True)

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=hparams.batch_size,
        num_workers=4, drop_last=True )

    device = torch.device("cuda" if use_cuda else "cpu")
    image_size, in_channels = args.img_size, args.in_channels#data_module.dataset_train[0][0].shape[1], data_module.dataset_train[0][0].shape[0]


    unetgen = UNET.GENUNet(
            image_size, in_channels, args.model_channels, args.out_channels,
            args.num_res_blocks, args.attention_resolutions, args.dropout,
            args.channel_mult, args.num_heads, args.num_head_channels,
            args.resblock_updown, audio_condition_type=args.audio_condition_type,
            precision=args.precision, n_motion_frames=args.n_motion_frames,
            grayscale_motion=args.grayscale_motion, n_audio_motion_embs=args.n_audio_motion_embs
            )


    model=unetgen.to(device=device)
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=hparams.initial_learning_rate, betas=(0.5,0.999))

    if args.checkpoint_path is not None:
        load_checkpoint(args.checkpoint_path, model, optimizer, reset_optimizer=False)


    load_checkpoint(args.syncnet_checkpoint_path, syncnet, None, reset_optimizer=True, overwrite_global_states=False)


    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    train(device, model, train_data_loader, test_data_loader, optimizer,
              checkpoint_dir=checkpoint_dir,
              checkpoint_interval=hparams.checkpoint_interval,
              nepochs=hparams.nepochs)
