import io
import os, sys
import requests
import PIL

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from dall_e          import map_pixels, unmap_pixels, load_model
from IPython.display import display, display_markdown
from decord import VideoReader
from decord import cpu
import glob
import numpy as np
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm
import os

target_image_size = 128

def download_image(url):
    resp = requests.get(url)
    resp.raise_for_status()
    return PIL.Image.open(io.BytesIO(resp.content))

def preprocess(img):
    s = min(img.size)

    if s < target_image_size:
        raise ValueError(f'min dim for image {s} < {target_image_size}')

    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0)
    return map_pixels(img)

  # This can be changed to a GPU, e.g. 'cuda:0'.
dev = torch.device('cuda:0')
enc_file = "./checkpoints/encoder.pkl"
dec_file = "./checkpoints/decoder.pkl"
with open(enc_file, 'rb') as f:
    enc =  torch.load(f, map_location=dev)
with open(dec_file, 'rb') as f:
    dec =  torch.load(f, map_location=dev)


vid_dir = '/scratch/kshitijd/Algonauts2020/AlgonautsVideos268_All_30fpsmax'
video_list = glob.glob(vid_dir+'/*.mp4')
video_list.sort()
save_dir = "/scratch/kshitijd/Algonauts2020/activations/dalle_latent"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
for v, video_file in tqdm(enumerate(video_list)):
    vr = VideoReader(video_file, ctx=cpu(0))
    #print(len(vr))
    num_frames = len(vr)
    indices = np.linspace(0, num_frames - 1, 16,dtype=int)
    #print(indices)
    images=[]
    for seg_ind in indices:
        images.append(Image.fromarray(vr[seg_ind-1].asnumpy()))
    latent = []
    for image in images:
        x = preprocess(image)
        z_logits = enc(x.cuda())
        #print(z.cpu().numpy().shape)
        latent.append(z_logits.cpu().numpy())
    latent = np.array(latent)

    save_path = os.path.join(save_dir, str(v).zfill(4)+"_"+"logits.npy")
    np.save(save_path,latent)
