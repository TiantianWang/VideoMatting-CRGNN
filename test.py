import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
import os
import argparse

import cv2
import glob
from PIL import Image
from skimage.measure import label

import sys
sys.path.append('models/')
from hlmobilenetv2 import hlmobilenetv2



import pdb


def image_alignment(img, trimap, output_stride):
    
    imsize = np.asarray(img.shape[:2], dtype=np.float)
    new_imsize = np.ceil(imsize / output_stride) * output_stride
    
    h, w = int(new_imsize[0]), int(new_imsize[1])

    img_resized = cv2.resize(img, dsize=(w,h), interpolation=cv2.INTER_LINEAR)
    trimap_resized = cv2.resize(trimap, dsize=(w,h), interpolation=cv2.INTER_NEAREST)

    if len(trimap_resized.shape) < 3:
        trimap_resized=np.expand_dims(trimap_resized, axis=2)
    img_trimap = np.concatenate((img_resized, trimap_resized), axis=2)

    return img_trimap

def image_read(name, img_path, trimap_path):

    img_file = os.path.join(img_path, name)
    trimap_file = os.path.join(trimap_path, name[:-3]+'png')

    trimap = np.array(Image.open(trimap_file)).astype(np.float32)
    img = np.array(Image.open(img_file)).astype(np.float32)
    image = image_alignment(img, trimap, args.output_stride)

    img = image[:,:,0:3]
    trimap = image[:,:,3]

    # if (len(trimap.shape) > 2):
    #     trimap = trimap[:, :, 0]
    
    img = torch.from_numpy(img)
    trimap = torch.from_numpy(trimap)

    
    img = img.permute(2, 0, 1).unsqueeze(dim=0).cuda()
    trimap = trimap.view(1, 1, trimap.shape[0], trimap.shape[1]).cuda()
    
    # if img.shape[1] > 3:
    #     img = img[:, 0:3]

    return img, trimap

def image_save(out_path, out_path_fg, name, trimap, pred, pred_fg):

    trimap = trimap.squeeze(dim=2)

    pred[trimap == 255] = 1
    pred[trimap == 0] = 0

    out_name = os.path.join(out_path, name[:-3] + 'png')
    out_name_fg = os.path.join(out_path_fg, name[:-3] + 'png')
    pred_out = pred.squeeze().detach().cpu().numpy()
    
    #refine alpha with connected component
    labels=label((pred_out>0.05).astype(int))
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    pred_out=pred_out*largestCC
    cv2.imwrite(out_name, np.uint8(pred_out * 255))

    pred_fg = pred_fg.squeeze().detach().cpu().numpy()
    pred_out_fg = np.expand_dims(pred_out, axis=0) * pred_fg
    pred_out_fg = pred_out_fg.transpose(1, 2, 0)* 255
    pred_out_fg = pred_out_fg.clip(0, 255).round()
    pred_out_fg = cv2.cvtColor(np.uint8(pred_out_fg), cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_name_fg, pred_out_fg)

def get_arguments():
    parser = argparse.ArgumentParser(description="Transformer Network")

    parser.add_argument('--output_stride', type=int, default=8, help="output stirde of the network")
    parser.add_argument('--crop_size', type=int, default=320, help="crop size of input image")
    parser.add_argument('--conv_operator', type=str, default='std_conv', help=" ")
    parser.add_argument('--decoder', type=str, default='indexnet', help=" ")
    parser.add_argument('--decoder_kernel_size', type=int, default=5, help=" ")
    parser.add_argument('--indexnet', type=str, default='depthwise', help=" ")
    parser.add_argument('--index_mode', type=str, default='m2o', help=" ")
    parser.add_argument('--use_nonlinear', type=str, default=True, help=" ")
    parser.add_argument('--use_context', type=str, default=True, help=" ")
    parser.add_argument('--apply_aspp', type=str, default=True, help=" ")
    parser.add_argument('--sync_bn', type=str, default=False, help=" ")

    return parser.parse_args()

args = get_arguments()


dataroot = './examples'
img_dir = os.path.join(dataroot, 'image')
trimap_dir = os.path.join(dataroot, 'trimap')

output_dir = './result-real/alpha' # output image path
output_dir_fg = './result-real/fg' # output fg path

ck_dir = './checkpoint'
ck_name = sorted(glob.glob(os.path.join(ck_dir, '*.pth')))

model = hlmobilenetv2(
    pretrained=True,
    freeze_bn=True, 
    output_stride=args.output_stride, 
    input_size=args.crop_size, 
    apply_aspp=args.apply_aspp,
    conv_operator=args.conv_operator,
    decoder=args.decoder,
    decoder_kernel_size=args.decoder_kernel_size,
    indexnet=args.indexnet,
    index_mode=args.index_mode,
    use_nonlinear=args.use_nonlinear,
    use_context=args.use_context,
    sync_bn=args.sync_bn
)

model = nn.DataParallel(model)

if torch.cuda.is_available():
    model.cuda()
model.eval()

for ck_ind, ckname in enumerate(ck_name):

    ckpt = torch.load(ckname)   # load checkpoint
    model.load_state_dict(ckpt, strict=True)
    ck_split = ckname.split('/')[-1][:-4]
    img_name = sorted(os.listdir(trimap_dir))
    for i, imgname in enumerate(img_name):
        print(i)
        img_path = os.path.join(img_dir, imgname)
        trimap_path = os.path.join(trimap_dir, imgname)
        
        out_path = os.path.join(output_dir, ck_split, imgname)
        out_path_fg = os.path.join(output_dir_fg, ck_split, imgname)

        if not os.path.exists(out_path):
            os.makedirs(out_path)
        if not os.path.exists(out_path_fg):
            os.makedirs(out_path_fg)

        img_name = sorted(os.listdir(img_path))
        
        pred_p = {}
        pred_fg_p = {}
        for j in range(1, len(img_name)-1):
            print(j)
            name_c = img_name[j]

            img_file_c = os.path.join(img_path, name_c)
            trimap_file_c = os.path.join(trimap_path, name_c[:-3]+'png')

            img_c = np.array(Image.open(img_file_c)).astype(np.float32)
            trimap_c = np.array(Image.open(trimap_file_c)).astype(np.float32)
            image_trimap = image_alignment(img_c, trimap_c, args.output_stride)

            img_c = image_trimap[:,:,0:3]
            trimap_c = image_trimap[:,:,3:]

            img_c = torch.from_numpy(img_c).cuda()
            img_c = img_c.permute(2, 0, 1).unsqueeze(dim=0)

            trimap_c = torch.from_numpy(trimap_c)
            trimap_c = trimap_c.view(1, 1, trimap_c.shape[0], trimap_c.shape[1]).cuda()

            name0 = img_name[j-1]
            name1 = img_name[j+1]
            img0, trimap0 = image_read(name0, img_path, trimap_path)
            img1, trimap1 = image_read(name1, img_path, trimap_path)

            with torch.no_grad():
                pred0, pred_fg0, pred_c, pred_fg_c, pred1, pred_fg1 = \
                    model(img0.clone(), trimap0.clone(), img_c.clone(), trimap_c.clone(), img1.clone(), trimap1.clone())
            trimap_c = trimap_c.squeeze(dim=2)

            if j == 1:
                pred_p[j-1] = pred0
                pred_p[j] = pred_c
                pred_p[j+1] = pred1
                
                pred_fg_p[j-1] = pred_fg0
                pred_fg_p[j] = pred_fg_c
                pred_fg_p[j+1] = pred_fg1

            else:
                pred_p[j-1] += pred0
                pred_p[j] += pred_c
                pred_p[j+1] += pred1

                pred_fg_p[j-1] += pred_fg0
                pred_fg_p[j] += pred_fg_c
                pred_fg_p[j+1] += pred_fg1

            _, _, h_align, w_align = img_c.shape
            pred_p[j+2] = torch.zeros((1, 1, h_align, w_align)).to(pred_c.device)
            pred_fg_p[j+2] = torch.zeros((1, 3, h_align, w_align)).to(pred_c.device)

            if j == 1:
                pred_ = pred_p[j-1]
                pred_ = torch.clamp(pred_, 0, 1)
                pred_fg_ = pred_fg_p[j-1]
            if j == 2:
                pred_ = pred_p[j-1] / 2
                pred_ = torch.clamp(pred_, 0, 1)
                pred_fg_ = pred_fg_p[j-1] / 2
            if j >= 3:
                pred_ = pred_p[j-1] / 3
                pred_ = torch.clamp(pred_, 0, 1)
                pred_fg_ = pred_fg_p[j-1] / 3
            

            image_save(out_path, out_path_fg, name0, trimap0, pred_, pred_fg_)

            # save the last 2 images
            if j == len(img_name) - 2:
                pred_ = pred_p[j] / 2
                pred_ = torch.clamp(pred_, 0, 1)
                pred_fg_ = pred_fg_p[j] / 2
                image_save(out_path, out_path_fg, name_c, trimap_c, pred_, pred_fg_)

                pred_ = pred_p[j+1]
                pred_ = torch.clamp(pred_, 0, 1)
                pred_fg_ = pred_fg_p[j+1]
                image_save(out_path, out_path_fg, name1, trimap1, pred_, pred_fg_)

            del pred_p[j-1]
            del pred_fg_p[j-1]
            del pred_
            del pred_fg_
            del pred0
            del pred_fg0
            del pred_c
            del pred_fg_c
            del pred1
            del pred_fg1


            
