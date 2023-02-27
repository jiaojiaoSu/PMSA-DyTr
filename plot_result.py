# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

import os
import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from datasets.coco import make_coco_transforms
from models import build_model

import matplotlib.pyplot as plt


def get_args_parser():
    parser = argparse.ArgumentParser('SAM-DETR: Accelerating DETR Convergence via Semantic-Aligned Matching', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=[], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float, help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    parser.add_argument('--multiscale', default=True, action='store_true')#add
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',default=True,
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")#add
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine',),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int, help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int, help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int, help="dimension of the FFN in the transformer")
    parser.add_argument('--hidden_dim', default=256, type=int, help="dimension of the transformer")
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int, help="Number of attention heads in the transformer attention")
    parser.add_argument('--num_queries', default=300, type=int, help="Number of query slots")

    parser.add_argument('--smca', default=True, action='store_true')#True

    # * Segmentation
    parser.add_argument('--masks', action='store_true', help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2.0, type=float, help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5.0, type=float, help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2.0, type=float, help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1.0, type=float)
    parser.add_argument('--dice_loss_coef', default=1.0, type=float)
    parser.add_argument('--cls_loss_coef', default=2.0, type=float)
    parser.add_argument('--bbox_loss_coef', default=5.0, type=float)
    parser.add_argument('--giou_loss_coef', default=2.0, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str, default='/media/zndx/存储库/sjj/transformer-series/Deformable-DETR-main/coco_neu')
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing. We must use cuda.')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='output/encoder6-decoder6/checkpoint0045.pth', help='resume from checkpoint, empty for training from scratch')#结果可视化的模型
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval_every_epoch', default=1, type=int, help='eval every ? epoch')
    parser.add_argument('--save_every_epoch', default=1, type=int, help='save model weights every ? epoch')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser


def main(args):

    utils.init_distributed_mode(args)

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only."
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, post_processors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total number of params in model: ', n_parameters)

    def match_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if "backbone.0" not in n and not match_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters()
                       if "backbone.0" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters()
                       if match_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # dataset_train = build_dataset(image_set='train', args=args)
    # dataset_val = build_dataset(image_set='val', args=args)
    #
    # if args.distributed:
    #     sampler_train = DistributedSampler(dataset_train)
    #     sampler_val = DistributedSampler(dataset_val, shuffle=False)
    # else:
    #     sampler_train = torch.utils.data.RandomSampler(dataset_train)
    #     sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    #
    # batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

    # data_loader_train = DataLoader(dataset_train,
    #                                batch_sampler=batch_sampler_train,
    #                                collate_fn=utils.collate_fn,
    #                                num_workers=args.num_workers)
    #
    # data_loader_val = DataLoader(dataset_val,
    #                              args.batch_size,
    #                              sampler=sampler_val,
    #                              drop_last=False,
    #                              collate_fn=utils.collate_fn,
    #                              num_workers=args.num_workers)
    #
    # if args.dataset_file == "coco_panoptic":
    #     # We also evaluate AP during panoptic training, on original coco DS
    #     coco_val = datasets.coco.build("val", args)
    #     base_ds = get_coco_api_from_dataset(coco_val)
    # else:
    #     base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    transforms = make_coco_transforms("val")
    DETECTION_THRESHOLD = 0.5
    inference_dir = "NEU_test1/"
    inference_dir1 = "NEU_out1/"

    image_dirs = os.listdir(inference_dir)
    image_dirs = [filename for filename in image_dirs if filename.endswith(".jpg") and 'det_res' not in filename]
    model.eval()
    with torch.no_grad():
        for image_dir in image_dirs:
            img = Image.open(os.path.join(inference_dir, image_dir)).convert("RGB")
            w, h = img.size
            orig_target_sizes = torch.tensor([[h, w]], device=device)
            img, _ = transforms(img, target=None)
            img = img.to(device)
            img = img.unsqueeze(0)   # adding batch dimension
            outputs = model(img)
            results = post_processors['bbox'](outputs, orig_target_sizes)[0]
            indexes = results['scores'] >= DETECTION_THRESHOLD
            scores = results['scores'][indexes]
            labels = results['labels'][indexes]
            boxes = results['boxes'][indexes]

            #Visualize the detection results
            import cv2
            img_det_result = cv2.imread(os.path.join(inference_dir, image_dir))
            for i in range(scores.shape[0]):
                x1, y1, x2, y2 = round(float(boxes[i, 0])), round(float(boxes[i, 1])), round(float(boxes[i, 2])), round(float(boxes[i, 3]))
                img_det_result = cv2.rectangle(img_det_result, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.imwrite(os.path.join(inference_dir1, "det_res_" + image_dir), img_det_result)

            #--------使用列表来存储结果
            # # use lists to store the outputs via up-values
            enc_attn_weights0, enc_attn_weights1,enc_attn_weights2= [],[],[]#卷积特征，
            enc_attn_weights3, enc_attn_weights4, enc_attn_weights5=[],[],[]

            hooks = [
                model.transformer.encoder.layers[0].self_attn.register_forward_hook(
                    lambda self, input, output: enc_attn_weights0.append(output.cpu() ) # 最后encoder
                ),
                model.transformer.encoder.layers[1].self_attn.register_forward_hook(
                    lambda self, input, output: enc_attn_weights1.append(output.cpu())  # 最后encoder
                ),
                model.transformer.encoder.layers[2].self_attn.register_forward_hook(
                    lambda self, input, output: enc_attn_weights2.append(output.cpu())  # 最后encoder
                ),
                model.transformer.encoder.layers[3].self_attn.register_forward_hook(
                    lambda self, input, output: enc_attn_weights3.append(output.cpu())#最后encoder
                ),
                model.transformer.encoder.layers[4].self_attn.register_forward_hook(
                    lambda self, input, output: enc_attn_weights4.append(output.cpu())  # 最后encoder
                ),
                model.transformer.encoder.layers[5].self_attn.register_forward_hook(
                    lambda self, input, output: enc_attn_weights5.append(output.cpu())  # 最后encoder
                ),
            ]
            enc_attn_weights_all= [enc_attn_weights0, enc_attn_weights1,enc_attn_weights2,enc_attn_weights3, enc_attn_weights4, enc_attn_weights5]
            #
            # # propagate through the model
            outputs = model(img)#please attention the location
            #
            # for hook in hooks:
            #     hook.remove()

            print(enc_attn_weights0[0].shape)#torch.Size([1, 13294, 256])
            from torchvision import transforms
            unloader=transforms.ToPILImage()
            attn_scale0,attn_scale1,attn_scale2,attn_scale3=[],[],[],[]
            for i in range(6):#6个encoder
                enc_attn_weights= enc_attn_weights_all[i]
                scale0 = torch.mean(enc_attn_weights[0][:, :10000, :], 2).reshape(100, 100).cpu()  # torch.Size([1, 10000, 256])
                scale1 = torch.mean(enc_attn_weights[0][:, 10000:12500, :], 2).reshape(50,50).cpu()  # torch.Size([1, 2500, 256])
                scale2 = torch.mean(enc_attn_weights[0][:, 12500:15000, :], 2).reshape(50,50).cpu()  # torch.Size([1, 625, 256])
                scale3 = torch.mean(enc_attn_weights[0][:, 15000:, :], 2).reshape(25,25).cpu()  # torch.Size([1, 625, 256])

                image0 = unloader(scale0)
                image1 = unloader(scale1)
                image2 = unloader(scale2)
                image3 = unloader(scale3)

                attn_scale0.append(image0)
                attn_scale1.append(image1)
                attn_scale2.append(image2)
                attn_scale3.append(image3)

            attn_scale_all=[ attn_scale0, attn_scale1, attn_scale2,attn_scale3]

            for i in range(4):#4个尺度
                fig=plt.figure()
                attn_scale=attn_scale_all[i]
                #，每个尺度画6个子图
                for plt_index in range(1, 7):  # 第一幅图的下标从1开始，设置6张子图
                    ax = fig.add_subplot(2, 3, plt_index)  # 2行3列
                    ax.set_title("encoder"+str(plt_index))
                    ax.imshow(attn_scale[plt_index-1])
                    pass

                plt.savefig('level'+str(i)+'.png')
                plt.show()

            # #--------使用列表来存储结果交叉注意力进行可视化
            # # use lists to store the outputs via up-values
            # cross_attn_weights0, cross_attn_weights1,cross_attn_weights2= [],[],[]#卷积特征，
            # cross_attn_weights3, cross_attn_weights4, cross_attn_weights5=[],[],[]
            # # # propagate through the model
            #
            #
            # hooks = [
            #     model.transformer.decoder.layers[0].cross_attn.register_forward_hook(
            #         lambda self, input, output: cross_attn_weights0.append(output[0].cpu() ) # 最后encoder
            #     ),
            #     model.transformer.decoder.layers[1].cross_attn.register_forward_hook(
            #         lambda self, input, output: cross_attn_weights1.append(output[0].cpu())  # 最后encoder
            #     ),
            #     model.transformer.decoder.layers[2].cross_attn.register_forward_hook(
            #         lambda self, input, output: cross_attn_weights2.append(output[0].cpu())  # 最后encoder
            #     ),
            #     model.transformer.decoder.layers[3].cross_attn.register_forward_hook(
            #         lambda self, input, output: cross_attn_weights3.append(output[0].cpu())#最后encoder
            #     ),
            #     model.transformer.decoder.layers[4].cross_attn.register_forward_hook(
            #         lambda self, input, output: cross_attn_weights4.append(output[0].cpu())  # 最后encoder
            #     ),
            #     model.transformer.decoder.layers[5].cross_attn.register_forward_hook(
            #         lambda self, input, output: cross_attn_weights5.append(output[0].cpu())  # 最后encoder
            #     ),
            # ]
            #
            #
            # outputs = model(img)  # please attention the location
            # # for hook in hooks:
            # #     hook.remove()
            # print("cross_attn_weights0:",cross_attn_weights0[0].shape)
            # cross_attn_weights_all= [cross_attn_weights0[0], cross_attn_weights1[0],cross_attn_weights2[0],cross_attn_weights3[0], cross_attn_weights4[0], cross_attn_weights5[0]]
            #
            # from torchvision import transforms
            # unloader=transforms.ToPILImage()
            # cross_scale=[]
            # for i in range(6):#6个encoder
            #     cross_attn= cross_attn_weights_all[i]
            #     cross_attn1= torch.mean( cross_attn, 2).reshape(15,20)#.reshape(100, 100).cpu()  # torch.Size([1, 10000, 256])
            #     image = unloader(cross_attn1)
            #     cross_scale.append(image)
            #
            # fig = plt.figure()
            # # ，每个尺度画6个子图
            # for plt_index in range(1, 7):  # 第一幅图的下标从1开始，设置6张子图
            #     ax = fig.add_subplot(2, 3, plt_index)  # 2行3列
            #     ax.set_title("decoder" + str(plt_index))
            #     ax.imshow(cross_scale[plt_index - 1])
            #     pass
            #
            # plt.savefig('cross' + str(i) + '.png')
            # plt.show()

            # #--------使用列表来存储结果decoder自注意力进行可视化
            # # use lists to store the outputs via up-values
            # de_attn_weights0, de_attn_weights1, de_attn_weights2 = [], [], []  # 卷积特征，
            # de_attn_weights3, de_attn_weights4, de_attn_weights5 = [], [], []
            # # # propagate through the model
            #
            # hooks = [
            #     model.transformer.decoder.layers[0].self_attn.register_forward_hook(
            #         lambda self, input, output: de_attn_weights0.append(output[0].cpu())  # 最后encoder tuple
            #     ),
            #     model.transformer.decoder.layers[1].self_attn.register_forward_hook(
            #         lambda self, input, output: de_attn_weights1.append(output[0].cpu())  # 最后encoder
            #     ),
            #     model.transformer.decoder.layers[2].self_attn.register_forward_hook(
            #         lambda self, input, output: de_attn_weights2.append(output[0].cpu())  # 最后encoder
            #     ),
            #     model.transformer.decoder.layers[3].self_attn.register_forward_hook(
            #         lambda self, input, output: de_attn_weights3.append(output[0].cpu())  # 最后encoder
            #     ),
            #     model.transformer.decoder.layers[4].self_attn.register_forward_hook(
            #         lambda self, input, output: de_attn_weights4.append(output[0].cpu())  # 最后encoder
            #     ),
            #     model.transformer.decoder.layers[5].self_attn.register_forward_hook(
            #         lambda self, input, output: de_attn_weights5.append(output[0].cpu())  # 最后encoder
            #     ),
            # ]
            # outputs = model(img)  # please attention the location
            # for hook in hooks:
            #     hook.remove()
            # print("de_attn_weights0:", de_attn_weights0[0].shape)
            # de_attn_weights_all = [de_attn_weights0[0], de_attn_weights1[0], de_attn_weights2[0],
            #                           de_attn_weights3[0], de_attn_weights4[0], de_attn_weights5[0]]
            #
            # from torchvision import transforms
            # unloader = transforms.ToPILImage()
            # de_scale = []
            # for i in range(6):  # 6个encoder
            #     de_attn = de_attn_weights_all[i]
            #     de_attn1 = torch.mean(de_attn, 2).reshape(15,20)  # .reshape(100, 100).cpu()  # torch.Size([1, 10000, 256])
            #     image = unloader(de_attn1)
            #     de_scale.append(image)
            #
            # fig = plt.figure()
            # # ，每个尺度画6个子图
            # for plt_index in range(1, 7):  # 第一幅图的下标从1开始，设置6张子图
            #     ax = fig.add_subplot(2, 3, plt_index)  # 2行3列
            #     ax.set_title("decoder" + str(plt_index))
            #     ax.imshow(de_scale[plt_index - 1])
            #     pass
            #
            # plt.savefig('decoder-self-attn.png')
            # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("SAM-DETR", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
