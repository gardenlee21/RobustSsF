import os
import time
import logging
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
import random
import numpy as np
import nibabel as nib
import scipy.misc
import cv2
import matplotlib.pyplot as plt

cudnn.benchmark = True

path = os.path.dirname(__file__)
from utils.generate import generate_snapshot

###modality missing mask
mask_indi = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
         [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
         [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
         [True, True, True, True]] #flair, t1c, t1, t2

def softmax_output_dice_class4(output, target):
    eps = 1e-8
    #######label1########
    o1 = (output == 1).float()
    t1 = (target == 1).float()
    intersect1 = torch.sum(2 * (o1 * t1), dim=(1,2,3)) + eps
    denominator1 = torch.sum(o1, dim=(1,2,3)) + torch.sum(t1, dim=(1,2,3)) + eps
    ncr_net_dice = intersect1 / denominator1

    o2 = (output == 2).float()
    t2 = (target == 2).float()
    intersect2 = torch.sum(2 * (o2 * t2), dim=(1,2,3)) + eps
    denominator2 = torch.sum(o2, dim=(1,2,3)) + torch.sum(t2, dim=(1,2,3)) + eps
    edema_dice = intersect2 / denominator2

    o3 = (output == 3).float()
    t3 = (target == 3).float()
    intersect3 = torch.sum(2 * (o3 * t3), dim=(1,2,3)) + eps
    denominator3 = torch.sum(o3, dim=(1,2,3)) + torch.sum(t3, dim=(1,2,3)) + eps
    enhancing_dice = intersect3 / denominator3

    ####post processing:
    if torch.sum(o3) < 500:
       o4 = o3 * 0.0
    else:
       o4 = o3
    t4 = t3
    intersect4 = torch.sum(2 * (o4 * t4), dim=(1,2,3)) + eps
    denominator4 = torch.sum(o4, dim=(1,2,3)) + torch.sum(t4, dim=(1,2,3)) + eps
    enhancing_dice_postpro = intersect4 / denominator4

    o_whole = o1 + o2 + o3 
    t_whole = t1 + t2 + t3 
    intersect_whole = torch.sum(2 * (o_whole * t_whole), dim=(1,2,3)) + eps
    denominator_whole = torch.sum(o_whole, dim=(1,2,3)) + torch.sum(t_whole, dim=(1,2,3)) + eps
    dice_whole = intersect_whole / denominator_whole

    o_core = o1 + o3
    t_core = t1 + t3
    intersect_core = torch.sum(2 * (o_core * t_core), dim=(1,2,3)) + eps
    denominator_core = torch.sum(o_core, dim=(1,2,3)) + torch.sum(t_core, dim=(1,2,3)) + eps
    dice_core = intersect_core / denominator_core

    dice_separate = torch.cat((torch.unsqueeze(ncr_net_dice, 1), torch.unsqueeze(edema_dice, 1), torch.unsqueeze(enhancing_dice, 1)), dim=1)
    dice_evaluate = torch.cat((torch.unsqueeze(dice_whole, 1), torch.unsqueeze(dice_core, 1), torch.unsqueeze(enhancing_dice, 1), torch.unsqueeze(enhancing_dice_postpro, 1)), dim=1)

    return dice_separate.cpu().numpy(), dice_evaluate.cpu().numpy()

def softmax_output_dice_class5(output, target):
    eps = 1e-8
    #######label1########
    o1 = (output == 1).float()
    t1 = (target == 1).float()
    intersect1 = torch.sum(2 * (o1 * t1), dim=(1,2,3)) + eps
    denominator1 = torch.sum(o1, dim=(1,2,3)) + torch.sum(t1, dim=(1,2,3)) + eps
    necrosis_dice = intersect1 / denominator1

    o2 = (output == 2).float()
    t2 = (target == 2).float()
    intersect2 = torch.sum(2 * (o2 * t2), dim=(1,2,3)) + eps
    denominator2 = torch.sum(o2, dim=(1,2,3)) + torch.sum(t2, dim=(1,2,3)) + eps
    edema_dice = intersect2 / denominator2

    o3 = (output == 3).float()
    t3 = (target == 3).float()
    intersect3 = torch.sum(2 * (o3 * t3), dim=(1,2,3)) + eps
    denominator3 = torch.sum(o3, dim=(1,2,3)) + torch.sum(t3, dim=(1,2,3)) + eps
    non_enhancing_dice = intersect3 / denominator3

    o4 = (output == 4).float()
    t4 = (target == 4).float()
    intersect4 = torch.sum(2 * (o4 * t4), dim=(1,2,3)) + eps
    denominator4 = torch.sum(o4, dim=(1,2,3)) + torch.sum(t4, dim=(1,2,3)) + eps
    enhancing_dice = intersect4 / denominator4

    ####post processing:
    if torch.sum(o4) < 500:
        o5 = o4 * 0
    else:
        o5 = o4
    t5 = t4
    intersect5 = torch.sum(2 * (o5 * t5), dim=(1,2,3)) + eps
    denominator5 = torch.sum(o5, dim=(1,2,3)) + torch.sum(t5, dim=(1,2,3)) + eps
    enhancing_dice_postpro = intersect5 / denominator5

    o_whole = o1 + o2 + o3 + o4
    t_whole = t1 + t2 + t3 + t4
    intersect_whole = torch.sum(2 * (o_whole * t_whole), dim=(1,2,3)) + eps
    denominator_whole = torch.sum(o_whole, dim=(1,2,3)) + torch.sum(t_whole, dim=(1,2,3)) + eps
    dice_whole = intersect_whole / denominator_whole

    o_core = o1 + o3 + o4
    t_core = t1 + t3 + t4
    intersect_core = torch.sum(2 * (o_core * t_core), dim=(1,2,3)) + eps
    denominator_core = torch.sum(o_core, dim=(1,2,3)) + torch.sum(t_core, dim=(1,2,3)) + eps
    dice_core = intersect_core / denominator_core

    dice_separate = torch.cat((torch.unsqueeze(necrosis_dice, 1), torch.unsqueeze(edema_dice, 1), torch.unsqueeze(non_enhancing_dice, 1), torch.unsqueeze(enhancing_dice, 1)), dim=1)
    dice_evaluate = torch.cat((torch.unsqueeze(dice_whole, 1), torch.unsqueeze(dice_core, 1), torch.unsqueeze(enhancing_dice, 1), torch.unsqueeze(enhancing_dice_postpro, 1)), dim=1)

    return dice_separate.cpu().numpy(), dice_evaluate.cpu().numpy()


def test_softmax(
        test_loader,
        model,
        dataname = 'BRATS2020',
        feature_mask=None,
        mask_name=None,
        lls_end_epoch=100,
        lls_end_iter=10):

    H, W, T = 240, 240, 155
    model.eval()
    vals_evaluation = AverageMeter()
    vals_separate = AverageMeter()
    one_tensor = torch.ones(1, 80, 80, 80).float().cuda()

    if dataname in ['BRATS2020', 'BRATS2018']:
        num_cls = 4
        class_evaluation= 'whole', 'core', 'enhancing', 'enhancing_postpro'
        class_separate = 'ncr_net', 'edema', 'enhancing'
    elif dataname == 'BRATS2015':
        num_cls = 5
        class_evaluation= 'whole', 'core', 'enhancing', 'enhancing_postpro'
        class_separate = 'necrosis', 'edema', 'non_enhancing', 'enhancing'
        

    for i, data in enumerate(test_loader):
        target = data[1].cuda()
        x = data[0].cuda()
        names = data[-1]
        if feature_mask is not None:
            mask = torch.from_numpy(np.array(feature_mask))
            mask = torch.unsqueeze(mask, dim=0).repeat(len(names), 1)
        else:
            mask = data[2]
        mask = mask.cuda()
        _, _, H, W, Z = x.size()
        #########get h_ind, w_ind, z_ind for sliding windows
        h_cnt = np.int(np.ceil((H - 80) / (80 * (1 - 0.5))))
        h_idx_list = range(0, h_cnt)
        h_idx_list = [h_idx * np.int(80 * (1 - 0.5)) for h_idx in h_idx_list]
        h_idx_list.append(H - 80)

        w_cnt = np.int(np.ceil((W - 80) / (80 * (1 - 0.5))))
        w_idx_list = range(0, w_cnt)
        w_idx_list = [w_idx * np.int(80 * (1 - 0.5)) for w_idx in w_idx_list]
        w_idx_list.append(W - 80)

        z_cnt = np.int(np.ceil((Z - 80) / (80 * (1 - 0.5))))
        z_idx_list = range(0, z_cnt)
        z_idx_list = [z_idx * np.int(80 * (1 - 0.5)) for z_idx in z_idx_list]
        z_idx_list.append(Z - 80)

        #####compute calculation times for each pixel in sliding windows
        weight1 = torch.zeros(1, 1, H, W, Z).float().cuda()
        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    weight1[:, :, h:h+80, w:w+80, z:z+80] += one_tensor
        weight = weight1.repeat(len(names), num_cls, 1, 1, 1)

        #####evaluation
        pred = torch.zeros(len(names), num_cls, H, W, Z).float().cuda()
        model.module.is_training=False
        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    x_input = x[:, :, h:h+80, w:w+80, z:z+80]
                    pred_parts = model(x_input, mask, 300, 50, lls_end_epoch, lls_end_iter)
                    pred[:, :, h:h+80, w:w+80, z:z+80] += torch.maximum(pred_parts[0], pred_parts[1]) #max
        pred = pred / weight
        b = time.time()
        pred = pred[:, :, :H, :W, :T]
        pred = torch.argmax(pred, dim=1)

        if dataname in ['BRATS2020', 'BRATS2018']:
            scores_separate, scores_evaluation = softmax_output_dice_class4(pred, target)
        elif dataname == 'BRATS2015':
            scores_separate, scores_evaluation = softmax_output_dice_class5(pred, target)
        for k, name in enumerate(names):
            #msg = 'Subject {}/{}, {}/{}'.format((i+1), len(test_loader), (k+1), len(names))
            #msg += '{:>20}, '.format(name)

            vals_separate.update(scores_separate[k])
            vals_evaluation.update(scores_evaluation[k])
            #msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, scores_evaluation[k])])
            #msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, scores_separate[k])])

            #logging.info(msg)
    msg = 'Average scores:'
    msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, vals_evaluation.avg)])
    #msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, vals_evaluation.avg)])
    print (msg)
    model.train()
    return vals_evaluation.avg

def visualization(test_loader,
        model,
        dataname = 'BRATS2020',
        feature_mask = None,
        lls_end_epoch=100,
        lls_end_iter=10) :
 
    H, W, T = 240, 240, 155
    model.eval()
    vals_evaluation = AverageMeter()
    vals_separate = AverageMeter()
    one_tensor = torch.ones(1, 80, 80, 80).float().cuda()

    if dataname in ['BRATS2020', 'BRATS2018']:
        num_cls = 4
        class_evaluation= 'whole', 'core', 'enhancing', 'enhancing_postpro'
        class_separate = 'ncr_net', 'edema', 'enhancing'
    elif dataname == 'BRATS2015':
        num_cls = 5
        class_evaluation= 'whole', 'core', 'enhancing', 'enhancing_postpro'
        class_separate = 'necrosis', 'edema', 'non_enhancing', 'enhancing'
    
    colors = np.random.randint(0, 255, size=(4, 3))
    colors = [tuple(color) for color in colors]    

    for i, data in enumerate(test_loader):
        target = data[1].cuda() #gt shape=[1,144,182,143]
        x = data[0].cuda() #input shape=[4,144,182,143]
        names = data[-1]
        # if feature_mask is not None:
        #     mask = torch.from_numpy(np.array(feature_mask))
        #     mask = torch.unsqueeze(mask, dim=0).repeat(len(names), 1)
        # else:
        #     mask = data[2]
        # mask = mask.cuda()
        _, _, H, W, Z = x.size()
        #########get h_ind, w_ind, z_ind for sliding windows
        h_cnt = np.int(np.ceil((H - 80) / (80 * (1 - 0.5))))
        h_idx_list = range(0, h_cnt)
        h_idx_list = [h_idx * np.int(80 * (1 - 0.5)) for h_idx in h_idx_list]
        h_idx_list.append(H - 80)

        w_cnt = np.int(np.ceil((W - 80) / (80 * (1 - 0.5))))
        w_idx_list = range(0, w_cnt)
        w_idx_list = [w_idx * np.int(80 * (1 - 0.5)) for w_idx in w_idx_list]
        w_idx_list.append(W - 80)

        z_cnt = np.int(np.ceil((Z - 80) / (80 * (1 - 0.5))))
        z_idx_list = range(0, z_cnt)
        z_idx_list = [z_idx * np.int(80 * (1 - 0.5)) for z_idx in z_idx_list]
        z_idx_list.append(Z - 80)
        
        s = random.sample(range(50,H-30), 3) #pick random slides
        #mask_index = mask_indi.index(feature_mask) #mask_index=missingmodality
        plt.axis('off')
        #plt.axes.get_xaxis().set_visible(False)
        
        for s_index in s:
            image_path_tar = '/home/gardenlee21/RFNet-main/image_path/' + str(i) + '_' + str(s_index) + '_target.png' #i=test_loader number, s_index=slide, ex)10_110_target.png
            #import pdb;pdb.set_trace()
            image_tar = target[:, :, :, s_index].squeeze().cpu().numpy()
            plt.imshow(image_tar, cmap='gnuplot')
            plt.savefig(image_path_tar, bbox_inches='tight', pad_inches=0)
            #save_image(target[:, :, :, s_index].float(), image_path_tar)
            print(image_path_tar)
            
            for j in range(4):
                image_path_x = '/home/gardenlee21/RFNet-main/image_path/'+ str(i) + '_' + str(s_index) + '_modal_' + str(j) + '.png' #ex) 10_110_modal_1.png
                image_x = x[:, j, :, :, s_index].squeeze().cpu().numpy()
                plt.imshow(image_x, cmap='gray')
                plt.savefig(image_path_x, bbox_inches='tight', pad_inches=0)
                #save_image(x[:, j, :, :, s_index].float(), image_path_x)
                print(image_path_x)
            
            for mask_index in range(15):
                mask_vis = mask_indi[::-1][mask_index]
                mask_vis = torch.from_numpy(np.array(mask_vis))
                mask_vis = torch.unsqueeze(mask_vis, dim=0).repeat(len(names), 1)
                mask_vis = mask_vis.cuda()
                #pred_parts = model(x, mask, 300, 50, lls_end_epoch, lls_end_iter) #shape=
                #pred_parts = model(x, mask_vis, 300, 50, lls_end_epoch, lls_end_iter) #shape=
                #mask_index = mask_indi.index(mask_vis)
                
                #####compute calculation times for each pixel in sliding windows
                weight1 = torch.zeros(1, 1, H, W, Z).float().cuda()
                for h in h_idx_list:
                    for w in w_idx_list:
                        for z in z_idx_list:
                            weight1[:, :, h:h+80, w:w+80, z:z+80] += one_tensor
                weight = weight1.repeat(len(names), num_cls, 1, 1, 1)

                #####evaluation
                pred = torch.zeros(len(names), num_cls, H, W, Z).float().cuda()
                model.module.is_training=False
                for h in h_idx_list:
                    for w in w_idx_list:
                        for z in z_idx_list:
                            x_input = x[:, :, h:h+80, w:w+80, z:z+80]
                            pred_parts = model(x_input, mask_vis, 300, 50, lls_end_epoch, lls_end_iter)
                            pred[:, :, h:h+80, w:w+80, z:z+80] += torch.maximum(pred_parts[0], pred_parts[1]) #max
                pred = pred / weight
                b = time.time()
                pred = pred[:, :, :H, :W, :T]
                pred = torch.argmax(pred, dim=1)
        
                image_path_pred = '/home/gardenlee21/RFNet-main/image_path/'+ str(i) + '_' + str(s_index) + '_mask_' + str(mask_index) + '_pred.png' #ex)10_110_mask_4_pred.png
                #import pdb;pdb.set_trace()
                #pred_vis = pred[:, s_index, :, :]
                image_pred = pred[:, :, :, s_index].squeeze().cpu().numpy()
                plt.imshow(image_pred, cmap='gnuplot')
                plt.savefig(image_path_pred, bbox_inches='tight', pad_inches=0)
                #save_image(pred[:, :, :, s_index].float(), image_path_pred)
                print(image_path_pred)
                #image_path_output2 = '/home/gardenlee21/RFNet-main/image_path/'+ str(i) + '_' + str(s_index) + '_mask_' + str(mask_index) + 'output2.png' #ex)10_110_mask_4_output2.png
                #save_image(pred_parts[1, :, s_index, :, :], image_path_output2)

    return 
           

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
