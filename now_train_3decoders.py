#coding=utf-8
import argparse
import os
import time
import logging
import random
import numpy as np
from collections import OrderedDict

from itertools import combinations
import torch
import torch.optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

import models
from data.transforms import *
from data.datasets_nii import Brats_loadall_nii, Brats_loadall_test_nii
from data.data_utils import init_fn
from utils import Parser,criterions
from utils.parser import setup 
from utils.lr_scheduler import LR_Scheduler, record_loss, MultiEpochsDataLoader 
from predict_max import AverageMeter, test_softmax

os.environ["CUDA_VISIBLE_DEVICES"]="5"
torch.cuda.empty_cache()

parser = argparse.ArgumentParser()

parser.add_argument('-batch_size', '--batch_size', default=1, type=int, help='Batch size')
parser.add_argument('--datapath', default='/home/gardenlee21/data/BRATS2018_Training_none_npy/', type=str)
parser.add_argument('--dataname', default='BRATS2018', type=str)
parser.add_argument('--savepath', default='/home/gardenlee21/RFNet-main/logs/0315_ssfnl_best', type=str)
parser.add_argument('--resume', default='/home/gardenlee21/RFNet-main/logs/0315_ssfnl_best/model_last.pth', type=str)
parser.add_argument('--pretrain', default=None, type=str)
parser.add_argument('--lr', default=2e-4, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--iter_per_epoch', default=50, type=int)
parser.add_argument('--lls_end_epoch', default=60, type=int)
parser.add_argument('--lls_end_iter', default=10, type=int)#1/5
parser.add_argument('--region_fusion_start_epoch', default=20, type=int)
#parser.add_argument('--couple_reg_start_epoch', default=80, type=int)
parser.add_argument('--couple_reg_parameter', default=0.01, type=int)
parser.add_argument('--prm_couple_reg_parameter', default=0.01, type=int)
#parser.add_argument('--seed', default=1024, type=int)
path = os.path.dirname(__file__)

## parse arguments
args = parser.parse_args()
setup(args, 'training')
args.train_transforms = 'Compose([RandCrop3D((80,80,80)), RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])'
args.test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'

ckpts = args.savepath
os.makedirs(ckpts, exist_ok=True)

###tensorboard writer
writer = SummaryWriter(os.path.join(args.savepath, 'summary'))

###modality missing mask
mask_indi = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
         [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
         [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
         [True, True, True, True]]
masks_torch = torch.from_numpy(np.array(mask_indi))

mask_name = ['t2', 't1c', 't1', 'flair', 
            't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
            'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
            'flairt1cet1t2']
print (masks_torch.int()) 

def main():
    ##########setting seed
    #random_seed = random.randint(1,1024)
    random_seed=1024
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    print("random seed value: ", random_seed)
    ##########setting models
    if args.dataname in ['BRATS2020', 'BRATS2018']:
        num_cls = 4
    elif args.dataname == 'BRATS2015':
        num_cls = 5
    else:
        print ('dataset is error')
        exit(0)
    model = models.Model(num_cls=num_cls)
    #print (model)
    model = torch.nn.DataParallel(model).cuda()
    
    ##########Setting learning schedule and optimizer
    lr_schedule = LR_Scheduler(args.lr, args.num_epochs)
    train_params = [{'params': model.parameters(), 'lr': args.lr, 'weight_decay':args.weight_decay}]
    optimizer = torch.optim.Adam(train_params,  betas=(0.9, 0.999), eps=1e-08, amsgrad=True)

    ##########Setting data
    if args.dataname in ['BRATS2020', 'BRATS2015']:
        train_file = 'train.txt'
        test_file = 'test.txt'
    elif args.dataname == 'BRATS2018':
        ####BRATS2018 contains three splits (1,2,3)
        train_file = 'train1.txt'
        test_file = 'test1.txt'

    logging.info(str(args))
    train_set = Brats_loadall_nii(transforms=args.train_transforms, root=args.datapath, num_cls=num_cls, train_file=train_file)
    test_set = Brats_loadall_test_nii(transforms=args.test_transforms, root=args.datapath, test_file=test_file)
    train_loader = MultiEpochsDataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=init_fn)
    test_loader = MultiEpochsDataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)

    ##########Evaluate
    if args.resume is not None:
        loaded_state_dict = torch.load(args.resume)['state_dict']
        model.load_state_dict(loaded_state_dict)
        #logging.info('best epoch: {}'.format(checkpoint['epoch']))
        #model.load_state_dict(checkpoint['state_dict'])
        
        test_score1 = AverageMeter()
        test_score2 = AverageMeter()
        test_scoref = AverageMeter()
        with torch.no_grad():
            logging.info('###########test set wi post process###########')
            for i, mask in enumerate(mask_indi[::-1]):
                logging.info('{}'.format(mask_name[::-1][i]))
                dice_score1, dice_score2, dice_scoref, cluster_pred = test_softmax(
                                                                    test_loader,
                                                                    model,
                                                                    dataname = args.dataname,
                                                                    feature_mask = mask,
                                                                    mask_name = mask_name[::-1][i],
                                                                    lls_end_epoch=args.lls_end_epoch,
                                                                    lls_end_iter=args.lls_end_iter)
                test_score1.update(dice_score1)
                test_score2.update(dice_score2)
                test_scoref.update(dice_scoref)
                plt.scatter(cluster_pred[0], cluster_pred[1], cluster_pred[2], marker='.', label=i)
            #logging.basicConfig(filename=args.savepath+"_test.log", level=logging.DEBUG)
            logging.info('Avg scores: {}\n'.format(test_score1.avg))
            logging.info('Avg scores: {}\n'.format(test_score2.avg))
            logging.info('Avg scores: {}\n'.format(test_scoref.avg))

            plt.legend()
            plt.savefig(args.savepath+'/fsne_pred.png')
            
            exit(0)

    ##########Training
    start = time.time()
    torch.set_grad_enabled(True)
    logging.info('#############training############')
    iter_per_epoch = args.iter_per_epoch
    train_iter = iter(train_loader)
    for epoch in range(args.num_epochs):
        step_lr = lr_schedule(optimizer, epoch)
        writer.add_scalar('lr', step_lr, global_step=(epoch+1))
        b = time.time()
        total_loss_avg = AverageMeter()
        fuse_loss_avg = AverageMeter()
        sep_loss_avg = AverageMeter()
        prm_loss_avg = AverageMeter()
        couple_reg_loss_avg = AverageMeter()
        prm_couple_reg_loss_avg = AverageMeter()
        for i in range(iter_per_epoch):
            step = (i+1) + epoch*iter_per_epoch
            ###Data load
            try:
                data = next(train_iter)
            except:
                train_iter = iter(train_loader)
                data = next(train_iter)
            #len(data)=4,index 3 == name (not used)
            x, target, mask = data[:3]
            
            #print(mask.shape)
            
            x = x.cuda(non_blocking=True) #shape: [batch_size,4,80,80,80]
            target = target.cuda(non_blocking=True) #shape: [batch_size,4,80,80,80]
            mask = mask.cuda(non_blocking=True) #shape: [batch_size,4]
            
            model.module.is_training = True
            #import pdb; pdb.set_trace()
            fuse_preds, sep_preds, prm_predss = model(x, mask, epoch, i, args.lls_end_epoch, args.lls_end_iter)
            #fuse_preds[0,1,2] shape: [batch_size,4,80,80,80]
            #sep_preds[0,1,2,3] shape: [batch_size,4,80,80,80]
            #prm_predss[0,1,2] shape: [batch_size,4,80,80,80]
            
            ###Loss compute
            fuse_cross_loss = torch.zeros(1).cuda().float()
            fuse_dice_loss = torch.zeros(1).cuda().float()
            for fuseindex in range(3): #0,1,2
                fuse_pred = fuse_preds[fuseindex] 
                fuse_cross_loss += criterions.softmax_weighted_loss(fuse_pred, target, num_cls=num_cls)
                fuse_dice_loss += criterions.dice_loss(fuse_pred, target, num_cls=num_cls)
            fuse_loss = fuse_dice_loss + fuse_cross_loss

            sep_cross_loss = torch.zeros(1).cuda().float()
            sep_dice_loss = torch.zeros(1).cuda().float()
            for sep_pred in sep_preds:
                sep_cross_loss += criterions.softmax_weighted_loss(sep_pred, target, num_cls=num_cls)
                sep_dice_loss += criterions.dice_loss(sep_pred, target, num_cls=num_cls)
            sep_loss = sep_dice_loss + sep_cross_loss

            prm_cross_loss = torch.zeros(1).cuda().float()
            prm_dice_loss = torch.zeros(1).cuda().float()
            for prm_preds in prm_predss:
                for prm_pred in prm_preds:
                    prm_cross_loss += criterions.softmax_weighted_loss(prm_pred, target, num_cls=num_cls)
                    prm_dice_loss += criterions.dice_loss(prm_pred, target, num_cls=num_cls)
            prm_loss = prm_dice_loss + prm_cross_loss

            #####couple reg loss computation
            ''' feature_couple = fuse_preds[0] + fuse_preds[1] #simply add
            feature_full = fuse_preds[2]
            couple_reg_cross_loss = criterions.softmax_weighted_loss(feature_couple, feature_full, num_cls=num_cls)
            couple_reg_dice_loss = criterions.dice_loss(feature_couple, feature_full, num_cls=num_cls)
            couple_reg_loss = couple_reg_cross_loss + couple_reg_dice_loss '''
            feature_full = fuse_preds[2]
            if epoch >= args.region_fusion_start_epoch and epoch < args.lls_end_epoch: #no matter iter
                couple_reg_cross_loss = torch.zeros(1).cuda().float()
                couple_reg_dice_loss = torch.zeros(1).cuda().float()
                features = (fuse_preds[0], fuse_preds[1])
                for feature in features:
                    couple_reg_cross_loss += criterions.softmax_weighted_loss(feature, feature_full, num_cls=num_cls)
                    couple_reg_dice_loss += criterions.dice_loss(feature, feature_full, num_cls=num_cls)
                couple_reg_loss = couple_reg_cross_loss + couple_reg_dice_loss
            elif epoch >= args.lls_end_epoch:
                if i < args.lls_end_iter:
                    couple_reg_cross_loss = torch.zeros(1).cuda().float()
                    couple_reg_dice_loss = torch.zeros(1).cuda().float()
                    features = (fuse_preds[0], fuse_preds[1])
                    for feature in features:
                        couple_reg_cross_loss += criterions.softmax_weighted_loss(feature, feature_full, num_cls=num_cls)
                        couple_reg_dice_loss += criterions.dice_loss(feature, feature_full, num_cls=num_cls)
                    couple_reg_loss = couple_reg_cross_loss + couple_reg_dice_loss
                else:
                    #feature_couple = fuse_preds[0] + fuse_preds[1] #simply add
                    feature_couple = fuse_preds[3] #concat+deconv
                    couple_reg_cross_loss = criterions.softmax_weighted_loss(feature_couple, feature_full, num_cls=num_cls)
                    couple_reg_dice_loss = criterions.dice_loss(feature_couple, feature_full, num_cls=num_cls)
                    couple_reg_loss = couple_reg_cross_loss + couple_reg_dice_loss
            else:
                #feature_couple = fuse_preds[0] + fuse_preds[1] #simply add
                feature_couple = fuse_preds[3] #concat+deconv
                couple_reg_cross_loss = criterions.softmax_weighted_loss(feature_couple, feature_full, num_cls=num_cls)
                couple_reg_dice_loss = criterions.dice_loss(feature_couple, feature_full, num_cls=num_cls)
                couple_reg_loss = couple_reg_cross_loss + couple_reg_dice_loss
            
            #####prm couple reg loss
            prms_full = prm_predss[2]
            prm_couple_cross_loss = torch.zeros(1).cuda().float()
            prm_couple_dice_loss = torch.zeros(1).cuda().float()
            if epoch >= args.region_fusion_start_epoch and epoch < args.lls_end_epoch: #no matter iter
                prmss = (prm_predss[0], prm_predss[1])
                for prms in prmss:
                    for j in range(4):
                        prm_couple_cross_loss += criterions.softmax_weighted_loss(prms[j], prms_full[j], num_cls=num_cls)
                        prm_couple_dice_loss += criterions.dice_loss(prms[j], prms_full[j], num_cls=num_cls)
                prm_couple_reg_loss = prm_couple_cross_loss + prm_couple_dice_loss
            elif epoch >= args.lls_end_epoch:
                if i < args.lls_end_iter:
                    prmss = (prm_predss[0], prm_predss[1])
                    for prms in prmss:
                        for j in range(4):
                            prm_couple_cross_loss += criterions.softmax_weighted_loss(prms[j], prms_full[j], num_cls=num_cls)
                            prm_couple_dice_loss += criterions.dice_loss(prms[j], prms_full[j], num_cls=num_cls)
                    prm_couple_reg_loss = prm_couple_cross_loss + prm_couple_dice_loss
                else:
                    for j in range(4):
                        prms_couple = prm_predss[0] + prm_predss[1] #simply add
                        prm_couple_cross_loss += criterions.softmax_weighted_loss(prms_couple[j], prms_full[j], num_cls=num_cls)
                        prm_couple_dice_loss += criterions.dice_loss(prms_couple[j], prms_full[j], num_cls=num_cls)
                        prm_couple_reg_loss = prm_couple_cross_loss + prm_couple_dice_loss
            else:
                for j in range(4):
                    prms_couple = prm_predss[0] + prm_predss[1] #simply add
                    prm_couple_cross_loss += criterions.softmax_weighted_loss(prms_couple[j], prms_full[j], num_cls=num_cls)
                    prm_couple_dice_loss += criterions.dice_loss(prms_couple[j], prms_full[j], num_cls=num_cls)
                    prm_couple_reg_loss = prm_couple_cross_loss + prm_couple_dice_loss
            ''' #sep loss 삭제
            if epoch < args.region_fusion_start_epoch:
                loss = fuse_loss #+ fuse_couple_loss * args.couple_reg_parameter * 0.0 #+ prm_loss + sep_loss
            else:
                loss = fuse_loss + couple_reg_loss * args.couple_reg_parameter #+ prm_loss + sep_loss '''
                
            if epoch < args.region_fusion_start_epoch:
                loss = fuse_loss * 0.0 + prm_loss + sep_loss \
                    + couple_reg_loss * args.couple_reg_parameter * 0.0 + prm_couple_reg_loss * args.prm_couple_reg_parameter * 0.0
                #0이더라도 call을 해야 loss 관련 computational graph가 store되지 않음
            #fusion start
            else:
                loss = fuse_loss + prm_loss + sep_loss \
                    + couple_reg_loss * args.couple_reg_parameter + prm_couple_reg_loss * args.prm_couple_reg_parameter

            ''' #couple reg 먼저, 그다음 fusion
            if epoch < args.couple_reg_start_epoch:
                loss = fuse_loss * 0.0 + sep_loss + prm_loss + fuse_couple_loss * args.couple_reg_parameter * 0.0
            elif epoch < args.region_fusion_start_epoch:
                loss = fuse_loss * 0.0 + sep_loss + prm_loss + fuse_couple_loss * args.couple_reg_parameter
            else:
                loss = fuse_loss + sep_loss + prm_loss + fuse_couple_loss * args.couple_reg_parameter '''
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ###log
            writer.add_scalar('loss', loss.item(), global_step=step)
            writer.add_scalar('fuse_cross_loss', fuse_cross_loss.item(), global_step=step)
            writer.add_scalar('fuse_dice_loss', fuse_dice_loss.item(), global_step=step)
            writer.add_scalar('sep_cross_loss', sep_cross_loss.item(), global_step=step)
            writer.add_scalar('sep_dice_loss', sep_dice_loss.item(), global_step=step)
            writer.add_scalar('prm_cross_loss', prm_cross_loss.item(), global_step=step)
            writer.add_scalar('prm_dice_loss', prm_dice_loss.item(), global_step=step)
            writer.add_scalar('couple_reg_cross_loss', couple_reg_cross_loss.item(), global_step=step)
            writer.add_scalar('couple_reg_dice_loss', couple_reg_dice_loss.item(), global_step=step)
            writer.add_scalar('prm_couple_cross_loss', prm_couple_cross_loss.item(), global_step=step)
            writer.add_scalar('prm_couple_dice_loss', prm_couple_dice_loss.item(), global_step=step)
            
            total_loss_avg.update(loss.item())
            fuse_loss_avg.update(fuse_loss.item())
            sep_loss_avg.update(sep_loss.item())
            prm_loss_avg.update(prm_loss.item())
            couple_reg_loss_avg.update(couple_reg_loss.item())
            prm_couple_reg_loss_avg.update(prm_couple_reg_loss.item())
            #msg = 'Epoch {}/{}, Iter {}/{}, Loss {:.4f}, '.format((epoch+1), args.num_epochs, (i+1), iter_per_epoch, loss.item())
            #msg += 'fusecross:{:.4f}, fusedice:{:.4f},'.format(fuse_cross_loss.item(), fuse_dice_loss.item())
            #msg += 'sepcross:{:.4f}, sepdice:{:.4f},'.format(sep_cross_loss.item(), sep_dice_loss.item())
            #msg += 'prmcross:{:.4f}, prmdice:{:.4f},'.format(prm_cross_loss.item(), prm_dice_loss.item())
            #msg += 'couplecross:{:.4f}, coupledice:{:.4f},'.format(couple_cross_loss.item(), couple_dice_loss.item())
            #logging.info(msg)
            if epoch >= args.lls_end_epoch and i == args.lls_end_iter:
                print("lls is ended")
        
        if epoch < args.region_fusion_start_epoch:
            print("fusion is not started, couple reg is not started")
        if epoch >= args.region_fusion_start_epoch and epoch < args.lls_end_epoch:
            print("fusion & couple reg with lls")
        if epoch >= args.lls_end_epoch:
            print("fusion & couple reg")
            
        ''' #couple reg 먼저, 그다음 fusion
        if epoch < args.couple_reg_start_epoch:
            print("fusion is not started, couple reg is not started")
        elif epoch < args.region_fusion_start_epoch:
            print("fusion is not started") '''
        
        ''' #같이
        if epoch < args.region_fusion_start_epoch:
            print("fusion is not started, couple reg is not started")  '''
        msg = 'Epoch {}/{}, Loss:{:.4f}, '.format((epoch+1), args.num_epochs, total_loss_avg.avg)
        #msg += 'fuseloss:{:.4f}, seploss:{:.4f}, prmloss:{:.4f}'.format(fuse_loss_avg.avg,sep_loss_avg.avg, prm_loss_avg.avg)
        #msg += 'fuseloss:{:.4f}, couopleregloss:{:.4f}'.format(fuse_loss_avg.avg, couple_reg_loss_avg.avg)
        msg += 'fuseloss:{:.4f}, seploss:{:.4f}, prmloss:{:.4f}, coupleregloss:{:.4f}, prmcoupleregloss:{:.4f}'.format(fuse_loss_avg.avg,\
            sep_loss_avg.avg, prm_loss_avg.avg, couple_reg_loss_avg.avg, prm_couple_reg_loss_avg.avg)
        logging.info(msg)
        logging.info('train time per epoch: {}'.format(time.time() - b))

        ##########model save
        file_name = os.path.join(ckpts, 'model_last.pth')
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            },
            file_name)
        
        if (epoch+1) % 50 == 0 or (epoch>=(args.num_epochs-10)):
            file_name = os.path.join(ckpts, 'model_{}.pth'.format(epoch+1))
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
                },
                file_name)


    msg = 'total time: {:.4f} hours'.format((time.time() - start)/3600)
    logging.info(msg)

    ##########Evaluate the last epoch model
    test_score = AverageMeter()
    with torch.no_grad():
        logging.info('###########test set wi/wo postprocess###########')
        for i, mask in enumerate(mask_indi):
            #logging.info('{}'.format(mask_name[i]))
            dice_score = test_softmax(
                            test_loader,
                            model,
                            dataname = args.dataname,
                            feature_mask = mask,
                            lls_end_epoch=args.lls_end_epoch,
                            lls_end_iter=args.lls_end_iter)
            test_score.update(dice_score)
        logging.info('Avg scores: {}'.format(test_score.avg))

if __name__ == '__main__':
    main()
