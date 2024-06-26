""" Training routine for GraspNet baseline model. """

import os
import sys
import numpy as np
from datetime import datetime
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'prof'))
from graspnet import GraspNet, get_loss
from pytorch_utils import BNMomentumScheduler
from label_generation import process_grasp_labels
from prof import memstat

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', required=True, help='Dataset root')
parser.add_argument('--train_dataset', required=True, default='train_small', help='Train Dataset type [default: train_small]')
parser.add_argument('--test_dataset', required=True, default='test_small', help='Test Dataset type [default: test_small]')
parser.add_argument('--camera', required=True, help='Camera split [realsense/kinect]')
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--log_dir', default='log', help='Dump dir to save model checkpoint [default: log]')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--max_epoch', type=int, default=18, help='Epoch to run [default: 18]')
parser.add_argument('--batch_size', type=int, default=2, help='Batch Size during training [default: 2]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--weight_decay', type=float, default=0, help='Optimization L2 weight decay [default: 0]')
parser.add_argument('--bn_decay_step', type=int, default=2, help='Period of BN decay (in epochs) [default: 2]')
parser.add_argument('--bn_decay_rate', type=float, default=0.5, help='Decay rate for BN decay [default: 0.5]')
parser.add_argument('--lr_decay_steps', default='8,12,16', help='When to decay the learning rate (in epochs) [default: 8,12,16]')
parser.add_argument('--lr_decay_rates', default='0.1,0.1,0.1', help='Decay rates for lr decay [default: 0.1,0.1,0.1]')
parser.add_argument('--graspnet_lazy_mode', default='True', help='GraspNet lazy mode enable to load labels when needed [default: True]')
cfgs = parser.parse_args()

LAZY_MODE_ENABLED = cfgs.graspnet_lazy_mode
if LAZY_MODE_ENABLED == False:
    # --- original code (use memory up to 40GB)---#
    # https://blog.csdn.net/qq_38056431/article/details/123208602
    from graspnet_dataset import GraspNetDataset, collate_fn, load_grasp_labels
else:
    # --- tolerance label load when needed ---#
    from graspnet_dataset_lazy import GraspNetDataset, collate_fn, load_grasp_labels, load_grasp_labels_list

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
EPOCH_CNT = 0
LR_DECAY_STEPS = [int(x) for x in cfgs.lr_decay_steps.split(',')]
LR_DECAY_RATES = [float(x) for x in cfgs.lr_decay_rates.split(',')]
assert(len(LR_DECAY_STEPS)==len(LR_DECAY_RATES))
DEFAULT_CHECKPOINT_PATH = os.path.join(cfgs.log_dir, 'checkpoint-rs.tar')
CHECKPOINT_PATH = cfgs.checkpoint_path if cfgs.checkpoint_path is not None \
    else DEFAULT_CHECKPOINT_PATH

if not os.path.exists(cfgs.log_dir):
    os.makedirs(cfgs.log_dir)

LOG_FOUT = open(os.path.join(cfgs.log_dir, 'log_train.txt'), 'a')
LOG_FOUT.write(str(cfgs)+'\n')
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

# Init datasets and dataloaders 
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass

# Create Dataset and Dataloader
print("Before DATASET...")
print("train_dataset, test_dataset: ", cfgs.train_dataset, cfgs.test_dataset)
print("LAZY_MODE_ENABLED: ", LAZY_MODE_ENABLED)
if LAZY_MODE_ENABLED == False:
    valid_obj_idxs, grasp_labels = load_grasp_labels(cfgs.dataset_root)
    TRAIN_DATASET = GraspNetDataset(cfgs.dataset_root, valid_obj_idxs, grasp_labels, camera=cfgs.camera, split=cfgs.train_dataset, num_points=cfgs.num_point, remove_outlier=True, augment=True)
    print("TRAIN_DATASET...")
    memstat()
    TEST_DATASET = GraspNetDataset(cfgs.dataset_root, valid_obj_idxs, grasp_labels, camera=cfgs.camera, split=cfgs.test_dataset, num_points=cfgs.num_point, remove_outlier=True, augment=False)
    print("TEST_DATASET...")
    memstat()
else:
    valid_obj_idxs, grasp_labels_list = load_grasp_labels_list(cfgs.dataset_root)
    TRAIN_DATASET = GraspNetDataset(cfgs.dataset_root, valid_obj_idxs, None, grasp_labels_list, camera=cfgs.camera, split=cfgs.train_dataset, num_points=cfgs.num_point, remove_outlier=True, augment=True)
    print("TRAIN_DATASET...")
    memstat()
    TEST_DATASET = GraspNetDataset(cfgs.dataset_root, valid_obj_idxs, None, grasp_labels_list, camera=cfgs.camera, split=cfgs.test_dataset, num_points=cfgs.num_point, remove_outlier=True, augment=False)
    print("TEST_DATASET...")
    memstat()

print("Before DATALOADER...")
memstat()
TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=cfgs.batch_size, shuffle=True,
    num_workers=4, worker_init_fn=my_worker_init_fn, collate_fn=collate_fn)
print("TRAIN_DATALOADER...")
memstat()
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=cfgs.batch_size, shuffle=False,
    num_workers=4, worker_init_fn=my_worker_init_fn, collate_fn=collate_fn)
print("TEST_DATALOADER...")
memstat()
print("Dataset size (train, test):", len(TRAIN_DATASET), len(TEST_DATASET))           #25600 7680
print("Dataloader size (train, test):", len(TRAIN_DATALOADER), len(TEST_DATALOADER))  #12800 3840

# Init the model and optimzier
net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
                        cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# total_num_gpus = torch.cuda.device_count()
# if total_num_gpus > 1:
#     print("Train with {} GPUs".format(total_num_gpus))
#     net = torch.nn.DataParallel(net)
net.to(device)
# Load the Adam optimizer
optimizer = optim.Adam(net.parameters(), lr=cfgs.learning_rate, weight_decay=cfgs.weight_decay)
# Load checkpoint if there is any
it = -1 # for the initialize value of `LambdaLR` and `BNMomentumScheduler`
start_epoch = 0
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    # if total_num_gpus > 1:
    #     for key_id in list(checkpoint['model_state_dict'].keys()):
    #         # print(key_id, "---> module." +key_id)
    #         checkpoint['model_state_dict']['module.'+key_id]=checkpoint['model_state_dict'][key_id]
    #         del checkpoint['model_state_dict'][key_id]
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    start_epoch=17
    log_string("-> loaded checkpoint %s (epoch: %d)"%(CHECKPOINT_PATH, start_epoch))
else:
    print("-> no checkpoint to load")
# Decay Batchnorm momentum from 0.5 to 0.999
# note: pytorch's BN momentum (default 0.1)= 1 - tensorflow's BN momentum
BN_MOMENTUM_INIT = 0.5
BN_MOMENTUM_MAX = 0.001
bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * cfgs.bn_decay_rate**(int(it / cfgs.bn_decay_step)), BN_MOMENTUM_MAX)
bnm_scheduler = BNMomentumScheduler(net, bn_lambda=bn_lbmd, last_epoch=start_epoch-1)


def get_current_lr(epoch):
    lr = cfgs.learning_rate
    for i,lr_decay_epoch in enumerate(LR_DECAY_STEPS):
        if epoch >= lr_decay_epoch:
            lr *= LR_DECAY_RATES[i]
    return lr

def adjust_learning_rate(optimizer, epoch):
    lr = get_current_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# TensorBoard Visualizers
TRAIN_WRITER = SummaryWriter(os.path.join(cfgs.log_dir, 'train'))
TEST_WRITER = SummaryWriter(os.path.join(cfgs.log_dir, 'test'))

# ------------------------------------------------------------------------- GLOBAL CONFIG END

def train_one_epoch(epoch):
    memstat()
    stat_dict = {} # collect statistics
    adjust_learning_rate(optimizer, EPOCH_CNT)
    bnm_scheduler.step() # decay BN momentum
    # set model to training mode
    net.train()
    for batch_idx, batch_data_label in enumerate(TRAIN_DATALOADER):
        # print("batch_idx, ", batch_idx)
        # print("batch_data_label, ", batch_data_label)
        for key in batch_data_label:
            if 'list' in key:
                for i in range(len(batch_data_label[key])):
                    for j in range(len(batch_data_label[key][i])):
                        batch_data_label[key][i][j] = batch_data_label[key][i][j].to(device)
            else:
                batch_data_label[key] = batch_data_label[key].to(device)

        # Forward pass
        end_points = net(batch_data_label)
        # print('end_points.keys():', end_points.keys())
        # print("end_points['input_xyz']:", end_points['input_xyz'])
        # print(end_points['input_xyz'].shape, end_points['input_xyz'].size) #torch.Size([2, 20000, 3])
        # print("end_points['batch_grasp_point']:", end_points['batch_grasp_point'])
        # print(end_points['batch_grasp_point'].shape, end_points['batch_grasp_point'].size)
        # tmp_end_points_np = end_points['batch_grasp_point'].detach().cpu().numpy()
        # print(tmp_end_points_np)
        # print(tmp_end_points_np.shape)
        # print(end_points['batch_grasp_point'][0][0])
        #for i in range(len(end_points['batch_grasp_point'][0])):
        #    print("i=", i)
        #    print(end_points['batch_grasp_point'][0][i].detach().cpu().numpy(), end_points['batch_grasp_point'][1][i].detach().cpu().numpy())
        # print("end_points['input_features']:", end_points['input_features']) #torch.Size([2, 1024, 3])
        
        # Compute loss and gradients, update parameters.
        loss, end_points = get_loss(end_points, epoch, batch_idx)
        print('loss in train_one_epoch:', loss, type(end_points))
        
        loss.backward()
        if (batch_idx+1) % 1 == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'prec' in key or 'recall' in key or 'count' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        batch_interval = 10
        if (batch_idx+1) % batch_interval == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx+1))
            for key in sorted(stat_dict.keys()):
                TRAIN_WRITER.add_scalar(key, stat_dict[key]/batch_interval, (EPOCH_CNT*len(TRAIN_DATALOADER)+batch_idx)*cfgs.batch_size)
                log_string('mean %s: %f'%(key, stat_dict[key]/batch_interval))
                stat_dict[key] = 0

def evaluate_one_epoch(epoch):
    stat_dict = {} # collect statistics
    # set model to eval mode (for bn and dp)
    net.eval()
    for batch_idx, batch_data_label in enumerate(TEST_DATALOADER):
        if batch_idx % 10 == 0:
            print('Eval batch: %d'%(batch_idx))
        for key in batch_data_label:
            if 'list' in key:
                for i in range(len(batch_data_label[key])):
                    for j in range(len(batch_data_label[key][i])):
                        batch_data_label[key][i][j] = batch_data_label[key][i][j].to(device)
            else:
                batch_data_label[key] = batch_data_label[key].to(device)
        
        # Forward pass
        with torch.no_grad():
            end_points = net(batch_data_label)

        # Compute loss
        loss, end_points = get_loss(end_points, epoch, batch_idx)

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'prec' in key or 'recall' in key or 'count' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

    for key in sorted(stat_dict.keys()):
        TEST_WRITER.add_scalar(key, stat_dict[key]/float(batch_idx+1), (EPOCH_CNT+1)*len(TRAIN_DATALOADER)*cfgs.batch_size)
        log_string('eval mean %s: %f'%(key, stat_dict[key]/(float(batch_idx+1))))

    mean_loss = stat_dict['loss/overall_loss']/float(batch_idx+1)
    return mean_loss


def train(start_epoch):
    t_start = datetime.now()
    memstat()
    global EPOCH_CNT 
    min_loss = 1e10
    loss = 0
    for epoch in range(start_epoch, cfgs.max_epoch):
        EPOCH_CNT = epoch
        log_string('**** EPOCH %03d ****' % (epoch))
        log_string('Current learning rate: %f'%(get_current_lr(epoch)))
        log_string('Current BN decay momentum: %f'%(bnm_scheduler.lmbd(bnm_scheduler.last_epoch)))
        log_string(str(datetime.now()))
        # Reset numpy seed.
        # REF: https://github.com/pytorch/pytorch/issues/5059
        np.random.seed()

        print("--------train_one_epoch--------")
        t1 = datetime.now()
        train_one_epoch(epoch)
        t2 = datetime.now()
        print("--------evaluate_one_epoch--------")
        loss = evaluate_one_epoch(epoch)
        t3 = datetime.now()
        print("----------->train_one_epoch time:", t2-t1)
        print("----------->evaluate_one_epoch time:", t3-t2)
        print("--------save checkpoint--------")
        # Save checkpoint
        save_dict = {'epoch': epoch+1, # after training one epoch, the start_epoch should be epoch+1
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }
        try: # with nn.DataParallel() the net is added as a submodule of DataParallel
            save_dict['model_state_dict'] = net.module.state_dict()
        except:
            save_dict['model_state_dict'] = net.state_dict()
        torch.save(save_dict, os.path.join(cfgs.log_dir, 'checkpoint.tar'))
        print("checkpoint saved: ", os.path.join(cfgs.log_dir, 'checkpoint.tar'))
    t_end = datetime.now()
    print("----------->training time:", t_end-t_start)

if __name__=='__main__':
    train(start_epoch)
