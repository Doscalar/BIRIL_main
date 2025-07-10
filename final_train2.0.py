import datetime
import os
import time
import warnings
import math
from myloader import TinyImageNet
from spikingjelly.activation_based.model.tv_ref_classify import presets, transforms, utils
import torch.utils.data
import torchvision
from spikingjelly.activation_based.model.tv_ref_classify.sampler import RASampler
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode
import random
import numpy as np
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
# from DVSGestureLoader import DVSGestureLoader
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
import MyNet
import matplotlib.pyplot as plt
import torch.utils.data as data
import sys
import argparse
import logging
from tqdm import tqdm
import torch
from spikingjelly.activation_based.model import spiking_resnet, train_classify,spiking_vgg,sew_resnet
from spikingjelly.activation_based import surrogate, neuron, functional,layer,learning
# import Data_loader
import torch.optim as optim
import utils
from torch.optim import SGD, Adam
try:
    from torchvision import prototype
except ImportError:
    prototype = None


parser = argparse.ArgumentParser()
writer = SummaryWriter('./')


def train_GS(model, optimizer, loss_fn, dataloader, metrics, params):

    print("STBP2")
    model.train()
    step_mode = 'm'
    functional.set_step_mode(model, 'm')
    summ = []
    loss_avg = utils.RunningAverage()

    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):

            if params.device:
                train_batch, labels_batch = train_batch.to(params.device), labels_batch.to(params.device)

            train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

            train_batch = train_batch.unsqueeze(0).repeat(params.T, 1, 1, 1, 1)
            # output_batch = model(train_batch)
            output_batch = model(train_batch).mean(0)
            loss = loss_fn(output_batch, labels_batch)


            optimizer.zero_grad()
            loss.backward()


            optimizer.step()
            functional.reset_net(model)


            if i % params.save_summary_steps == 0:

                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()


                summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                                 for metric in metrics}
                summary_batch['loss'] = loss.item()
                summ.append(summary_batch)


            loss_avg.update(loss.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()


    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)
    
    
    
def train_STDP(model, optimizer, loss_fn, dataloader, metrics, params):
    
    print("STDP")

    model.train()
    def f_weight( x):
            return torch.clamp(x, -1, 1.)
    tau_pre = 2.
    tau_post = 100.
    step_mode = 'm'
    functional.set_step_mode(model, 'm')
    # instances_stdp = ( layer.Linear,layer.Conv1d,layer.BatchNorm2d,layer.Conv2d,layer.MaxPool2d,) # layer.AdaptiveAvgPool2d,
    instances_stdp = (layer.Linear, layer.Conv1d, layer.BatchNorm2d, layer.Conv2d,)
    stdp_learners = []  
    
      
    summ = []
    loss_avg = utils.RunningAverage()

    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):

            if params.device:
                train_batch, labels_batch = train_batch.to(params.device), labels_batch.to(params.device)

            train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

            train_batch = train_batch.unsqueeze(0).repeat(params.T, 1, 1, 1, 1)
            
            for name, layerr in model.named_children():
                    if isinstance(layerr.__class__.__name__, instances_stdp):
                        stdp_learners.append(
                            learning.STDPLearner(step_mode=step_mode, synapse=model[i], sn=model[i + 1],
                                                tau_pre=tau_pre,
                                                tau_post=tau_post,
                                                f_pre=f_weight(train_batch), f_post=f_weight(train_batch)
                                                )
                        )
                        
            params_stdp = []
            for m in model.modules():
                if isinstance(m, instances_stdp):
                    for p in m.parameters():
                        params_stdp.append(p)

            params_stdp_set = set(params_stdp)
            params_gradient_descent = []
            for p in model.parameters():
                if p not in params_stdp_set:
                    params_gradient_descent.append(p)
            output_batch = model(train_batch).mean(0)
            # output_batch = model(train_batch)
            loss = loss_fn(output_batch, labels_batch)
            
            optimizer1 = SGD(params_stdp, lr=params.lr, momentum=0.)
            optimizer1.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            
            
            optimizer1.zero_grad()

            optimizer.step()
            optimizer1.step()
            functional.reset_net(model)
            for i in range(stdp_learners.__len__()):
                stdp_learners[i].reset()

            if i % params.save_summary_steps == 0:

                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()


                summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                                 for metric in metrics}
                summary_batch['loss'] = loss.item()
                summ.append(summary_batch)


            loss_avg.update(loss.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()


    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)
    
def train_STDP_GS(model, optimizer, loss_fn, dataloader, metrics, params):

    print("STDP_STBP2")
    
    model.train()
    def f_weight( x):
            return torch.clamp(x, -1, 1.)
    tau_pre = 2.
    tau_post = 100.
    step_mode = 'm'
    functional.set_step_mode(model, 'm')
    instances_stdp = (layer.Conv1d,layer.Conv2d, )
    stdp_learners = []  
    
      
    summ = []
    loss_avg = utils.RunningAverage()

    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):

            if params.device:
                train_batch, labels_batch = train_batch.to(params.device), labels_batch.to(params.device)

            train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

            train_batch = train_batch.unsqueeze(0).repeat(params.T, 1, 1, 1, 1)
            
            for name, layerr in model.named_children():
                    if isinstance(layerr.__class__.__name__, instances_stdp):
                        stdp_learners.append(
                            learning.STDPLearner(step_mode=step_mode, synapse=model[i], sn=model[i + 1],
                                                tau_pre=tau_pre,
                                                tau_post=tau_post,
                                                f_pre=f_weight(train_batch), f_post=f_weight(train_batch)
                                                )
                        )
                        
            params_stdp = []
            for m in model.modules():
                if isinstance(m, instances_stdp):
                    for p in m.parameters():
                        params_stdp.append(p)

            params_stdp_set = set(params_stdp)
            params_gradient_descent = []
            for p in model.parameters():
                if p not in params_stdp_set:
                    params_gradient_descent.append(p)
            output_batch = model(train_batch).mean(0)
            # output_batch = model(train_batch)
            loss = loss_fn(output_batch, labels_batch)
            
            optimizer1 = SGD(params_stdp, lr=params.lr, momentum=0.)
            optimizer1.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            # optimizer1.zero_grad()

            optimizer.step()
            
            optimizer1.zero_grad()
            
            optimizer1.step()
            functional.reset_net(model)
            for i in range(stdp_learners.__len__()):
                stdp_learners[i].reset()

            if i % params.save_summary_steps == 0:

                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()


                summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                                 for metric in metrics}
                summary_batch['loss'] = loss.item()
                summ.append(summary_batch)


            loss_avg.update(loss.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()


    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)


      

def evaluate(model, loss_fn, dataloader, metrics, params):



    model.eval()


    summ = []
    correct=0
    summ = []
    correct_sum = 0
    test_sum=0


    for data_batch, labels_batch in dataloader:


        data_batch = data_batch.unsqueeze(0).repeat(params.T, 1, 1, 1, 1)
        data_batch, labels_batch = data_batch.to(params.device), labels_batch.to(params.device)

        data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)
        
        
        
        
        
        # output_batch = model(data_batch)
        output_batch = model(data_batch).mean(0)
        correct_sum += (output_batch.max(1)[1] == labels_batch.to(params.device)).float().sum().item()
        test_sum +=  labels_batch.numel()
        functional.reset_net(model)

    print('bbb')
    test_accuracy = correct_sum / test_sum
        
        
    torch.save({
                
                'state_dict': model.state_dict(),
                
            }, os.path.join('./', 'rescifar100.pth.tar'))
            #print('saved')
    print(test_accuracy)
    return test_accuracy
def evaluate_STDP(model, loss_fn, dataloader, metrics, params):



    model.eval()


    summ = []
    correct=0
    summ = []
    correct_sum = 0
    test_sum=0


    for data_batch, labels_batch in dataloader:


        data_batch = data_batch.unsqueeze(0).repeat(params.T, 1, 1, 1, 1)
        data_batch, labels_batch = data_batch.to(params.device), labels_batch.to(params.device)

        data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)
        
        
        
        
        
        # output_batch = model(data_batch)
        output_batch = model(data_batch).mean(0)
        correct_sum += (output_batch.max(1)[1] == labels_batch.to(params.device)).float().sum().item()
        test_sum +=  labels_batch.numel()
        functional.reset_net(model)

    print('bbb')
    test_accuracy = correct_sum / test_sum
        
        
    torch.save({
                
                'state_dict': model.state_dict(),
                
            }, os.path.join('./', 'rescifar100.pth.tar'))
            #print('saved')
    print(test_accuracy)
    return test_accuracy

def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer,
                       loss_fn,  metrics,params, model_dir, restore_file=None):

    if restore_file is not None:
        restore_path = os.path.join(params.model_dir, params.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        # utils.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0

    val_accuracies = []

    # scheduler = StepLR(optimizer, step_size=150, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=320)


    writer.flush()
    max=0.0

    for epoch in range(params.epochs):
        scheduler.step()


        logging.info("Epoch {}/{}".format(epoch + 1, params.epochs))

        if params.TrainModel == 'GS':
                 train_GS(model, optimizer, loss_fn, train_dataloader, metrics, params)
        elif params.TrainModel == 'STDP':
             train_STDP(model, optimizer, loss_fn, train_dataloader, metrics, params)
        elif params.TrainModel == 'STDP_GS':
             train_STDP_GS(model, optimizer, loss_fn, train_dataloader, metrics, params)
        elif params.TrainModel == 'GS_STDP_Iteration':
            if params.K == 2:
                if epoch % params.K  == 0 :
                    train_GS(model, optimizer, loss_fn, train_dataloader, metrics, params)
                elif epoch % params.K  == 1 :
                    train_STDP(model, optimizer, loss_fn, train_dataloader, metrics, params)
            if params.K == 3:
                if epoch % params.K  == 1 :
                    train_STDP(model, optimizer, loss_fn, train_dataloader, metrics, params)
                elif epoch % params.K  == 0 or epoch % params.K ==2 :
                    train_GS(model, optimizer, loss_fn, train_dataloader, metrics, params)  
            if params.K == 4:
                if epoch % params.K  == 1 :
                    train_STDP(model, optimizer, loss_fn, train_dataloader, metrics, params)
                elif epoch % params.K  == 0 or epoch % params.K ==2 or epoch % params.K ==3:
                    train_GS(model, optimizer, loss_fn, train_dataloader, metrics, params) 
            if params.K == 5:
                if epoch % params.K  == 0 :
                    train_STDP(model, optimizer, loss_fn, train_dataloader, metrics, params)
                elif epoch % params.K  == 1 or epoch % params.K ==2 or epoch % params.K ==3 or epoch % params.K ==4:
                    train_GS(model, optimizer, loss_fn, train_dataloader, metrics, params)
            if params.K == 6:
                if epoch % params.K  == 0 or epoch % params.K  == 1 or epoch % params.K ==3 or epoch % params.K ==4:
                    train_GS(model, optimizer, loss_fn, train_dataloader, metrics, params)
                elif epoch % params.K  == 2:
                    train_STDP(model, optimizer, loss_fn, train_dataloader, metrics, params)
                elif epoch % params.K  == 5:
                    train_STDP_GS(model, optimizer, loss_fn, train_dataloader, metrics, params)
            if params.K == 9:
                if epoch % params.K  == 0 or epoch % params.K  == 1 or epoch % params.K ==3 or epoch % params.K ==4 or epoch % params.K ==6 or epoch % params.K ==7: 
                    train_GS(model, optimizer, loss_fn, train_dataloader, metrics, params)
                elif epoch % params.K  == 2 or epoch % params.K  == 5:
                    train_STDP(model, optimizer, loss_fn, train_dataloader, metrics, params)
                elif epoch % params.K  == 8:
                    train_STDP_GS(model, optimizer, loss_fn, train_dataloader, metrics, params)
            if params.K == -3:
                if epoch % abs(params.K)  == 0 :
                    train_GS(model, optimizer, loss_fn, train_dataloader, metrics, params)
                elif epoch % abs(params.K)  == 1 or epoch % abs(params.K) ==2 :
                    train_STDP(model, optimizer, loss_fn, train_dataloader, metrics, params)
            if params.K == -4:
                if epoch % abs(params.K)  == 0 :
                    train_GS(model, optimizer, loss_fn, train_dataloader, metrics, params)
                elif epoch % abs(params.K)  == 1 or epoch % abs(params.K) ==2 or epoch % abs(params.K) ==3 :
                    train_STDP(model, optimizer, loss_fn, train_dataloader, metrics, params)
            if params.K == -5:
                if epoch % abs(params.K)  == 0 :
                    train_GS(model, optimizer, loss_fn, train_dataloader, metrics, params)
                elif epoch % abs(params.K)  == 1 or epoch % abs(params.K) ==2 or epoch % abs(params.K) ==3 or epoch % abs(params.K) ==4:
                    train_STDP(model, optimizer, loss_fn, train_dataloader, metrics, params)        
                     
        elif params.TrainModel == 'GS_STDP_GS_Iteration':
            if params.K == 2:
                if epoch % params.K  == 0 :
                    train_GS(model, optimizer, loss_fn, train_dataloader, metrics, params)
                elif epoch % params.K  == 1 :
                    train_STDP_GS(model, optimizer, loss_fn, train_dataloader, metrics, params)
            if params.K == 3:
                if epoch % params.K  == 1 :
                    train_STDP_GS(model, optimizer, loss_fn, train_dataloader, metrics, params)
                elif epoch % params.K  == 0 or epoch % params.K ==2 :
                    train_GS(model, optimizer, loss_fn, train_dataloader, metrics, params)  
            if params.K == 4:
                if epoch % params.K  == 1 :
                    train_STDP_GS(model, optimizer, loss_fn, train_dataloader, metrics, params)
                elif epoch % params.K  == 0 or epoch % params.K ==2 or epoch % params.K ==3:
                    train_GS(model, optimizer, loss_fn, train_dataloader, metrics, params) 
            if params.K == 5:
                if epoch % params.K  == 1 :
                    train_STDP_GS(model, optimizer, loss_fn, train_dataloader, metrics, params)
                elif epoch % params.K  == 0 or epoch % params.K ==2 or epoch % params.K ==3 or epoch % params.K ==4:
                    train_GS(model, optimizer, loss_fn, train_dataloader, metrics, params)
            if params.K == -3:
                if epoch % abs(params.K)  == 0 :
                    train_GS(model, optimizer, loss_fn, train_dataloader, metrics, params)
                elif epoch % abs(params.K)  == 1 or epoch % abs(params.K) ==2 :
                    train_STDP_GS(model, optimizer, loss_fn, train_dataloader, metrics, params)
            if params.K == -4:
                if epoch % abs(params.K)  == 0 :
                    train_GS(model, optimizer, loss_fn, train_dataloader, metrics, params)
                elif epoch % abs(params.K)  == 1 or epoch % abs(params.K) ==2 or epoch % abs(params.K) ==3 :
                    train_STDP_GS(model, optimizer, loss_fn, train_dataloader, metrics, params)
            if params.K == -5:
                if epoch % abs(params.K)  == 0 :
                    train_GS(model, optimizer, loss_fn, train_dataloader, metrics, params)
                elif epoch % abs(params.K)  == 1 or epoch % abs(params.K) ==2 or epoch % abs(params.K) ==3 or epoch % abs(params.K) ==4:
                    train_STDP_GS(model, optimizer, loss_fn, train_dataloader, metrics, params)        
                    
       
            
        # train(model, optimizer, loss_fn, train_dataloader, metrics, params)
        if params.TrainModel == 'GS':
            val_acc = evaluate(model, loss_fn, val_dataloader,  metrics,params)
        else:    
            if "Iteration" in  params.TrainModel :
            
                if epoch  == 0:
                    val_acc = evaluate(model, loss_fn, val_dataloader,  metrics,params)
                else :
                    val_acc = evaluate_STDP(model, loss_fn, val_dataloader,  metrics,params)
            else:
                val_acc = evaluate_STDP(model, loss_fn, val_dataloader,  metrics,params)
        
        print(epoch)
        print(val_acc)
        
        val_accuracies.append(val_acc)  # 记录验证精度

        if val_acc > max:
            max = val_acc
            torch.save({
                
                'state_dict': model.state_dict(),
                
            }, os.path.join('./', 'bestmodel1.pth.tar'))
        print(max)

        writer.add_scalar('test_accuracy', val_acc, epoch)
    if params.ME == '1':
        new_model = evaluate_model()
        final_acc = evaluate(new_model, loss_fn, val_dataloader,  metrics,params)
        print(final_acc)
    
    # 绘制验证精度图
    plt.figure(figsize=(8, 5))
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy of CIFAR100')
    plt.legend()

    # 保存图形到文件
    plt.savefig('./CIFAR100_validation_accuracy.png')
    plt.show()
    
    

def evaluate_model(self , args):
        model1_params = torch.load(args.model1pth)  # 加载第一个模型的参数
        model2_params = torch.load(args.model2pth)  # 加载第二个模型的参数

        weighted_params = torch.nn.Parameter(torch.zeros_like(model1_params))

        factor = args.factor
        factor2 = 1 - factor  # 给 model2 的参数分配剩余权重

        for name in model1_params:
            weighted_params.data[name].copy_(factor * model1_params[name] + factor2 * model2_params[name])

        new_model = self.load_model()
        new_model.load_state_dict(weighted_params)
        return new_model


def accuracy(outputs, labels):

    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs==labels)/float(labels.size)

def get_args_parser(add_help=True):

        # parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)
        parser.add_argument("--datapath", default='./', type=str,help="dataset path")
        parser.add_argument("--TrainModel", default="STDP_GS", type=str,
                            help="train model :GS是梯度替代，STDP是STDP，STDP_GS是梯度替代和STDP的结合,STBP是STBP\
                                GS_STDP_Iteration是GS和STDP迭代训练，GS_STDP_GS_Iteration是GS和GS_STDP迭代训练，STBP_STDP_Iteration是STBP和STDP迭代训练，STBP_STDP_GS_Iteration")
        parser.add_argument("--model", default="spiking_resnet18", type=str, help="model name")
        parser.add_argument("--K", default=6, type=int, help="GS和STDP迭代的轮数，2为每次迭代，3为一次STDP2次GS，-3为一次GS两次STDP")
        parser.add_argument("--data", default="CIFAR100", type=str, help="data name，CIFAR10，CIFAR100，MINIST，DVSGeserature，ImageNet")
        parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
        parser.add_argument('--T', default=4,type=int, help="total time-steps")
        parser.add_argument("--epochs", default=350, type=int, metavar="N", help="number of total epochs to run")
        parser.add_argument("--lr", default=0.001, type=float, help="initial learning rate")
        
        
        
        parser.add_argument("--opt1", default="sgd", type=str, help="optimizer")
        parser.add_argument("--opt2", default="adam", type=str, help="optimizer")
        parser.add_argument("--output-dir", default="./logs", type=str, help="path to save outputs")
        parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
        parser.add_argument("--model1pth", default="resnet18", type=str, help="模型1的保存路径")
        parser.add_argument("--model2pth", default="resnet18", type=str, help="模型2的保存路径")
        parser.add_argument("--ME", default="0", type=str, help="是否将两个模型合并结果，0是不要，1是要")
        parser.add_argument("--factor", default=0.5, type=float, help="将两个模型加权的比例")
        parser.add_argument('--model_dir', default='experiments/base_model', help="Directory of params.json")
        parser.add_argument('--restore_file', default='bestmodel1', help="name of the file in --model_dir \
                     containing weights to load")
        parser.add_argument('--save_summary_steps', default=100, type=int)
        parser.add_argument("--EvaluateModel", default="STDP_GS", type=str,
                            help="evaluate model :GS是梯度替代单独评估，STDP是STDP单独评估，STDP_GS是梯度替代和STDP的结合评估,\
                            STBP是STBP单独评估，GS_STDP_GS是梯度替代STDP梯度替代结合，STBP_STDP_GS是这三者结合评估")
        parser.add_argument("-b", "--batch-size", default=64, type=int,help="images per gpu, the total batch size is $NGPU x batch_size")
        
        
        parser.add_argument('--cupy', action="store_true", help="set the neurons to use cupy backend")
        parser.add_argument(
            "-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 16)"
        )
        parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
        parser.add_argument(
            "--wd",
            "--weight-decay",
            default=0.,
            type=float,
            metavar="W",
            help="weight decay (default: 0.)",
            dest="weight_decay",
        )
        parser.add_argument(
            "--norm-weight-decay",
            default=None,
            type=float,
            help="weight decay for Normalization layers (default: None, same value as --wd)",
        )
        parser.add_argument(
            "--label-smoothing", default=0.1, type=float, help="label smoothing (default: 0.1)", dest="label_smoothing"
        )
        parser.add_argument("--mixup-alpha", default=0.2, type=float, help="mixup alpha (default: 0.2)")
        parser.add_argument("--cutmix-alpha", default=1.0, type=float, help="cutmix alpha (default: 1.0)")
        parser.add_argument("--lr-scheduler", default="cosa", type=str, help="the lr scheduler (default: cosa)")
        parser.add_argument("--lr-warmup-epochs", default=5, type=int,
                            help="the number of epochs to warmup (default: 5)")
        parser.add_argument(
            "--lr-warmup-method", default="linear", type=str, help="the warmup method (default: linear)"
        )
        parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
        parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
        parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
        parser.add_argument("--resume", default=None, type=str,
                            help="path of checkpoint. If set to 'latest', it will try to load the latest checkpoint")
        parser.add_argument(
            "--cache-dataset",
            dest="cache_dataset",
            help="Cache the datasets for quicker initialization. It also serializes the transforms",
            action="store_true",
        )
        parser.add_argument(
            "--sync-bn",
            dest="sync_bn",
            help="Use sync batch norm",
            action="store_true",
        )
        parser.add_argument(
            "--test-only",
            dest="test_only",
            help="Only test the model",
            action="store_true",
        )
        parser.add_argument(
            "--pretrained",
            dest="pretrained",
            help="Use pre-trained models from the modelzoo",
            action="store_true",
        )
        parser.add_argument("--auto-augment", default='ta_wide', type=str,
                            help="auto augment policy (default: ta_wide)")
        parser.add_argument("--random-erase", default=0.1, type=float, help="random erasing probability (default: 0.1)")

        # Mixed precision training parameters

        # distributed training parameters
        parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
        parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
        parser.add_argument(
            "--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters"
        )
        parser.add_argument(
            "--model-ema-steps",
            type=int,
            default=32,
            help="the number of iterations that controls how often to update the EMA model (default: 32)",
        )
        parser.add_argument(
            "--model-ema-decay",
            type=float,
            default=0.99998,
            help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",
        )
        parser.add_argument(
            "--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)"
        )
        parser.add_argument(
            "--val-resize-size", default=232, type=int, help="the resize size used for validation (default: 232)"
        )
        parser.add_argument(
            "--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)"
        )
        parser.add_argument(
            "--train-crop-size", default=176, type=int, help="the random crop size used for training (default: 176)"
        )
        parser.add_argument("--clip-grad-norm", default=None, type=float,
                            help="the maximum gradient norm (default None)")
        parser.add_argument("--ra-sampler", action="store_true",
                            help="whether to use Repeated Augmentation in training")
        parser.add_argument(
            "--ra-reps", default=4, type=int, help="number of repetitions for Repeated Augmentation (default: 4)"
        )

        # Prototype models only
        parser.add_argument(
            "--prototype",
            dest="prototype",
            help="Use prototype model builders instead those from main area",
            action="store_true",
        )
        parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
        parser.add_argument("--seed", default=2020, type=int, help="the random seed")

        parser.add_argument("--print-logdir", action="store_true",
                            help="print the dirs for tensorboard logs and pt files and exit")
        parser.add_argument("--clean", action="store_true", help="delete the dirs for tensorboard logs and pt files")
        parser.add_argument("--disable-pinmemory", action="store_true",
                            help="not use pin memory in dataloader, which can help reduce memory consumption")
        parser.add_argument("--disable-amp", action="store_true",
                            help="not use automatic mixed precision training")
        parser.add_argument("--local_rank", type=int, help="args for DDP, which should not be set by user")
        parser.add_argument("--disable-uda", action="store_true",
                            help="not set 'torch.use_deterministic_algorithms(True)', which can avoid the error raised by some functions that do not have a deterministic implementation")
        args = parser.parse_args()
        return args
    
if __name__ == '__main__':

    device = 'cuda:0'
    params = get_args_parser()
    random.seed(230)
    torch.manual_seed(230)
    if params.device: torch.cuda.manual_seed(230)
    logging.info("Loading the datasets...")

    if params.data == 'CIFAR10':
        print(1)
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        cifar10_training = torchvision.datasets.CIFAR10(root='./',
                                                        train=True, download=True,
                                                        transform=transform_train)
        train_dl = torch.utils.data.DataLoader(cifar10_training, batch_size=64, shuffle=True, drop_last=True)

        cifar10_testing = torchvision.datasets.CIFAR10(root='./',
                                                        train=False, download=True,
                                                        transform=transform_test)
        dev_dl = torch.utils.data.DataLoader(cifar10_testing, batch_size=64, shuffle=False, drop_last=True)
        num_class = 10
    elif params.data == 'CIFAR100':
        print(2)
        mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
        std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        transform_test = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        cifar100_training = torchvision.datasets.CIFAR100(root='./',
                                                        train=True, download=True,
                                                        transform=transform_train)
        train_dl = torch.utils.data.DataLoader(cifar100_training, batch_size=64, shuffle=True, drop_last=True)

        cifar100_testing = torchvision.datasets.CIFAR100(root='./',
                                                        train=False, download=True,
                                                        transform=transform_test)
        dev_dl = torch.utils.data.DataLoader(cifar100_testing, batch_size=64, shuffle=False, drop_last=True)
        num_class = 100
    elif params.data == 'MINIST':
        mean = [0.1307]
        std = [0.3081]
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        mnist_training = torchvision.datasets.MNIST(root='./',
                                            train=True, download=True,
                                            transform=transform_train)
        train_dl = torch.utils.data.DataLoader(mnist_training, batch_size=64, shuffle=True, drop_last=True)

        mnist_testing = torchvision.datasets.MNIST(root='./',
                                           train=False, download=True,
                                           transform=transform_test)
        dev_dl = torch.utils.data.DataLoader(mnist_testing, batch_size=64, shuffle=False, drop_last=True)
        num_class = 10
    elif params.data == 'DVSGesterature':
        train_set = DVS128Gesture(root='./gdata', train=True, data_type='frame', frames_number=16, split_by='number')
        test_set = DVS128Gesture(root='./gdata', train=False, data_type='frame', frames_number=16, split_by='number')
        train_dl= torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=4,
            shuffle=True,
            num_workers=1,
            drop_last=True
            )

        dev_dl = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=4,
            shuffle=False,
            num_workers=1,
            drop_last=False
            )
        
        num_class = 11
    elif params.data == 'ImageNet':
       # ImageNet mean and standard deviation
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # Train data transformation
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        # Test data transformation
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        data_dir = params.datapath# 设置数据集的路径

        #train_dataset = torchvision.datasets.ImageFolder(data_dir, transform_train)
        #test_dataset = torchvision.datasets.ImageFolder(data_dir, transform_test)
        # train_size = int(0.8 * len(image_dataset))
        # test_size = len(image_dataset) - train_size
        # train_dataset, test_dataset = torch.utils.data.random_split(image_dataset, [train_size, test_size])
        
        
        train_dataset = TinyImageNet(data_dir, train=True,transform=transform_train)
        test_dataset = TinyImageNet(data_dir, train=False,transform=transform_test)
         
        
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8)
        num_class = 200

    logging.info("- done.")
    # print(params.model)
    if params.model == "spiking_resnet18":
        model = spiking_resnet.spiking_resnet18().to(device)
        print('spiking_resnet')
        print(params.lr)
        optimizer = optim.Adam(model.parameters(), lr=params.lr, weight_decay=0.)
        loss_fn = nn.CrossEntropyLoss()
        metrics = {'accuracy': accuracy,
                                 # could add more metrics such as accuracy for each token type
                                 }
    elif params.model == "spiking_vgg11":
        model = spiking_vgg.spiking_vgg11().to(device)
        # model = text_model.to(device) # ,num_classes= 100 ,timesteps = params.T
        print('spiking_vgg')
        print(params.lr)
        optimizer = optim.Adam(model.parameters(), lr=params.lr, weight_decay=0)
        loss_fn = nn.CrossEntropyLoss()
        metrics = {'accuracy': accuracy,
                                 # could add more metrics such as accuracy for each token type
                                 }
    elif params.model == "MyResnet":
        model = MyNet.spiking_resnet18().to(device)
        print('MyNet')
        print(params.learning_rate)
        optimizer = optim.Adam(model.parameters(), lr=params.lr, weight_decay=0)
        loss_fn = nn.CrossEntropyLoss()
        metrics = {'accuracy': accuracy,
                                 # could add more metrics such as accuracy for each token type
                                 }
    logging.info("Starting training for {} epoch(s)".format(params.epochs))   
    train_and_evaluate(model, train_dl, dev_dl, optimizer, loss_fn,  metrics,params,
                           params.model_dir, params.restore_file)





