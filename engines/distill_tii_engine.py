import math
import sys
import os
import datetime
import json
from typing import Iterable
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np

from timm.utils import accuracy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from torch import optim
import utils
from torch.distributions.multivariate_normal import MultivariateNormal


def train_and_evaluate(model: torch.nn.Module, model_without_ddp: torch.nn.Module,
                       criterion, data_loader: Iterable, data_loader_per_cls: Iterable,
                       optimizer: torch.optim.Optimizer, lr_scheduler,
                       device: torch.device,
                       class_mask=None, target_task_map=None, args=None, ):
    # create matrix to save end-of-task accuracies
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    pre_ca_acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    uncertainty_acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    task_gaussian_acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    global cls_mean
    global cls_cov
    global old_head
    cls_mean = dict()
    cls_cov = dict()

    args.uncertain_lr = args.uncertain_lr * args.batch_size / 256.0

    for task_id in range(args.num_tasks):

        # Create new optimizer for each task to clear optimizer status
        if task_id > 0 and args.reinit_optimizer:
            optimizer = create_optimizer(args, model)
            if args.sched != 'constant':
                lr_scheduler, _ = create_scheduler(args, optimizer)
            elif args.sched == 'constant':
                lr_scheduler = None

        if task_id > 0:
            old_head = model.get_head()

        for epoch in range(args.epochs):
            # Train model
            train_stats = train_one_epoch(model=model, criterion=criterion,
                                        data_loader=data_loader[task_id]['train'], optimizer=optimizer,
                                        device=device, epoch=epoch, max_norm=args.clip_grad,
                                        set_training_mode=True, task_id=task_id, class_mask=class_mask, args=args,
                                        )

            if lr_scheduler:
                lr_scheduler.step(epoch)

        print('-' * 20)
        print(f'Evaluate task {task_id + 1} before CA')
        test_stats_pre_ca = evaluate_till_now(model=model, data_loader=data_loader,
                                              device=device,
                                              task_id=task_id, class_mask=class_mask, target_task_map=target_task_map,
                                              acc_matrix=pre_ca_acc_matrix, args=args)
        print('-' * 20)

        # TODO compute mean and variance
        print('-' * 20)
        print(f'Compute mean and variance for task {task_id + 1}')
        _compute_mean(model=model, data_loader=data_loader_per_cls, device=device, class_mask=class_mask[task_id],
                      args=args)
        print('-' * 20)

        # TODO classifier alignment
        if task_id > 0:
            print('-' * 20)
            print(f'Align classifier for task {task_id + 1}')
            # train_task_adaptive_prediction(model, args, device, class_mask, task_id,)
            train_task_adaptive(model, args, device, class_mask, task_id, data_loader[task_id]['train'])
            print('-' * 20)
            if args.uncertain:
                print('Evaluate task {} before uncertain training'.format(task_id + 1))
                evaluate_till_now(model=model, data_loader=data_loader,
                                device=device,
                                task_id=task_id, class_mask=class_mask,
                                target_task_map=target_task_map,
                                acc_matrix=uncertainty_acc_matrix, args=args)
                print('-' * 20)
                
                if args.adapt_prior:
                    print("Adapt new prior")
                    train_adapt_prior(model, args, device, class_mask, task_id, data_loader[task_id]['train'])
                    print('-' * 20)
                    print('Evaluate task {} after adapt prior'.format(task_id + 1))
                    evaluate_till_now(model=model, data_loader=data_loader,
                                    device=device,
                                    task_id=task_id, class_mask=class_mask,
                                    target_task_map=target_task_map,
                                    acc_matrix=task_gaussian_acc_matrix, args=args)
                    print('-' * 20)

                print("Gaussian training")
                gaussian_train(model, args, device, class_mask, task_id)

        # Evaluate model
        print('-' * 20)
        print(f'Evaluate task {task_id + 1} final')
        test_stats = evaluate_till_now(model=model, data_loader=data_loader,
                                       device=device,
                                       task_id=task_id, class_mask=class_mask, target_task_map=target_task_map,
                                       acc_matrix=acc_matrix, args=args)
        print('-' * 20)

        if args.output_dir and utils.is_main_process():
            Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)

            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))
            state_dict = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args,
            }
            if args.sched is not None and args.sched != 'constant':
                state_dict['lr_scheduler'] = lr_scheduler.state_dict()

            utils.save_on_master(state_dict, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch, }

        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir,
                                   '{}_stats.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))),
                      'a') as f:
                f.write(json.dumps(log_stats) + '\n')


@torch.no_grad()
def evaluate(model: torch.nn.Module, data_loader,
             device, task_id=-1, class_mask=None, target_task_map=None, args=None, ):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f'Test: Task[{task_id + 1}]'

    model.eval()
    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(input)
            logits = output['logits']

            # here is the trick to mask out classes of non-current tasks
            if args.task_inc and class_mask is not None:
                mask = class_mask[task_id]
                mask = torch.tensor(mask, dtype=torch.int64).to(device)
                logits_mask = torch.ones_like(logits, device=device) * float('-inf')
                logits_mask = logits_mask.index_fill(1, mask, 0)
                logits = logits + logits_mask

            loss = criterion(logits, target)
            
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            metric_logger.update(Loss=loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

            task_id_preds = torch.max(logits, dim=1)[1]
            task_id_preds = torch.tensor([target_task_map[v.item()] for v in task_id_preds]).to(device)
            batch_size = input.shape[0]
            tii_acc = torch.sum(task_id_preds == task_id) / batch_size
            metric_logger.meters['TII Acc'].update(tii_acc.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        '* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f} TII Acc {tii_acc.global_avg:.3f}'
        .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'],
                losses=metric_logger.meters['Loss'], tii_acc=metric_logger.meters['TII Acc']))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_till_now(model: torch.nn.Module, data_loader,
                      device, task_id=-1, class_mask=None, target_task_map=None, acc_matrix=None, args=None, ):
    stat_matrix = np.zeros((3, args.num_tasks))  # 3 for Acc@1, Acc@5, Loss
    tii_acc_matrix = np.zeros((args.num_tasks, ))

    for i in range(task_id + 1):
        test_stats = evaluate(model=model, data_loader=data_loader[i]['val'],
                              device=device, task_id=i, class_mask=class_mask, target_task_map=target_task_map,
                              args=args)

        stat_matrix[0, i] = test_stats['Acc@1']
        stat_matrix[1, i] = test_stats['Acc@5']
        stat_matrix[2, i] = test_stats['Loss']

        acc_matrix[i, task_id] = test_stats['Acc@1']
        tii_acc_matrix[i] = test_stats['TII Acc']

    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id + 1)
    avg_tii_acc = np.divide(np.sum(tii_acc_matrix), task_id + 1)
    diagonal = np.diag(acc_matrix)

    result_str = "[Average accuracy till task{}]\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}\tTII Acc: {:.4f}".format(
        task_id + 1,
        avg_stat[0],
        avg_stat[1],
        avg_stat[2],
        avg_tii_acc)
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                              acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
    print(result_str)

    return test_stats


@torch.no_grad()
def _compute_mean(model: torch.nn.Module, data_loader: Iterable, device: torch.device, class_mask=None, args=None, ):
    model.eval()

    for cls_id in class_mask:
        data_loader_cls = data_loader[cls_id]['train']
        features_per_cls = []
        for i, (inputs, targets) in enumerate(data_loader_cls):
            inputs = inputs.to(device, non_blocking=True)
            features = model(inputs)['pre_logits']
            features_per_cls.append(features)
        features_per_cls = torch.cat(features_per_cls, dim=0)
        features_per_cls_list = [torch.zeros_like(features_per_cls, device=device) for _ in range(args.world_size)]

        if args.distributed:
            dist.barrier()
            dist.all_gather(features_per_cls_list, features_per_cls)

        if args.ca_storage_efficient_method == 'covariance':
            # features_per_cls = torch.cat(features_per_cls_list, dim=0)
            # print(features_per_cls.shape)
            cls_mean[cls_id] = features_per_cls.mean(dim=0)
            cls_cov[cls_id] = torch.cov(features_per_cls.T) + (torch.eye(cls_mean[cls_id].shape[-1]) * 1e-4).to(device)
        if args.ca_storage_efficient_method == 'variance':
            # features_per_cls = torch.cat(features_per_cls_list, dim=0)
            # print(features_per_cls.shape)
            cls_mean[cls_id] = features_per_cls.mean(dim=0)
            cls_cov[cls_id] = torch.diag(torch.cov(features_per_cls.T) + (torch.eye(cls_mean[cls_id].shape[-1]) * 1e-4).to(device))
        if args.ca_storage_efficient_method == 'multi-centroid':
            from sklearn.cluster import KMeans
            n_clusters = args.n_centroids
            # features_per_cls = torch.cat(features_per_cls_list, dim=0).cpu().numpy()
            features_per_cls = features_per_cls.cpu().numpy()
            kmeans = KMeans(n_clusters=n_clusters, n_init='auto')
            kmeans.fit(features_per_cls)
            cluster_lables = kmeans.labels_
            cluster_means = []
            cluster_vars = []
            for i in range(n_clusters):
               cluster_data = features_per_cls[cluster_lables == i]
               cluster_mean = torch.tensor(np.mean(cluster_data, axis=0), dtype=torch.float64).to(device)
               cluster_var = torch.tensor(np.var(cluster_data, axis=0), dtype=torch.float64).to(device)
               cluster_means.append(cluster_mean)
               cluster_vars.append(cluster_var)
            
            cls_mean[cls_id] = cluster_means
            cls_cov[cls_id] = cluster_vars


# def train_task_adaptive_prediction(model: torch.nn.Module, args, device, class_mask=None, task_id=-1):
#     model.train()
#     run_epochs = args.crct_epochs
#     param_list = [p for p in model.parameters() if p.requires_grad]
#     print('-' * 20)
#     print('Learnable parameters:')
#     for name, p in model.named_parameters():
#         if p.requires_grad:
#             print(name)
#     print('-' * 20)
#     network_params = [{'params': param_list, 'lr': args.ca_lr, 'weight_decay': args.weight_decay}]
#     if 'mae' in args.model or 'beit' in args.model:
#         optimizer = optim.AdamW(network_params, lr=args.ca_lr / 10, weight_decay=args.weight_decay)
#     else:
#         optimizer = optim.SGD(network_params, lr=args.ca_lr, momentum=0.9, weight_decay=5e-4)

#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)
#     criterion = torch.nn.CrossEntropyLoss().to(device)


#     # TODO: efficiency may be improved by encapsulating sampled data into Datasets class and using distributed sampler.
#     for epoch in range(run_epochs):
            
#         metric_logger = utils.MetricLogger(delimiter="  ")
#         metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#         metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

#         sampled_data, sampled_label = sample_data(task_id, class_mask, device, args, include_current_task=True, train=False)
#         inputs = sampled_data
#         targets = sampled_label

#         sf_indexes = torch.randperm(inputs.size(0))
#         inputs = inputs[sf_indexes]
#         targets = targets[sf_indexes]
#         #print(targets)

#         for pos in range(0, inputs.size(0), args.batch_size):
#             inp = inputs[pos:pos + args.batch_size]
#             tgt = targets[pos:pos + args.batch_size]
#             outputs = model(inp, fc_only=True)
#             logits = outputs['logits']

#             if args.train_mask and class_mask is not None:
#                 mask = []
#                 for id in range(task_id+1):
#                     mask.extend(class_mask[id])
#                 # print(mask)
#                 not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
#                 not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
#                 logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

#             loss = criterion(logits, tgt)

#             acc1, acc5 = accuracy(logits, tgt, topk=(1, 5))

#             if not math.isfinite(loss.item()):
#                 print("Loss is {}, stopping training".format(loss.item()))
#                 sys.exit(1)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             torch.cuda.synchronize()

#             metric_logger.update(Loss=loss.item())
#             metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
#             metric_logger.meters['Acc@1'].update(acc1.item(), n=inp.shape[0])
#             metric_logger.meters['Acc@5'].update(acc5.item(), n=inp.shape[0])

#         # gather the stats from all processes
#         metric_logger.synchronize_between_processes()
#         print("Averaged stats:", metric_logger)
#         scheduler.step()


@torch.no_grad()
def compute_confusion_matrix(model: torch.nn.Module, data_loader,
                             device, target_task_map=None, args=None, ):
    confusion_matrix = np.zeros((args.num_tasks, args.num_tasks))
    model.eval()

    for i in range(args.num_tasks):

        with torch.no_grad():
            for input, target in data_loader[i]['val']:
                input = input.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                output = model(input)
                logits = output['logits']

                task_id_preds = torch.max(logits, dim=1)[1]
                task_id_preds = torch.tensor([target_task_map[v.item()] for v in task_id_preds]).to(device)
                for j in range(len(task_id_preds)):
                    confusion_matrix[i, task_id_preds[j]] += 1

    # Plot confusion matrix
    from matplotlib import pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(10, 10))
    sns.heatmap(confusion_matrix, annot=True, fmt='g', cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'))

    print(f'TII Acc: {np.trace(confusion_matrix) / np.sum(confusion_matrix)}')

    return confusion_matrix


def sample_data(task_id, class_mask, device, args, include_current_task=True, train=False):
    sampled_data = []
    sampled_label = []
    if train:
        num_sampled_pcls = int(args.batch_size / args.nb_classes * args.num_tasks)
    else:
        num_sampled_pcls = args.batch_size
    if include_current_task:
        max_task = task_id + 1
    else:
        max_task = task_id

    if args.ca_storage_efficient_method in ['covariance', 'variance']:
        for i in range(max_task):
            for c_id in class_mask[i]:
                mean = torch.tensor(cls_mean[c_id], dtype=torch.float64).to(device)
                cov = cls_cov[c_id].to(device)
                if args.ca_storage_efficient_method == 'variance':
                    cov = torch.diag(cov)
                m = MultivariateNormal(mean.float(), cov.float())
                sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                sampled_data.append(sampled_data_single)

                sampled_label.extend([c_id] * num_sampled_pcls)

    elif args.ca_storage_efficient_method == 'multi-centroid':
        num_sampled_pcls = num_sampled_pcls // args.n_centroids
        for i in range(max_task):
            for c_id in class_mask[i]:
                for cluster in range(len(cls_mean[c_id])):
                    mean = cls_mean[c_id][cluster]
                    var = cls_cov[c_id][cluster]
                    if var.mean() == 0:
                        continue
                    m = MultivariateNormal(mean.float(), (torch.diag(var) + 1e-4 * torch.eye(mean.shape[0]).to(mean.device)).float())
                    sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                    sampled_data.append(sampled_data_single)
                    sampled_label.extend([c_id] * num_sampled_pcls)

    else:
        raise NotImplementedError


    sampled_data = torch.cat(sampled_data, dim=0).float().to(device)
    sampled_label = torch.tensor(sampled_label).long().to(device)

    return sampled_data, sampled_label



def train_one_epoch(model: torch.nn.Module, criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, args=None
                    ):
    model.train(set_training_mode)

    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Train: Epoch[{epoch + 1:{int(math.log10(args.epochs)) + 1}}/{args.epochs}]'

    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(input)
        logits = output['logits']

        # here is the trick to mask out classes of non-current tasks
        if args.train_mask and class_mask is not None:
            mask = class_mask[task_id]
            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
        loss = criterion(logits, target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        torch.cuda.synchronize()
        metric_logger.update(Loss=loss.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

    
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def gaussian_train(model: torch.nn.Module, args, device, class_mask=None, task_id=-1):
    prior_head = model.get_head()

    model.train()
    run_epochs = args.uncertain_epochs
    param_list = [p for p in model.parameters() if p.requires_grad]
    network_params = [{'params': param_list, 'lr': args.uncertain_lr, 'weight_decay': args.weight_decay}]
    if 'mae' in args.model or 'beit' in args.model:
        optimizer = optim.AdamW(network_params, lr=args.uncertain_lr / 10, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(network_params, lr=args.uncertain_lr, momentum=0.9, weight_decay=5e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)


    # TODO: efficiency may be improved by encapsulating sampled data into Datasets class and using distributed sampler.
    for epoch in range(run_epochs):
        temp = args.min_temp
        
        if args.reset_prior and epoch % (args.reset_prior_interval + 1) == 0:
            prior_head = model.get_head()

        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

        sampled_data, sampled_label = sample_data(task_id, class_mask, device, args, include_current_task=True, train=False)
        inputs = sampled_data
        targets = sampled_label

        sf_indexes = torch.randperm(inputs.size(0))
        inputs = inputs[sf_indexes]
        targets = targets[sf_indexes]

        for pos in range(0, inputs.size(0), args.batch_size):
            inp = inputs[pos:pos + args.batch_size]
            tgt = targets[pos:pos + args.batch_size]
            outputs = model(inp, fc_only=True)
            logits = outputs['logits']


            if args.train_mask and class_mask is not None:
                mask = []
                for id in range(task_id+1):
                    mask.extend(class_mask[id])
                not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

            with torch.no_grad():
                prior_output = model.forward_new_head(inp, *prior_head)

            prior_logits = prior_output['logits'] / temp
            if args.train_mask and class_mask is not None:
                prior_logits = prior_logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

            if args.rejection:
                prior_tgt = prior_logits.argmax(dim=1)
                logits = logits[prior_tgt == tgt]
                prior_logits = prior_logits[prior_tgt == tgt]   
                inp = inp[prior_tgt == tgt]
                tgt = tgt[prior_tgt == tgt]   
            
            log_q = F.log_softmax(logits, dim=1)
            
            log_prior = F.log_softmax(prior_logits, dim=1)
            log_r = (F.log_softmax(log_q[:, mask], dim=0) + log_prior[:, mask])
            log_r = F.log_softmax(log_r, dim=1)

            if args.uncertain_loss2 == "qr":
                loss = ((F.softmax(logits, dim=1)[:, mask] * log_q[:, mask]).sum(dim=1) - (F.softmax(logits, dim=1)[:, mask] * log_r).sum(dim=1)).mean()
            elif args.uncertain_loss2 == "ce":
                prior = F.softmax(prior_logits, dim=1)[: , mask]
                loss = (-F.log_softmax(logits[:, mask], dim=1) * prior).sum(1).mean()

            acc1, acc5 = accuracy(logits, tgt, topk=(1, 5))

            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()))
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            torch.cuda.synchronize()

            metric_logger.update(Loss=loss.item())
            metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
            metric_logger.meters['Acc@1'].update(acc1.item(), n=inp.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=inp.shape[0])

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        scheduler.step()


def train_task_adaptive(model: torch.nn.Module, args, device, class_mask=None, task_id=-1, data_loader=None):
    model.train()
    prior_head = old_head

    run_epochs = args.crct_epochs
    param_list = [p for p in model.parameters() if p.requires_grad]

    network_params = [{'params': param_list, 'lr': args.ca_lr, 'weight_decay': args.weight_decay}]
    if 'mae' in args.model or 'beit' in args.model:
        optimizer = optim.AdamW(network_params, lr=args.ca_lr / 10, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(network_params, lr=args.ca_lr, momentum=0.9, weight_decay=5e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)


    # TODO: efficiency may be improved by encapsulating sampled data into Datasets class and using distributed sampler.
    for epoch in range(run_epochs):
        if args.temp_anneal and run_epochs > 1:
            temp = args.temp - ((args.temp - args.min_temp) * epoch / (run_epochs - 1))
        else:
            temp = args.min_temp

        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

        for input, target in data_loader:
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(input)
            logits = output['logits']

            # here is the trick to mask out classes of non-current tasks
            if args.train_mask and class_mask is not None:
                mask = []
                for id in range(task_id+1):
                    mask.extend(class_mask[id])
                not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
                
            loss = F.cross_entropy(logits, target, reduction='sum')
            num_samples = input.size(0)

            sampled_data, sampled_label = sample_data(task_id, class_mask, device, args, include_current_task=False, train=True)
            inputs = sampled_data
            targets = sampled_label

            sf_indexes = torch.randperm(inputs.size(0))
            inputs = inputs[sf_indexes]
            targets = targets[sf_indexes]

            for pos in range(0, inputs.size(0), args.batch_size):
                inp = inputs[pos:pos + args.batch_size]
                tgt = targets[pos:pos + args.batch_size]
                sampled_outputs = model(inp, fc_only=True)
                sampled_logits = sampled_outputs['logits']

                if args.train_mask and class_mask is not None:
                    mask = []
                    old_mask = []
                    for id in range(task_id+1):
                        mask.extend(class_mask[id])
                        if id < task_id:
                            old_mask.extend(class_mask[id])
                    # print(mask)
                    not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                    not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                    not_old_mask = np.setdiff1d(np.arange(args.nb_classes), old_mask)
                    not_old_mask = torch.tensor(not_old_mask, dtype=torch.int64).to(device)
                    sampled_logits = sampled_logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

                with torch.no_grad():
                    prior_output = model.forward_new_head(inp, *prior_head)
                
                prior_logits = prior_output['logits'] / temp
                if args.train_mask and class_mask is not None:
                    prior_logits = prior_logits.index_fill(dim=1, index=not_old_mask, value=float('-inf'))
                    prior_logits = prior_logits[:, mask]

                if args.rejection:
                    prior_tgt = prior_logits.argmax(dim=1)
                    sampled_logits = sampled_logits[prior_tgt == tgt]
                    prior_logits = prior_logits[prior_tgt == tgt]
                    inp = inp[prior_tgt == tgt]
                    tgt = tgt[prior_tgt == tgt]      
                
                prior = F.softmax(prior_logits, dim=1)

                if args.uncertain_loss1 == "ce":
                    loss += (-F.log_softmax(sampled_logits[:, mask], dim=1) * prior).sum(1).sum()
                elif args.uncertain_loss1 == "qr":
                    log_q = F.log_softmax(sampled_logits, dim=1)[:, mask]
                    log_prior = prior.clamp(1e-8).log()
                    log_r = (F.log_softmax(log_q, dim=0) + log_prior)
                    log_r = F.log_softmax(log_r, dim=1)

                    loss += ((F.softmax(sampled_logits, dim=1)[:, mask] * log_q).sum(dim=1) - (F.softmax(sampled_logits, dim=1)[:, mask] * log_r).sum(dim=1)).sum()

                num_samples += inp.size(0)

            loss = loss / num_samples

            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()))
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()

            metric_logger.update(Loss=loss.item())
            metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
            metric_logger.meters['Acc@1'].update(acc1.item(), n=inp.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=inp.shape[0])

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        scheduler.step()



def train_adapt_prior(model: torch.nn.Module, args, device, class_mask=None, task_id=-1, data_loader=None):
    model.train()
    prior_head = model.get_head()

    run_epochs = args.crct_epochs
    param_list = [p for p in model.parameters() if p.requires_grad]

    network_params = [{'params': param_list, 'lr': args.ca_lr, 'weight_decay': args.weight_decay}]
    if 'mae' in args.model or 'beit' in args.model:
        optimizer = optim.AdamW(network_params, lr=args.ca_lr / 10, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(network_params, lr=args.ca_lr, momentum=0.9, weight_decay=5e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)


    # TODO: efficiency may be improved by encapsulating sampled data into Datasets class and using distributed sampler.
    for epoch in range(run_epochs):
        temp = args.min_temp

        if args.reset_prior and epoch % (args.reset_prior_interval + 1) == 0:
            prior_head = model.get_head()
            
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

        for input, target in data_loader:
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(input)
            logits = output['logits']

            # here is the trick to mask out classes of non-current tasks
            if args.train_mask and class_mask is not None:
                mask = []
                for id in range(task_id+1):
                    mask.extend(class_mask[id])
                not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
                
            loss = F.cross_entropy(logits, target, reduction='sum')
            num_samples = input.size(0)

            sampled_data, sampled_label = sample_data(task_id, class_mask, device, args, include_current_task=False, train=True)
            inputs = sampled_data
            targets = sampled_label

            sf_indexes = torch.randperm(inputs.size(0))
            inputs = inputs[sf_indexes]
            targets = targets[sf_indexes]

            for pos in range(0, inputs.size(0), args.batch_size):
                inp = inputs[pos:pos + args.batch_size]
                tgt = targets[pos:pos + args.batch_size]
                sampled_outputs = model(inp, fc_only=True)
                sampled_logits = sampled_outputs['logits']

                if args.train_mask and class_mask is not None:
                    mask = []
                    for id in range(task_id+1):
                        mask.extend(class_mask[id])
                    # print(mask)
                    not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                    not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                    sampled_logits = sampled_logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

                with torch.no_grad():
                    prior_output = model.forward_new_head(inp, *prior_head)
                
                prior_logits = prior_output['logits'] / temp
                if args.train_mask and class_mask is not None:
                    prior_logits = prior_logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
                    prior_logits = prior_logits[:, mask]

                if args.rejection:
                    prior_tgt = prior_logits.argmax(dim=1)
                    sampled_logits = sampled_logits[prior_tgt == tgt]
                    prior_logits = prior_logits[prior_tgt == tgt]
                    inp = inp[prior_tgt == tgt]
                    tgt = tgt[prior_tgt == tgt]      
                
                prior = F.softmax(prior_logits, dim=1)

                if args.uncertain_loss1 == "ce":
                    loss += (-F.log_softmax(sampled_logits[:, mask], dim=1) * prior).sum(1).sum()
                elif args.uncertain_loss1 == "qr":
                    log_q = F.log_softmax(sampled_logits, dim=1)[:, mask]
                    log_prior = prior.clamp(1e-8).log()
                    log_r = (F.log_softmax(log_q, dim=0) + log_prior)
                    log_r = F.log_softmax(log_r, dim=1)

                    loss += ((F.softmax(sampled_logits, dim=1)[:, mask] * log_q).sum(dim=1) - (F.softmax(sampled_logits, dim=1)[:, mask] * log_r).sum(dim=1)).sum()

                num_samples += inp.size(0)

            loss = loss / num_samples

            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()))
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()

            metric_logger.update(Loss=loss.item())
            metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
            metric_logger.meters['Acc@1'].update(acc1.item(), n=inp.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=inp.shape[0])

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        scheduler.step()