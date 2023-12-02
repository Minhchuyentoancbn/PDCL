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
    global cls_mean
    global cls_cov
    global cls_mean_param
    global cls_cov_param
    global learnable_prototypes

    cls_mean = dict()
    cls_cov = dict()
    cls_mean_param = dict()
    cls_cov_param = dict()
    learnable_prototypes = list()

    for task_id in range(args.num_tasks):

        # Create new optimizer for each task to clear optimizer status
        if task_id > 0 and args.reinit_optimizer:
            optimizer = create_optimizer(args, model)
            if args.sched != 'constant':
                lr_scheduler, _ = create_scheduler(args, optimizer)
            elif args.sched == 'constant':
                lr_scheduler = None

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

        # Update prototypes
        update_prototypes(model=model, args=args, device=device, class_mask=class_mask, task_id=task_id, data_loader=data_loader[task_id]['train'])

        # TODO classifier alignment
        if task_id > 0:
            print('-' * 20)
            print(f'Align classifier for task {task_id + 1}')
            train_task_adaptive_prediction(model, args, device, class_mask, task_id)
            print('-' * 20)

        # Evaluate model
        print('-' * 20)
        print(f'Evaluate task {task_id + 1} after CA')
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


def train_one_epoch(model: torch.nn.Module, criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, args=None):
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
        
        if args.use_gaussian:
            if args.train_mask and class_mask is not None:
                mask = []
                for id in range(task_id+1):
                    mask.extend(class_mask[id])
                # print(mask)
                not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

            loss = criterion(logits, target)

            if cls_mean:
                sampled_data = []
                sampled_label = []
                num_sampled_pcls = int(args.batch_size / args.nb_classes * args.num_tasks)

                if args.ca_storage_efficient_method in ['covariance', 'variance']:
                    for i in range(task_id):
                        for c_id in class_mask[i]:
                            # mean = torch.tensor(cls_mean[c_id], dtype=torch.float64).to(device)
                            # cov = cls_cov[c_id].to(device)
                            mean = torch.tensor(cls_mean_param[c_id], dtype=torch.float64).to(device)
                            cov = (cls_cov_param[c_id] @ cls_cov_param[c_id].T).to(device) + 1e-4 * torch.eye(cls_cov_param[c_id].shape[0]).to(device)
                            if args.ca_storage_efficient_method == 'variance':
                                cov = torch.diag(torch.diag(cov))
                            m = MultivariateNormal(mean.float(), cov.float())
                            sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                            sampled_data.append(sampled_data_single)

                            sampled_label.extend([c_id] * num_sampled_pcls)

                elif args.ca_storage_efficient_method == 'multi-centroid':
                    num_sampled_pcls = num_sampled_pcls // args.n_centroids
                    for i in range(task_id):
                        for c_id in class_mask[i]:
                            for cluster in range(len(cls_mean[c_id])):
                                # mean = cls_mean[c_id][cluster]
                                # var = cls_cov[c_id][cluster]
                                mean = cls_mean_param[c_id][cluster]
                                var = (cls_cov_param[c_id][cluster] @ cls_cov_param[c_id][cluster].T) + 1e-4 * torch.eye(cls_cov_param[c_id][cluster].shape[0]).to(cls_cov_param[c_id][cluster].device)
                                if var.sum() == 0:
                                    continue
                                m = MultivariateNormal(mean.float(), var.float())
                                sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                                sampled_data.append(sampled_data_single)
                                sampled_label.extend([c_id] * num_sampled_pcls)
                
                sampled_data = torch.cat(sampled_data, dim=0).float().to(device)
                sampled_label = torch.tensor(sampled_label).long().to(device)
                sampled_output = model(sampled_data, fc_only=True)
                sampled_logits = sampled_output['logits']
                if args.train_mask and class_mask is not None:
                    sampled_logits = sampled_logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
                sampled_loss = criterion(sampled_logits, sampled_label)

                loss += args.reg * (input.shape[0] / sampled_data.shape[0]) * sampled_loss

        else:
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

    # Reset learnable prototypes
    global learnable_prototypes
    learnable_prototypes = list()

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

            cls_mean_param[cls_id] = nn.Parameter(cls_mean[cls_id])
            # Decompose covariance matrix
            l = torch.linalg.cholesky(cls_cov[cls_id])
            cls_cov_param[cls_id] = nn.Parameter(l)
            learnable_prototypes.append(cls_mean_param[cls_id])
            learnable_prototypes.append(cls_cov_param[cls_id])

        if args.ca_storage_efficient_method == 'variance':
            # features_per_cls = torch.cat(features_per_cls_list, dim=0)
            # print(features_per_cls.shape)
            cls_mean[cls_id] = features_per_cls.mean(dim=0)
            cls_cov[cls_id] = torch.diag(torch.cov(features_per_cls.T) + (torch.eye(cls_mean[cls_id].shape[-1]) * 1e-4).to(device))
            
            cls_mean_param[cls_id] = nn.Parameter(cls_mean[cls_id])
            # Decompose covariance matrix
            l = torch.linalg.cholesky(torch.diag(cls_cov[cls_id]))
            cls_cov_param[cls_id] = nn.Parameter(l)
            learnable_prototypes.append(cls_mean_param[cls_id])
            learnable_prototypes.append(cls_cov_param[cls_id])
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
            cluster_means_param = []
            cluster_vars_param = []

            for i in range(n_clusters):
                cluster_data = features_per_cls[cluster_lables == i]
                cluster_mean = torch.tensor(np.mean(cluster_data, axis=0), dtype=torch.float64).to(device)
                cluster_var = torch.tensor(np.var(cluster_data, axis=0), dtype=torch.float64).to(device)
                cluster_means.append(cluster_mean)
                cluster_vars.append(cluster_var)

                cluster_mean_proto = nn.Parameter(cluster_mean)
                cluster_means_param.append(cluster_mean_proto)
                learnable_prototypes.append(cluster_mean_proto)
                # Decompose covariance matrix
                l = torch.linalg.cholesky(torch.diag(cluster_var) + 1e-4 * torch.eye(cluster_mean.shape[0]).to(cluster_mean.device))
                cluster_var_proto = nn.Parameter(l)
                cluster_vars_param.append(cluster_var_proto)
                learnable_prototypes.append(cluster_var_proto)
            
            cls_mean[cls_id] = cluster_means
            cls_cov[cls_id] = cluster_vars

            cls_mean_param[cls_id] = cluster_means_param
            cls_cov_param[cls_id] = cluster_vars_param

    count = 0
    for p in learnable_prototypes:
        if p.requires_grad:
            count += 1
    print(f'Len of learnable prototypes: {count}')


def train_task_adaptive_prediction(model: torch.nn.Module, args, device, class_mask=None, task_id=-1):
    model.train()
    run_epochs = args.crct_epochs
    crct_num = 0
    param_list = [p for p in model.parameters() if p.requires_grad]
    print('-' * 20)
    print('Learnable parameters:')
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(name)
    print('-' * 20)
    network_params = [{'params': param_list, 'lr': args.ca_lr, 'weight_decay': args.weight_decay}]
    if 'mae' in args.model or 'beit' in args.model:
        optimizer = optim.AdamW(network_params, lr=args.ca_lr / 10, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(network_params, lr=args.ca_lr, momentum=0.9, weight_decay=5e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)
    criterion = nn.CrossEntropyLoss().to(device)

    for i in range(task_id):
        crct_num += len(class_mask[i])

    # TODO: efficiency may be improved by encapsulating sampled data into Datasets class and using distributed sampler.
    for epoch in range(run_epochs):

        sampled_data = []
        sampled_label = []
        num_sampled_pcls = args.batch_size

        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

        if args.ca_storage_efficient_method in ['covariance', 'variance']:
            for i in range(task_id + 1):
                for c_id in class_mask[i]:
                    # mean = torch.tensor(cls_mean[c_id], dtype=torch.float64).to(device)
                    # cov = cls_cov[c_id].to(device)
                    mean = torch.tensor(cls_mean_param[c_id], dtype=torch.float64).to(device)
                    cov = (cls_cov_param[c_id] @ cls_cov_param[c_id].T).to(device)
                    if args.ca_storage_efficient_method == 'variance':
                        # cov = torch.diag(cov)
                        cov = torch.diag(torch.diag(cov))
                    m = MultivariateNormal(mean.float(), cov.float())
                    sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                    sampled_data.append(sampled_data_single)
                    sampled_label.extend([c_id] * num_sampled_pcls)

        elif args.ca_storage_efficient_method == 'multi-centroid':
            for i in range(task_id + 1):
                for c_id in class_mask[i]:
                    for cluster in range(len(cls_mean[c_id])):
                        # mean = cls_mean[c_id][cluster]
                        # var = cls_cov[c_id][cluster]
                        mean = cls_mean_param[c_id][cluster]
                        var = (cls_cov_param[c_id][cluster] @ cls_cov_param[c_id][cluster].T) + 1e-4 * torch.eye(cls_cov_param[c_id][cluster].shape[0]).to(cls_cov_param[c_id][cluster].device)
                        if var.sum() == 0:
                            continue
                        m = MultivariateNormal(mean.float(), var.float())
                        sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                        sampled_data.append(sampled_data_single)
                        sampled_label.extend([c_id] * num_sampled_pcls)

        else:
            raise NotImplementedError


        sampled_data = torch.cat(sampled_data, dim=0).float().to(device)
        sampled_label = torch.tensor(sampled_label).long().to(device)

        inputs = sampled_data
        targets = sampled_label

        sf_indexes = torch.randperm(inputs.size(0))
        inputs = inputs[sf_indexes]
        targets = targets[sf_indexes]
        #print(targets)

        for _iter in range(crct_num):
            inp = inputs[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
            tgt = targets[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]

            outputs = model(inp, fc_only=True)
            logits = outputs['logits']

            if args.train_mask and class_mask is not None:
                mask = []
                for id in range(task_id+1):
                    mask.extend(class_mask[id])
                # print(mask)
                not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

            loss = criterion(logits, tgt)  # base criterion (CrossEntropyLoss)

            acc1, acc5 = accuracy(logits, tgt, topk=(1, 5))

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


def update_prototypes(model: torch.nn.Module, args, device, class_mask=None, task_id=-1, data_loader=None):
    if task_id > 0:
        run_epochs = args.proto_epochs
        print('-' * 20)
        print('Start updating prototypes')
        print('-' * 20)
        if 'mae' in args.model or 'beit' in args.model:
            optimizer = optim.AdamW(learnable_prototypes, lr=args.proto_lr / 10)
        else:
            optimizer = optim.SGD(learnable_prototypes, lr=args.proto_lr, momentum=0.9)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)
        criterion = nn.CrossEntropyLoss().to(device) 

        if args.train_mask and class_mask is not None:
            mask = []
            for id in range(task_id+1):
                mask.extend(class_mask[id])
            # print(mask)
            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)

        for epoch in range(run_epochs):
            metric_logger = utils.MetricLogger(delimiter="  ")
            metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
            metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
            # Compute class mean for each class seen so far
            train_prototypes = torch.ones((args.nb_classes, args.hidden_dim), device=device)

            if cls_mean:
                num_sampled_pcls = args.batch_size

                if args.ca_storage_efficient_method in ['covariance', 'variance']:
                    for i in range(task_id + 1):
                        for c_id in class_mask[i]:
                            # mean = torch.tensor(cls_mean[c_id], dtype=torch.float64).to(device)
                            # cov = cls_cov[c_id].to(device)
                            mean = torch.tensor(cls_mean_param[c_id], dtype=torch.float64).to(device)
                            cov = (cls_cov_param[c_id] @ cls_cov_param[c_id].T).to(device)
                            if args.ca_storage_efficient_method == 'variance':
                                cov = torch.diag(torch.diag(cov))
                            m = MultivariateNormal(mean.float(), cov.float())
                            sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                            sampled_data_single = sampled_data_single.to(device)

                            with torch.no_grad():
                                sampled_output = model(sampled_data_single, fc_only=True)
                                sampled_pre_features = sampled_output['pre_features']
                            train_prototypes[c_id] = sampled_pre_features.mean(dim=0)

                            
                elif args.ca_storage_efficient_method == 'multi-centroid':
                    num_sampled_pcls = num_sampled_pcls // args.n_centroids
                    for i in range(task_id + 1):
                        for c_id in class_mask[i]:
                            cluster_features = []
                            for cluster in range(len(cls_mean[c_id])):
                                # mean = cls_mean[c_id][cluster]
                                # var = cls_cov[c_id][cluster]
                                mean = cls_mean_param[c_id][cluster]
                                var = (cls_cov_param[c_id][cluster] @ cls_cov_param[c_id][cluster].T)
                                if var.sum() == 0:
                                    continue
                                m = MultivariateNormal(mean.float(), var.float())
                                sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                                with torch.no_grad():
                                    sampled_output = model(sampled_data_single, fc_only=True)
                                    sampled_pre_features = sampled_output['pre_features']
                                cluster_features.append(sampled_pre_features)
                            
                            cluster_features = torch.cat(cluster_features, dim=0)
                            train_prototypes[c_id] = cluster_features.mean(dim=0)

                # Sample test data
                for input, target in data_loader:
                    input = input.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)

                    output = model(input)

                    test_features = output['pre_features']
                    logits = F.linear(F.normalize(test_features), F.normalize(train_prototypes))
                    
                    if args.train_mask and class_mask is not None:
                        logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

                    loss = criterion(logits, target)

                    if cls_mean:
                        sampled_data = []
                        sampled_label = []
                        num_sampled_pcls = int(args.batch_size / args.nb_classes * args.num_tasks)

                        if args.ca_storage_efficient_method in ['covariance', 'variance']:
                            for i in range(task_id):
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
                            for i in range(task_id):
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
                        
                        sampled_data = torch.cat(sampled_data, dim=0).float().to(device)
                        sampled_label = torch.tensor(sampled_label).long().to(device)
                        sampled_output = model(sampled_data, fc_only=True)

                        sampled_test_features = sampled_output['pre_features']
                        sampled_logits = F.linear(F.normalize(sampled_test_features), F.normalize(train_prototypes))

                        if args.train_mask and class_mask is not None:
                            sampled_logits = sampled_logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
                        sampled_loss = criterion(sampled_logits, sampled_label)

                        loss += args.proto_reg * (input.shape[0] / sampled_data.shape[0]) * sampled_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(learnable_prototypes, args.clip_grad)
                    optimizer.step()
                    torch.cuda.synchronize()

                    if not math.isfinite(loss.item()):
                        print("Loss is {}, stopping training".format(loss.item()))
                        sys.exit(1)
                    acc1, acc5 = accuracy(logits, target, topk=(1, 5))
                    metric_logger.update(Loss=loss.item())
                    metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
                    metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
                    metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
            
                metric_logger.synchronize_between_processes()
                print("Averaged stats:", metric_logger)
                scheduler.step()        

    print('-' * 20)
    # Freeze learnable prototypes
    for param in learnable_prototypes:
        param.requires_grad = False



# def train_task_adaptive_prediction(model: torch.nn.Module, args, device, class_mask=None, task_id=-1):
#     model.train()
#     run_epochs = args.crct_epochs
#     crct_num = 0
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
#     criterion = nn.CrossEntropyLoss().to(device)

#     for i in range(task_id):
#         crct_num += len(class_mask[i])

#     # TODO: efficiency may be improved by encapsulating sampled data into Datasets class and using distributed sampler.
#     for epoch in range(run_epochs):

#         sampled_data = []
#         sampled_label = []
#         num_sampled_pcls = args.batch_size

#         metric_logger = utils.MetricLogger(delimiter="  ")
#         metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
#         metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

#         if args.ca_storage_efficient_method in ['covariance', 'variance']:
#             for i in range(task_id + 1):
#                 for c_id in class_mask[i]:
#                     mean = torch.tensor(cls_mean[c_id], dtype=torch.float64).to(device)
#                     cov = cls_cov[c_id].to(device)
#                     if args.ca_storage_efficient_method == 'variance':
#                         cov = torch.diag(cov)
#                     m = MultivariateNormal(mean.float(), cov.float())
#                     sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
#                     sampled_data.append(sampled_data_single)

#                     sampled_label.extend([c_id] * num_sampled_pcls)

#         elif args.ca_storage_efficient_method == 'multi-centroid':
#             for i in range(task_id + 1):
#                 for c_id in class_mask[i]:
#                     for cluster in range(len(cls_mean[c_id])):
#                         mean = cls_mean[c_id][cluster]
#                         var = cls_cov[c_id][cluster]
#                         if var.mean() == 0:
#                             continue
#                         m = MultivariateNormal(mean.float(), (torch.diag(var) + 1e-4 * torch.eye(mean.shape[0]).to(mean.device)).float())
#                         sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
#                         sampled_data.append(sampled_data_single)
#                         sampled_label.extend([c_id] * num_sampled_pcls)

#         else:
#             raise NotImplementedError


#         sampled_data = torch.cat(sampled_data, dim=0).float().to(device)
#         sampled_label = torch.tensor(sampled_label).long().to(device)

#         inputs = sampled_data
#         targets = sampled_label

#         sf_indexes = torch.randperm(inputs.size(0))
#         inputs = inputs[sf_indexes]
#         targets = targets[sf_indexes]
#         #print(targets)

#         for _iter in range(crct_num):
#             inp = inputs[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
#             tgt = targets[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]

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

#             loss = criterion(logits, tgt)  # base criterion (CrossEntropyLoss)

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

#             # gather the stats from all processes
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
