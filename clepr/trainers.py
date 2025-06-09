from __future__ import print_function, absolute_import
import time

import torch
import torch.nn as nn
from torch.nn import functional as F

from .evaluation_metrics import accuracy
from .loss import TripletLoss, CrossEntropyLabelSmooth, SoftTripletLoss, SoftEntropy, KLDivLoss, CrossEntropyLabelSmoothFilterNoise
from .utils.meters import AverageMeter


class PreTrainer(object):
    def __init__(self, model, num_classes, margin=0.0):
        super(PreTrainer, self).__init__()
        self.model = model
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes).cuda()
        self.criterion_triple = SoftTripletLoss(margin=margin).cuda()

    def train(self, epoch, data_loader_source, data_loader_target, optimizer, train_iters=200, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_tr = AverageMeter()
        precisions = AverageMeter()

        end = time.time()

        for i in range(train_iters):
            source_inputs = data_loader_source.next()
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            s_inputs, targets = self._parse_data(source_inputs)
            t_inputs, _ = self._parse_data(target_inputs)
            s_features, s_cls_out = self.model(s_inputs)
            # target samples: only forward
            t_features, _ = self.model(t_inputs)

            # backward main #
            loss_ce, loss_tr, prec1 = self._forward(s_features, s_cls_out, targets)
            loss = loss_ce + loss_tr

            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tr.item())
            precisions.update(prec1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if ((i + 1) % print_freq == 0):
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} ({:.3f})\t'
                      'Loss_tr {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_tr.val, losses_tr.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets

    def _forward(self, s_features, s_outputs, targets):
        loss_ce = self.criterion_ce(s_outputs, targets)
        if isinstance(self.criterion_triple, SoftTripletLoss):
            loss_tr = self.criterion_triple(s_features, s_features, targets)
        elif isinstance(self.criterion_triple, TripletLoss):
            loss_tr, _ = self.criterion_triple(s_features, targets)
        prec, = accuracy(s_outputs.data, targets.data)
        prec = prec[0]

        return loss_ce, loss_tr, prec

class ClusterBaseTrainer(object):
    def __init__(self, model, num_cluster=500):
        super(ClusterBaseTrainer, self).__init__()
        self.model = model
        self.num_cluster = num_cluster

        self.criterion_ce = CrossEntropyLabelSmooth(num_cluster).cuda()
        self.criterion_tri = SoftTripletLoss(margin=0.0).cuda()

    def train(self, epoch, data_loader_target, optimizer, print_freq=1, train_iters=200):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_ce = AverageMeter()
        losses_tri = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs, targets = self._parse_data(target_inputs)

            # forward
            f_out_t, p_out_t = self.model(inputs)
            p_out_t = p_out_t[:,:self.num_cluster]

            loss_ce = self.criterion_ce(p_out_t, targets)
            loss_tri = self.criterion_tri(f_out_t, f_out_t, targets)
            loss = loss_ce + loss_tri

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            prec, = accuracy(p_out_t.data, targets.data)

            losses_ce.update(loss_ce.item())
            losses_tri.update(loss_tri.item())
            precisions.update(prec[0])

            # print log #
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} ({:.3f})\t'
                      'Loss_tri {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader_target),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_tri.val, losses_tri.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets


class MMTTrainerCL(object):
    def __init__(self, model_1, model_2,
                       model_1_ema, model_2_ema, num_cluster=500, alpha=0.999, temperature=0.07):
        super(MMTTrainerCL, self).__init__()
        self.model_1 = model_1
        self.model_2 = model_2
        self.num_cluster = num_cluster

        self.model_1_ema = model_1_ema
        self.model_2_ema = model_2_ema
        self.alpha = alpha

        self.criterion_ce = CrossEntropyLabelSmooth(num_cluster).cuda()
        self.criterion_ce_soft = KLDivLoss().cuda()
        self.criterion_tri = SoftTripletLoss(margin=0.0).cuda()
        self.criterion_tri_soft = SoftTripletLoss(margin=None).cuda()
        
        # 对比学习的温度参数
        self.temperature = temperature
    
    def contrastive_loss(self, f1, f2, targets, exclude_self=False):
        """
        基于targets的对比学习损失
        f1, f2: 两个模型的特征输出
        targets: 标签
        exclude_self: 计算loss的时候是否排除相同下标
        """
        # 归一化特征
        f1 = F.normalize(f1, dim=1)
        f2 = F.normalize(f2, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(f1, f2.T) / self.temperature
        
        # 创建标签矩阵：相同target的位置为True
        targets_1 = targets.unsqueeze(1).expand(-1, targets.size(0))
        targets_2 = targets.unsqueeze(0).expand(targets.size(0), -1)
        mask_positive = (targets_1 == targets_2).float().cuda()
        
        # 对每个样本，计算其与所有正样本的相似度
        exp_sim = torch.exp(similarity_matrix)
        
        if exclude_self:
            # 创建mask来排除当前样本本身作为正样本
            mask_self = torch.eye(f1.size(0)).cuda()
            mask_positive = mask_positive * (1 - mask_self)  # 排除自身作为正样本
            # 如果exclude_self为True，在计算loss时也排除相同下标的样本对
            exp_sim = exp_sim * (1 - mask_self)  # 在计算相似度时排除对角线
            all_sim = exp_sim.sum(dim=1)  # 重新计算所有相似度之和（已排除对角线）
        else:
            # 如果exclude_self为False，保持原有计算方式
            all_sim = exp_sim.sum(dim=1)
        
        # 对每个样本，计算其正样本的loss
        pos_sim = exp_sim * mask_positive
        pos_sim = pos_sim.sum(dim=1)
        
        # 避免除零错误
        eps = 1e-8
        
        # 计算最终的loss: -log(sum(pos_sim) / sum(all_sim))
        loss = -torch.log(pos_sim / (all_sim + eps) + eps)
        
        # 计算正样本个数（除去自身）
        num_pos = mask_positive.sum(dim=1)
        
        # 只考虑有正样本的情况
        valid_mask = (num_pos > 0)
        loss = (loss * valid_mask).sum() / (valid_mask.sum() + eps)
        
        return loss

    def train(self, epoch, data_loader_target,
            optimizer, ce_soft_weight=0.5, tri_soft_weight=0.5, 
            contrast_weight=0.5, print_freq=1, train_iters=200):
        self.model_1.train()
        self.model_2.train()
        self.model_1_ema.train()
        self.model_2_ema.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_ce = [AverageMeter(),AverageMeter()]
        losses_tri = [AverageMeter(),AverageMeter()]
        losses_ce_soft = AverageMeter()
        losses_tri_soft = AverageMeter()
        losses_contrast = AverageMeter()
        precisions = [AverageMeter(),AverageMeter()]

        end = time.time()
        for i in range(train_iters):
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs_1, inputs_2, targets = self._parse_data(target_inputs)

            # forward
            f_out_t1, p_out_t1 = self.model_1(inputs_1)
            f_out_t2, p_out_t2 = self.model_2(inputs_2)
            p_out_t1 = p_out_t1[:,:self.num_cluster]
            p_out_t2 = p_out_t2[:,:self.num_cluster]

            f_out_t1_ema, p_out_t1_ema = self.model_1_ema(inputs_1)
            f_out_t2_ema, p_out_t2_ema = self.model_2_ema(inputs_2)
            p_out_t1_ema = p_out_t1_ema[:,:self.num_cluster]
            p_out_t2_ema = p_out_t2_ema[:,:self.num_cluster]

            loss_ce_1 = self.criterion_ce(p_out_t1, targets)
            loss_ce_2 = self.criterion_ce(p_out_t2, targets)

            loss_tri_1 = self.criterion_tri(f_out_t1, f_out_t1, targets)
            loss_tri_2 = self.criterion_tri(f_out_t2, f_out_t2, targets)

            # Mean Mean model
            p_out_ema = (p_out_t1_ema+p_out_t2_ema)/2
            f_out_ema = (f_out_t1_ema+f_out_t2_ema)/2
            loss_ce_soft = self.criterion_ce_soft(p_out_t1, p_out_ema) + self.criterion_ce_soft(p_out_t2, p_out_ema)
            loss_tri_soft = self.criterion_tri_soft(f_out_t1, f_out_ema, targets) + \
                            self.criterion_tri_soft(f_out_t2, f_out_ema, targets)
            
            # 计算基于target的对比损失
            #loss_contrast = self.contrastive_loss(f_out_t1, f_out_t2, targets)
            
            #loss_contrast = (self.contrastive_loss(f_out_t1, f_out_t2, targets, exclude_self=False)+self.contrastive_loss(f_out_t1, f_out_t1, targets, exclude_self=True)+self.contrastive_loss(f_out_t2, f_out_t2, targets, exclude_self=True))/3 # exclude self
            
            loss_contrast = (self.contrastive_loss(f_out_t1, f_out_ema, targets, exclude_self=False)+self.contrastive_loss(f_out_t2, f_out_ema, targets, exclude_self=False)+self.contrastive_loss(f_out_t1, f_out_t1, targets, exclude_self=True)+self.contrastive_loss(f_out_t2, f_out_t2, targets, exclude_self=True))/4 # new loss
            #loss_contrast = (self.contrastive_loss(f_out_t1, f_out_ema, targets, exclude_self=False)+self.contrastive_loss(f_out_t2, f_out_ema, targets, exclude_self=False))/2 # new loss 2
            
            #loss_contrast = (self.contrastive_loss(f_out_t1, f_out_t1, targets, exclude_self=True)+self.contrastive_loss(f_out_t2, f_out_t2, targets, exclude_self=True))/2 # exclude self v2
            #loss_contrast = self.contrastive_loss(f_out_t1, f_out_t2, targets, exclude_self=False)#exclude self v3
            
            # 总损失
            loss = (loss_ce_1 + loss_ce_2)*(1-ce_soft_weight) + \
                   (loss_tri_1 + loss_tri_2)*(1-tri_soft_weight) + \
                   loss_ce_soft*ce_soft_weight + loss_tri_soft*tri_soft_weight + \
                   loss_contrast*contrast_weight

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self._update_ema_variables(self.model_1, self.model_1_ema, self.alpha, epoch*len(data_loader_target)+i)
            self._update_ema_variables(self.model_2, self.model_2_ema, self.alpha, epoch*len(data_loader_target)+i)

            prec_1, = accuracy(p_out_t1.data, targets.data)
            prec_2, = accuracy(p_out_t2.data, targets.data)

            losses_ce[0].update(loss_ce_1.item())
            losses_ce[1].update(loss_ce_2.item())
            losses_tri[0].update(loss_tri_1.item())
            losses_tri[1].update(loss_tri_2.item())
            losses_ce_soft.update(loss_ce_soft.item())
            losses_tri_soft.update(loss_tri_soft.item())
            losses_contrast.update(loss_contrast.item())
            precisions[0].update(prec_1[0])
            precisions[1].update(prec_2[0])

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} / {:.3f}\t'
                      'Loss_tri {:.3f} / {:.3f}\t'
                      'Loss_ce_soft {:.3f}\t'
                      'Loss_tri_soft {:.3f}\t'
                      'Loss_contrast {:.3f}\t'
                      'Prec {:.2%} / {:.2%}\t'
                      .format(epoch, i + 1, len(data_loader_target),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce[0].avg, losses_ce[1].avg,
                              losses_tri[0].avg, losses_tri[1].avg,
                              losses_ce_soft.avg, losses_tri_soft.avg,
                              losses_contrast.avg,
                              precisions[0].avg, precisions[1].avg))

    def _update_ema_variables(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def _parse_data(self, inputs):
        imgs_1, imgs_2, pids = inputs
        inputs_1 = imgs_1.cuda()
        inputs_2 = imgs_2.cuda()
        targets = pids.cuda()
        return inputs_1, inputs_2, targets