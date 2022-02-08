import torch
from torch.utils import data
from utils.buffer.buffer import Buffer
from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from utils.setup_elements import transforms_match, input_size_match
from utils.utils import maybe_cuda, AverageMeter
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale
import torch.nn as nn
import copy
from utils.loss import SupConLoss2
import sys

class SupContrastReplay(ContinualLearner):
    def __init__(self, model, opt, params):
        super(SupContrastReplay, self).__init__(model, opt, params)
        self.buffer = Buffer(model, params)
        self.mem_size = params.mem_size
        self.eps_mem_batch = params.eps_mem_batch
        self.mem_iters = params.mem_iters
        self.transform = nn.Sequential(
            RandomResizedCrop(size=(input_size_match[self.params.data][1], input_size_match[self.params.data][2]), scale=(0.2, 1.)),
            RandomHorizontalFlip(),
            ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            RandomGrayscale(p=0.2)

        )

    def train_learner(self, x_train, y_train):
        self.before_train(x_train, y_train)
        # set up loader
        train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True, num_workers=0,
                                       drop_last=True)
        # set up model
        self.model = self.model.train()

        # setup tracker
        losses = AverageMeter()
        distill = AverageMeter()
        acc_batch = AverageMeter()
        data_time = AverageMeter()
        batch_time = AverageMeter()
        criterion = SupConLoss2(temperature=0.2)

        for ep in range(self.epoch):
            model2 = copy.deepcopy(self.model)
            for i, batch_data in enumerate(train_loader):
                # batch update
                batch_x, batch_y = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)
                bsz = batch_y.shape[0]
                for j in range(self.mem_iters):
                    mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)

                    if mem_x.size(0) > 0:
                        mem_x = maybe_cuda(mem_x, self.cuda)
                        mem_y = maybe_cuda(mem_y, self.cuda)
                        combined_batch = torch.cat((mem_x, batch_x))
                        combined_labels = torch.cat((mem_y, batch_y))
                        bsz=combined_labels.shape[0]
                        combined_batch_aug = self.transform(combined_batch)
                        #print("fisrt",batch_x.shape,mem_x.shape)
                        #print("feature_shape:",self.model.forward(combined_batch).squeeze(1).shape,self.model.forward(combined_batch_aug).squeeze(1).shape)
                        features_ori = torch.cat([self.model.forward(combined_batch).unsqueeze(1), self.model.forward(combined_batch_aug).unsqueeze(1)], dim=1)
                        
                        loss = self.criterion(features_ori, combined_labels)
                        if ep>=1:
                            features = torch.cat([self.model.forward(combined_batch).squeeze(1), self.model.forward(combined_batch_aug).squeeze(1)], dim=0)
                            features1_prev_task = features
                            #print("featue22:",features.shape)
                            #print("target:",features1_prev_task.shape,features1_prev_task.T.shape)
                            features1_sim = torch.div(torch.matmul(features1_prev_task, features1_prev_task.T), 0.2)
                            logits_mask = torch.scatter(
                            torch.ones_like(features1_sim),
                            1,
                            torch.arange(features1_sim.size(0)).view(-1, 1).cuda(non_blocking=True),
                            0
                            )
                            logits_max1, _ = torch.max(features1_sim * logits_mask, dim=1, keepdim=True)
                            features1_sim = features1_sim - logits_max1.detach()
                            row_size = features1_sim.size(0)
                            logits1 = torch.exp(features1_sim[logits_mask.bool()].view(row_size, -1)) / torch.exp(features1_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)
                            #print("features.shape,bsz",features.shape,int(features.shape[0]/2))
                            #print("label.shape",batch_y.shape,combined_labels.shape)
                            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                            loss += criterion(features, combined_labels)
                            with torch.no_grad():
                                features2_prev_task = torch.cat([model2.forward(combined_batch).squeeze(1), model2.forward(combined_batch_aug).squeeze(1)], dim=0)
                                features2_sim = torch.div(torch.matmul(features2_prev_task, features2_prev_task.T), 0.01)
                                logits_max2, _ = torch.max(features2_sim*logits_mask, dim=1, keepdim=True)
                                features2_sim = features2_sim - logits_max2.detach()
                                logits2 = torch.exp(features2_sim[logits_mask.bool()].view(row_size, -1)) /  torch.exp(features2_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)
                            loss_distill = (-logits2 * torch.log(logits1)).sum(1).mean()
                            loss += 1.0 * loss_distill
                            distill.update(loss_distill.item(), bsz)
                        
                        #loss = self.criterion(features, combined_labels)
                        losses.update(loss.item(), bsz)
                        self.opt.zero_grad()
                        loss.backward()
                        self.opt.step()

                # update mem
                self.buffer.update(batch_x, batch_y)
                if i % 100 == 1 and self.verbose:
                    
                    print(
                            '==>>> it: {}, avg. loss: {:.6f}, '
                                .format(i, losses.avg(), acc_batch.avg())
                        )
                        # print info
                    if ep>2:    
                        print('Train: {} distill:{.3f})'.format(
                           ep, distill.avg()))
                    #sys.stdout.flush()
        self.after_train()
