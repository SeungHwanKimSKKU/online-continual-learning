from abc import abstractmethod
import abc
import numpy as np
import torch
from torch.nn import functional as F
from utils.kd_manager import KdManager
from utils.utils import maybe_cuda, AverageMeter
from torch.utils.data import TensorDataset, DataLoader
import copy
from utils.loss import SupConLoss
import pickle
from agents.basic import Protonet
from fewshot.utils import *
import pdb
class ContinualLearner(torch.nn.Module, metaclass=abc.ABCMeta):
    '''
    Abstract module which is inherited by each and every continual learning algorithm.
    '''

    def __init__(self, model, opt, params):
        super(ContinualLearner, self).__init__()
        self.params = params
        self.model = model
        self.opt = opt
        self.data = params.data
        self.cuda = params.cuda
        self.epoch = params.epoch
        self.batch = params.batch
        self.verbose = params.verbose
        self.old_labels = []
        self.new_labels = []
        self.task_seen = 0
        self.kd_manager = KdManager()
        self.error_list = []
        self.new_class_score = []
        self.old_class_score = []
        self.fc_norm_new = []
        self.fc_norm_old = []
        self.bias_norm_new = []
        self.bias_norm_old = []
        self.lbl_inv_map = {}
        self.class_task_map = {}
        #self.init_weights=self.init_weights
        self.log_sigma_l = torch.log(torch.FloatTensor([1.0]))
        self.learn_sigma_l = False
        if self.learn_sigma_l:
            self.log_sigma_l = nn.Parameter(self.log_sigma_l, requires_grad=True)
        else:
            self.log_sigma_l = Variable(self.log_sigma_l, requires_grad=True).cuda()
    def before_train(self, x_train, y_train):
        new_labels = list(set(y_train.tolist()))
        self.new_labels += new_labels
        for i, lbl in enumerate(new_labels):
            self.lbl_inv_map[lbl] = len(self.old_labels) + i

        for i in new_labels:
            self.class_task_map[i] = self.task_seen

    @abstractmethod
    def train_learner(self, x_train, y_train):
        pass

    def after_train(self):
        #self.old_labels = list(set(self.old_labels + self.new_labels))
        self.old_labels += self.new_labels
        self.new_labels_zombie = copy.deepcopy(self.new_labels)
        self.new_labels.clear()
        self.task_seen += 1
        if self.params.trick['review_trick'] and hasattr(self, 'buffer'):
            self.model.train()
            mem_x = self.buffer.buffer_img[:self.buffer.current_index]
            mem_y = self.buffer.buffer_label[:self.buffer.current_index]
            # criterion = torch.nn.CrossEntropyLoss(reduction='mean')
            if mem_x.size(0) > 0:
                rv_dataset = TensorDataset(mem_x, mem_y)
                rv_loader = DataLoader(rv_dataset, batch_size=self.params.eps_mem_batch, shuffle=True, num_workers=0,
                                       drop_last=True)
                for ep in range(1):
                    for i, batch_data in enumerate(rv_loader):
                        # batch update
                        batch_x, batch_y = batch_data
                        batch_x = maybe_cuda(batch_x, self.cuda)
                        batch_y = maybe_cuda(batch_y, self.cuda)
                        logits = self.model.forward(batch_x)
                        if self.params.agent == 'SCR':
                            logits = torch.cat([self.model.forward(batch_x).unsqueeze(1),
                                                  self.model.forward(self.transform(batch_x)).unsqueeze(1)], dim=1)
                        loss = self.criterion(logits, batch_y)
                        self.opt.zero_grad()
                        loss.backward()
                        params = [p for p in self.model.parameters() if p.requires_grad and p.grad is not None]
                        grad = [p.grad.clone()/10. for p in params]
                        for g, p in zip(grad, params):
                            p.grad.data.copy_(g)
                        self.opt.step()

        if self.params.trick['kd_trick'] or self.params.agent == 'LWF':
            self.kd_manager.update_teacher(self.model)

    def criterion(self, logits, labels):
        labels = labels.clone()
        ce = torch.nn.CrossEntropyLoss(reduction='mean')
        if self.params.trick['labels_trick']:
            unq_lbls = labels.unique().sort()[0]
            for lbl_idx, lbl in enumerate(unq_lbls):
                labels[labels == lbl] = lbl_idx
            # Calcualte loss only over the heads appear in the batch:
            return ce(logits[:, unq_lbls], labels)
        elif self.params.trick['separated_softmax']:
            old_ss = F.log_softmax(logits[:, self.old_labels], dim=1)
            new_ss = F.log_softmax(logits[:, self.new_labels], dim=1)
            ss = torch.cat([old_ss, new_ss], dim=1)
            for i, lbl in enumerate(labels):
                labels[i] = self.lbl_inv_map[lbl.item()]
            return F.nll_loss(ss, labels)
        elif self.params.agent in ['SCR', 'SCP']:
            SC = SupConLoss(temperature=self.params.temp)
            return SC(logits, labels)
        else:
            return ce(logits, labels)

    def forward(self, x):
        return self.model.forward(x)
    def conv_block(in_channels, out_channels):
        return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
    def final_conv_block(in_channels, out_channels):
        return nn.Sequential(
                nn.Conv2d(in_channels, out_channels+1, 3, padding=1),
                nn.BatchNorm2d(out_channels+1),
                nn.MaxPool2d(2)
            )

        #self.encoder = nn.Sequential(
        #            conv_block(x_dim[0], hid_dim),
        #            conv_block(hid_dim, hid_dim),
        #            conv_block(hid_dim, hid_dim),
        #            conv_block(hid_dim, z_dim),
        #            Flatten()
        #    )

    #self.init_weights()

    def init_weights(self):
        def conv_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                init.xavier_uniform(m.weight, gain=np.sqrt(2))
                init.constant(m.bias, 0)
            elif classname.find('BatchNorm') != -1:
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)

        self.encoder = self.encoder.apply(conv_init)

    def _compute_distances(self, protos, example):
        dist = torch.sum((example - protos)**2, dim=2)
        return dist

    #def _process_batch(self, batch, super_classes=False):
    #    """Convert np arrays to variable"""
    #    x_train = Variable(torch.from_numpy(batch.x_train).type(torch.FloatTensor), requires_grad=False).cuda()
    #    x_test = Variable(torch.from_numpy(batch.x_test).type(torch.FloatTensor), requires_grad=False).cuda()

    #    if batch.x_unlabel is not None and batch.x_unlabel.size > 0:
    #        x_unlabel = Variable(torch.from_numpy(batch.x_unlabel).type(torch.FloatTensor), requires_grad=False).cuda()
    #        y_unlabel = Variable(torch.from_numpy(batch.y_unlabel.astype(np.int64)), requires_grad=False).cuda()
    #    else:
    #        x_unlabel = None
    #        y_unlabel = None

    #    if super_classes:
    #        labels_train = Variable(torch.from_numpy(batch.y_train_str[:,1]).type(torch.LongTensor), requires_grad=False).unsqueeze(0).cuda()
    #        labels_test = Variable(torch.from_numpy(batch.y_test_str[:,1]).type(torch.LongTensor), requires_grad=False).unsqueeze(0).cuda()
    #    else:
    #        labels_train = Variable(torch.from_numpy(batch.y_train.astype(np.int64)[:,:,1]),requires_grad=False).cuda()
    #        labels_test = Variable(torch.from_numpy(batch.y_test.astype(np.int64)[:,:,1]),requires_grad=False).cuda()

    #    return Episode(x_train,
    #                                 labels_train,
    #                                 np.expand_dims(batch.train_indices,0),
    #                                 x_test,
    #                                 labels_test,
    #                                 np.expand_dims(batch.test_indices,0),
    #                                 x_unlabel=x_unlabel,
    #                                 y_unlabel=y_unlabel,
    #                                 unlabel_indices=np.expand_dims(batch.unlabel_indices,0),
    #                                 y_train_str=batch.y_train_str,
    #                                 y_test_str=batch.y_test_str)
    #

    def _noisify_labels(self, y_train, num_noisy=1):
        if num_noisy > 0:
            num_classes = len(np.unique(y_train))
            shot = int(y_train.shape[1]/num_classes)
            selected_idxs = y_train[:,::int(shot/num_noisy)]
            y_train[:,::int(shot/num_noisy)] = np.random.randint(0, num_classes, len(selected_idxs[0]))
        return y_train


    #def _run_forward(self, cnn_input):
    #    n_class = cnn_input.size(1)
    #    n_support = cnn_input.size(0)
    #    encoded = self.encoder.forward(cnn_input.view(n_class * n_support, *cnn_input.size()[2:]))
    #    return encoded.unsqueeze(0)

    def _compute_protos(self, h, probs):
        """Compute the prototypes
        Args:
            h: [B, N, D] encoded inputs
            probs: [B, N, nClusters] soft assignment
        Returns:
            cluster protos: [B, nClusters, D]
        """
        #h=h.view(h.shape[0],-1,probs.shape[1])
        print("1.h.shape,probs.shape:",h.shape,probs.shape)
        h = torch.unsqueeze(h, 1)       # [B, N, 1, D] [B,1,D]
        print("2.h.shape,probs.shape:",h.shape,probs.shape)
        
        probs = torch.unsqueeze(probs, 2)       # [B, N, nClusters, 1] [B,nCluster,1] 
        print("3.h.shape,probs.shape:",h.shape,probs.shape)
        prob_sum = torch.sum(probs, 1)  # [B, nClusters, 1] ?
        print("4.h.shape,probs.shape:",h.shape,probs.shape,prob_sum.shape)
        zero_indices = (prob_sum.view(-1) == 0).nonzero()
        print("5.h.shape,probs.shape:",h.shape,probs.shape,prob_sum.shape,zero_indices.shape,zero_indices)
        print("torch.numel(zero_indices):",torch.numel(zero_indices))
        if torch.numel(zero_indices) != 0:
            values = torch.masked_select(torch.ones_like(prob_sum), torch.eq(prob_sum, 0.0))
            print("6.h.shape,probs.shape:",h.shape,probs.shape,prob_sum.shape,zero_indices.shape,values)
            prob_sum = prob_sum.put_(zero_indices, values)
            print("7.h.shape,probs.shape:",h.shape,probs.shape,prob_sum.shape,zero_indices.shape,values,prob_sum)
        print("8.h.shape,probs.shape:",h.shape,probs.shape,prob_sum.shape,zero_indices.shape)
        protos = h*probs    # [B, N, nClusters, D]
        protos = torch.sum(protos, 1)/prob_sum
        print("final_protos.shape:",protos.shape)
        return protos #[B, nClusters, D]

    def _get_count(self, probs, soft=True):
        """
        Args:
            probs: [B, N, nClusters] soft assignments
        Returns:
            counts: [B, nClusters] number of elements in each cluster
        """
        if not soft:
            _, max_indices = torch.max(probs, 2)    # [B, N]
            nClusters = probs.size()[2]
            max_indices = one_hot(max_indices, nClusters)
            counts = torch.sum(max_indices, 1).cuda()
        else:
            counts = torch.sum(probs, 1)
        return counts

    def _embedding_variance(self, x):
        """Compute variance in embedding space
        Args:
            x: examples from one class
        Returns:
            in-class variance
        """
        h = self._run_forward(x)   # [B, N, D]
        D = h.size()[2]
        h = h.view(-1, D)   # [BxN, D]
        variance = torch.var(h, 0)
        return torch.sum(variance)

    def _within_class_variance(self, x_list):
        protos = []
        for x in x_list:
            h = self._run_forward(x)
            D = h.size()[2]
            h = h.view(-1, D)   # [BxN, D]
            proto = torch.mean(h, 0)
            protos.append(proto)
        protos = torch.cat(protos, 0)
        variance = torch.var(protos, 0)
        return torch.sum(variance)

    def _within_class_distance(self, x_list):
        protos = []
        for x in x_list:
            h = self._run_forward(x)
            D = h.size()[2]
            h = h.view(-1, D)   # [BxN, D]
            proto = torch.mean(h, 0, keepdim=True)
            protos.append(proto)
        protos = torch.cat(protos, 0).data.cpu().numpy()   # [C, D]
        num_classes = protos.shape[0]
        distances = []
        for i in range(num_classes):
            for j in range(num_classes):
                if i > j:
                    dist = np.sum((protos[i, :] - protos[j, :])**2)
                    distances.append(dist)
        return np.mean(distances)


    #def forward(self, sample):
    #    raise NotImplementedError
    def evaluate(self, test_loaders):
        self.model.eval()
        acc_array = np.zeros(len(test_loaders))
        if self.params.trick['ncm_trick'] or self.params.agent in ['ICARL', 'SCR', 'SCP']:
            exemplar_means = {}
            cls_exemplar = {cls: [] for cls in self.old_labels}
            buffer_filled = self.buffer.current_index
            clses=[]
            features_all=[]
            
            for x, y in zip(self.buffer.buffer_img[:buffer_filled], self.buffer.buffer_label[:buffer_filled]):
                cls_exemplar[y.item()].append(x)
            for cls, exemplar in cls_exemplar.items():
                features = []
                # Extract feature for each exemplar in p_y
                for ex in exemplar:
                    feature = self.model.features(ex.unsqueeze(0)).detach().clone()
                    feature = feature.squeeze()
                    feature.data = feature.data / feature.data.norm()  # Normalize
                    features.append(feature)
                    #print("feature.shape:",feature.shape)
                    
                    clses.append(cls)
                    #if len(features_all)==0:
                    #    features_all=feature
                    #else:
                    #    features_all=torch.cat([features_all,feature])

                if len(features) == 0:
                    mu_y = maybe_cuda(torch.normal(0, 1, size=tuple(self.model.features(x.unsqueeze(0)).detach().size())), self.cuda)
                    mu_y = mu_y.squeeze()
                else:
                    features = torch.stack(features)
                    mu_y = features.mean(0).squeeze()
                if len(features_all)==0:
                    features_all=features
                else:
                    features_all=torch.cat([features_all,features])
                mu_y.data = mu_y.data / mu_y.data.norm()  # Normalize
                exemplar_means[cls] = mu_y
        with torch.no_grad():
            clses=torch.tensor(clses).cuda()
            total_test=[]
            if self.params.error_analysis:
                error = 0
                no = 0
                nn = 0
                oo = 0
                on = 0
                new_class_score = AverageMeter()
                old_class_score = AverageMeter()
                correct_lb = []
                predict_lb = []
            for task, test_loader in enumerate(test_loaders):
                acc = AverageMeter()
                for i, (batch_x, batch_y) in enumerate(test_loader):
                    #if len(total_test)==0:
                    #    total_test=batch_x
                    #else:
                    #    total_test=torch.cat([total_test,batch_x])
                    batch_x = maybe_cuda(batch_x, self.cuda)
                    batch_y = maybe_cuda(batch_y, self.cuda)
                    
                    if self.params.trick['ncm_trick'] or self.params.agent in ['ICARL', 'SCR', 'SCP']:
                        feature = self.model.features(batch_x)  # (batch_size, feature_size)
                        for j in range(feature.size(0)):  # Normalize
                            feature.data[j] = feature.data[j] / feature.data[j].norm()
                        #feature = feature.unsqueeze(2)  # (batch_size, feature_size, 1)
                        #means = torch.stack([exemplar_means[cls] for cls in self.old_labels])  # (n_classes, feature_size)

                        #old ncm
                        #means = torch.stack([means] * batch_x.size(0))  # (batch_size, n_classes, feature_size)
                        #means = means.transpose(1, 2)
                        #feature = feature.expand_as(means)  # (batch_size, feature_size, n_classes)
                        #dists = (feature - means).pow(2).sum(1).squeeze()  # (batch_size, n_classes)
                        #_, pred_label = dists.min(1)
                        # may be faster
                        # feature = feature.squeeze(2).T
                        # _, preds = torch.matmul(means, feature).max(0)
                        #################modify
                        nClusters = len(np.unique(clses.cpu()))
                        print("nClusters:",nClusters)
                        h_train = features_all
                        prob_train = one_hot(clses, nClusters).cuda()
                        print("prob_train.shape:",prob_train.shape)
                        protos = self._compute_protos(h_train, prob_train)
                        print("protos.shape:",protos.shape)#[B, nClusters, D]
                        bsize = h_train.size()[0]
                        radii = Variable(torch.ones(bsize, nClusters)).cuda() * torch.exp(self.log_sigma_l)
                        #deal with semi-supervised data
                        '''
                        if batch_y is not None:
                            h_unlabel = feature
                            print("h_train.shape,h_unlabel.shape",h_train.shape,h_unlabel.shape)
                            ###h_all = torch.cat([h_train, h_unlabel], dim=1)
                            h_all = torch.cat([h_train, h_unlabel])
                            print("h_all.shape:",h_all.shape)
                            #perform some clustering steps
                            #compute_logits_radii(cluster_centers, data, radii, prior_weight=1.)
                            for ii in range(1):
                                prob_unlabel = assign_cluster_radii(protos, h_unlabel, radii)
                                prob_unlabel_nograd = Variable(prob_unlabel.data, requires_grad=False).cuda()
                                prob_all = torch.cat([prob_train, prob_unlabel_nograd])
                                protos = self._compute_protos(h_all, prob_all)
                        '''
                        logits = compute_logits_radii(protos, features_all, radii)
                        labels = clses 
                        _, y_pred = torch.max(logits, dim=2)
                        #loss = F.cross_entropy(logits.squeeze(), batch.y_test.squeeze())
                        #acc_val = torch.eq(y_pred.squeeze(), batch_y.squeeze()).float().mean().item()
                        acc_val = torch.eq(y_pred.squeeze(), clses.squeeze()).float().mean().item()
                        #correct_cnt = (np.array(self.old_labels)[
                        #                   pred_label.tolist()] == batch_y.cpu().numpy()).sum().item() / batch_y.size(0)
                        correct_cnt =acc_val
                    else:
                        logits = self.model.forward(batch_x)
                        _, pred_label = torch.max(logits, 1)
                        correct_cnt = (pred_label == batch_y).sum().item()/batch_y.size(0)

                    if self.params.error_analysis:
                        correct_lb += [task] * len(batch_y)
                        for i in pred_label:
                            predict_lb.append(self.class_task_map[i.item()])
                        if task < self.task_seen-1:
                            # old test
                            total = (pred_label != batch_y).sum().item()
                            wrong = pred_label[pred_label != batch_y]
                            error += total
                            on_tmp = sum([(wrong == i).sum().item() for i in self.new_labels_zombie])
                            oo += total - on_tmp
                            on += on_tmp
                            old_class_score.update(logits[:, list(set(self.old_labels) - set(self.new_labels_zombie))].mean().item(), batch_y.size(0))
                        elif task == self.task_seen -1:
                            # new test
                            total = (pred_label != batch_y).sum().item()
                            error += total
                            wrong = pred_label[pred_label != batch_y]
                            no_tmp = sum([(wrong == i).sum().item() for i in list(set(self.old_labels) - set(self.new_labels_zombie))])
                            no += no_tmp
                            nn += total - no_tmp
                            new_class_score.update(logits[:, self.new_labels_zombie].mean().item(), batch_y.size(0))
                        else:
                            pass
                    acc.update(correct_cnt, batch_y.size(0))
                acc_array[task] = acc.avg()
        print(acc_array)
        if self.params.error_analysis:
            self.error_list.append((no, nn, oo, on))
            self.new_class_score.append(new_class_score.avg())
            self.old_class_score.append(old_class_score.avg())
            print("no ratio: {}\non ratio: {}".format(no/(no+nn+0.1), on/(oo+on+0.1)))
            print(self.error_list)
            print(self.new_class_score)
            print(self.old_class_score)
            self.fc_norm_new.append(self.model.linear.weight[self.new_labels_zombie].mean().item())
            self.fc_norm_old.append(self.model.linear.weight[list(set(self.old_labels) - set(self.new_labels_zombie))].mean().item())
            self.bias_norm_new.append(self.model.linear.bias[self.new_labels_zombie].mean().item())
            self.bias_norm_old.append(self.model.linear.bias[list(set(self.old_labels) - set(self.new_labels_zombie))].mean().item())
            print(self.fc_norm_old)
            print(self.fc_norm_new)
            print(self.bias_norm_old)
            print(self.bias_norm_new)
            with open('confusion', 'wb') as fp:
                pickle.dump([correct_lb, predict_lb], fp)
        return acc_array
