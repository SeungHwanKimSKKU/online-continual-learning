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
from collections import OrderedDict
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
        self.NUM_SAMPLES=10
        self.counter=0
        self.task_representation = None
        self.class_representations = OrderedDict()  # Dictionary mapping class label (integer) to encoded representation
        self.class_precision_matrices = OrderedDict()
        self.z_g_dim = 40
        #self.class_precision_matrices=0
        self.adaptive_module=None
        self.NUM_SAMPLES=1
        self.num_maps_per_layer = [64, 128, 256, 512]
        self.num_blocks_per_layer = [2, 2, 2, 2]
        self.num_initial_conv_maps = 64
        self.feature_adaption_network=None
    def mean_pooling(self,x):
        return torch.mean(x, dim=0, keepdim=True)
    def create_adaptive_module(self):
        self.feature_adaptation_network = FilmAdaptationNetwork(
                layer=FilmLayerNetwork,
                num_maps_per_layer=self.num_maps_per_layer,
                num_blocks_per_layer=self.num_blocks_per_layer,
                z_g_dim=self.z_g_dim
            )
        return self.feature_adaptation_network

    def estimate_cov(self, examples, rowvar=False, inplace=False):
        """
        Function based on the suggested implementation of Modar Tensai
        and his answer as noted in: https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/5
        Estimate a covariance matrix given data.
        Covariance indicates the level to which two variables vary together.
        If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
        then the covariance matrix element `C_{ij}` is the covariance of
        `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.
        Args:
            examples: A 1-D or 2-D array containing multiple variables and observations.
                Each row of `m` represents a variable, and each column a single
                observation of all those variables.
            rowvar: If `rowvar` is True, then each row represents a
                variable, with observations in the columns. Otherwise, the
                relationship is transposed: each column represents a variable,
                while the rows contain observations.
        Returns:
            The covariance matrix of the variables.
        """
        if examples.dim() > 2:
            raise ValueError('m has more than 2 dimensions')
        if examples.dim() < 2:
            examples = examples.view(1, -1)
        if not rowvar and examples.size(0) != 1:
            examples = examples.t()
        factor = 1.0 / (examples.size(1) - 1)
        if inplace:
            examples -= torch.mean(examples, dim=1, keepdim=True)
        else:
            examples = examples - torch.mean(examples, dim=1, keepdim=True)
        examples_t = examples.t()
        return factor * examples.matmul(examples_t).squeeze()
    @staticmethod
    def _extract_class_indices(labels, which_class):
        class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
        class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
        return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector

    def distribute_model(self):
        self.feature_extractor.cuda(1)
        self.feature_adaptation_network.cuda(1)

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

    def _build_class_reps_and_covariance_estimates(self, context_features, context_labels):
        """
        Construct and return class level representations and class covariance estimattes for each class in task.
        :param context_features: (torch.tensor) Adapted feature representation for each image in the context set.
        :param context_labels: (torch.tensor) Label for each image in the context set.
        :return: (void) Updates the internal class representation and class covariance estimates dictionary.
        """

        """
        Calculating a task level covariance estimate using the provided function.
        """
        task_covariance_estimate = self.estimate_cov(context_features)
        for c in torch.unique(context_labels):
            # filter out feature vectors which have class c
            class_features = torch.index_select(context_features, 0, self._extract_class_indices(context_labels, c))
            # mean pooling examples to form class means
            class_rep = self.mean_pooling(class_features)
            # updating the class representations dictionary with the mean pooled representation
            self.class_representations[c.item()] = class_rep
            """
            Calculating the mixing ratio lambda_k_tau for regularizing the class level estimate with the task level estimate."
            Then using this ratio, to mix the two estimate; further regularizing with the identity matrix to assure invertability, and then
            inverting the resulting matrix, to obtain the regularized precision matrix. This tensor is then saved in the corresponding
            dictionary for use later in infering of the query data points.
            """
            lambda_k_tau = (class_features.size(0) / (class_features.size(0) + 1))
            self.class_precision_matrices[c.item()] = torch.inverse((lambda_k_tau * self.estimate_cov(class_features)) + ((1 - lambda_k_tau) * task_covariance_estimate) \
                    + torch.eye(class_features.size(1), class_features.size(1)).cuda(0))
    def estimate_cov(self, examples, rowvar=False, inplace=False):
        """
        Function based on the suggested implementation of Modar Tensai
        and his answer as noted in: https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/5
        Estimate a covariance matrix given data.
        Covariance indicates the level to which two variables vary together.
        If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
        then the covariance matrix element `C_{ij}` is the covariance of
        `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.
        Args:
            examples: A 1-D or 2-D array containing multiple variables and observations.
                Each row of `m` represents a variable, and each column a single
                observation of all those variables.
            rowvar: If `rowvar` is True, then each row represents a
                variable, with observations in the columns. Otherwise, the
                relationship is transposed: each column represents a variable,
                while the rows contain observations.
        Returns:
            The covariance matrix of the variables.
        """
        if examples.dim() > 2:
            raise ValueError('m has more than 2 dimensions')
        if examples.dim() < 2:
            examples = examples.view(1, -1)
        if not rowvar and examples.size(0) != 1:
            examples = examples.t()
        factor = 1.0 / (examples.size(1) - 1)
        if inplace:
            examples -= torch.mean(examples, dim=1, keepdim=True)
        else:
            examples = examples - torch.mean(examples, dim=1, keepdim=True)
        examples_t = examples.t()
        return factor * examples.matmul(examples_t).squeeze()

    @staticmethod
    def _extract_class_indices(labels, which_class):
        class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
        class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
        return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector

    def distribute_model(self):
        self.feature_extractor.cuda(1)
        self.feature_adaptation_network.cuda(1)
    def split_first_dim_linear(self,x, first_two_dims):
        """
        Undo the stacking operation
        """
        x_shape = x.size()
        new_shape = first_two_dims
        if len(x_shape) > 1:
            new_shape += [x_shape[-1]]
        return x.view(new_shape)
    def evaluate(self, test_loaders):
        self.model.eval()
        acc_array = np.zeros(len(test_loaders))
        if self.params.trick['ncm_trick'] or self.params.agent in ['ICARL', 'SCR', 'SCP']:
            self.counter+=1
            exemplar_means = {}
            cls_exemplar = {cls: [] for cls in self.old_labels}
            buffer_filled = self.buffer.current_index
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
                    clses.append(cls)
                if len(features) == 0:
                    mu_y = maybe_cuda(torch.normal(0, 1, size=tuple(self.model.features(x.unsqueeze(0)).detach().size())), self.cuda)
                    mu_y = mu_y.squeeze()
                else:
                    features = torch.stack(features)
                    mu_y = features.mean(0).squeeze()
                mu_y.data = mu_y.data / mu_y.data.norm()  # Normalize
                exemplar_means[cls] = mu_y
        with torch.no_grad():
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
                feature_total=[]
                acc = AverageMeter()
                for i, (batch_x, batch_y) in enumerate(test_loader):
                    batch_x = maybe_cuda(batch_x, self.cuda)
                    batch_y = maybe_cuda(batch_y, self.cuda)
                    if self.params.trick['ncm_trick'] or self.params.agent in ['ICARL', 'SCR', 'SCP']:
                        feature = self.model.features(batch_x)  # (batch_size, feature_size)
                        for j in range(feature.size(0)):  # Normalize
                            feature.data[j] = feature.data[j] / feature.data[j].norm()
                        features_total = torch.cat([features,mu_y])
                        print("feature_total.shape:",feature_total.shape)
                        clses=torch.tensor(clses)
                        total_class=torch.cat(clses,batch_y)
                        print("total_classes.shape:"total_class.shape)
                        ####feature = feature.unsqueeze(2)  # (batch_size, feature_size, 1)
                        #modi start
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
                        print("batch_y:",batch_y)
                        print("self.old_labels",self.old_labels)
                        self._build_class_reps_and_covariance_estimates(feature,batch_y)
                        class_means = torch.stack([self.class_representations[label] for label in batch_y.cpu().detach().numpy().tolist()]).squeeze(1)
                        class_precision_matrices = torch.stack([self.class_precision_matrices[label] for label in batch_y.cpu().detach().numpy().tolist()])
                        class_means = torch.stack(list(self.class_representations.values())).squeeze(1)
                        class_precision_matrices = torch.stack(list(self.class_precision_matrices.values()))
                        # grabbing the number of classes and query examples for easier use later in the function
                        number_of_classes = class_means.size(0)
                        number_of_targets = feature.size(0)

                        """
                        Calculating the Mahalanobis distance between query examples and the class means
                        including the class precision estimates in the calculations, reshaping the distances
                        """
                        repeated_target = feature.repeat(1, number_of_classes).view(-1, class_means.size(1))
                        repeated_class_means = class_means.repeat(number_of_targets, 1)
                        repeated_difference = (repeated_class_means - repeated_target)
                        repeated_difference = repeated_difference.view(number_of_targets, number_of_classes, repeated_difference.size(1)).permute(1, 0, 2)
                        first_half = torch.matmul(repeated_difference, class_precision_matrices)
                        sample_logits = torch.mul(first_half, repeated_difference).sum(dim=2).transpose(1,0) * -1
                        #output=self.split_first_dim_linear(sample_logits, [self.NUM_SAMPLES, feature.shape[0]])
                        print("sample_logits:",sample_logits.shape,sample_logits)
                        _, pred_label = sample_logits.min(1)
                        #pred_label = pred_label+self.counter
                        print(",pred_label:",pred_label)
                        print("pred_label.shape:",pred_label.shape)
                        #outputs=self.split_first_dim_linear(sample_logits, [self.NUM_SAMPLES, feature.shape[0]])
                        
                        #print("outputs.shape:",outputs.shape)
                        print("batch_y.shape",batch_y.shape)
                        print("np.array(self.old_labels)[pred_label.tolist()]",np.array(self.old_labels)[pred_label.tolist()])
                        print("batch_y.cpu().numpy()",batch_y.cpu().numpy())
                        print("np.array(self.old_labels)[pred_label.tolist()]",np.array(self.old_labels)[pred_label.tolist()]== batch_y.cpu().numpy())

                        correct_cnt = (np.array(self.old_labels)[
                                           pred_label.tolist()] == batch_y.cpu().numpy()).sum().item() / batch_y.size(0)
                        
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
                            total = (outputs != batch_y).sum().item()
                            error += total
                            wrong = pred_label[pred_label != batch_y]
                            no_tmp = sum([(wrong == i).sum().item() for i in list(set(self.old_labels) - set(self.new_labels_zombie))])
                            no += no_tmp
                            nn += total - no_tmp
                            new_class_score.update(logits[:, self.new_labels_zombie].mean().item(), batch_y.size(0))
                        else:
                            pass
                    acc.update(correct_cnt, batch_y.size(0))
                    self.counter+=1
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


