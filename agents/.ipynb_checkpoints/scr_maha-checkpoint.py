import torch
from torch.utils import data
from utils.buffer.buffer import Buffer
from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from utils.setup_elements import transforms_match, input_size_match
from utils.utils import maybe_cuda, AverageMeter
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale
import torch.nn as nn
from collections import OrderedDict
from maha_dist_util.utils import 


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
        self.task_representation = None
        self.class_representations = OrderedDict()  # Dictionary mapping class label (integer) to encoded representation
        self.z_g_dim = 40
        self.class_precision_matrices=None
        self.adaptive_module=None
        
        num_maps_per_layer = [64, 128, 256, 512]
        num_blocks_per_layer = [2, 2, 2, 2]
        num_initial_conv_maps = 64
        
    def mean_pooling(x):
        return torch.mean(x, dim=0, keepdim=True)
    def create_adaptive_module():
        self.feature_adaptation_network = FilmAdaptationNetwork(
                layer=FilmLayerNetwork,
                num_maps_per_layer=num_maps_per_layer,
                num_blocks_per_layer=num_blocks_per_layer,
                z_g_dim=self.z_g_dim
            )
        return self.feature_adaption_network
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
            class_rep = mean_pooling(class_features)
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
        acc_batch = AverageMeter()

        for ep in range(self.epoch):
            for i, batch_data in enumerate(train_loader):
                # batch update
                batch_x, batch_y = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)

                for j in range(self.mem_iters):
                    mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)

                    if mem_x.size(0) > 0:
                        mem_x = maybe_cuda(mem_x, self.cuda)
                        mem_y = maybe_cuda(mem_y, self.cuda)
                        #combined_batch = torch.cat((mem_x, batch_x))
                        #combined_labels = torch.cat((mem_y, batch_y))
                        #combined_batch_aug = self.transform(combined_batch)
                        ##############################################################
                        context_features =  self.model.forward(self.transform(mem_x))
                        target_features = self.model.forward(self.transform(batch_x))
                        self.z_g_dim=int(context_features.shape[0])
                        self.adaptive_module = self.create_adaptive_module()
                        self.adaptive_module = self.adaptive_module.cuda()
                        print("encoder_out:",self.model.forward(combined_batch).shape)
                        self._build_class_reps_and_covariance_estimates(context_features, mem_y)
                        class_means = torch.stack([self.class_representations[label] for label in label_set]).squeeze(1)
                        class_precision_matrices = torch.stack([self.class_precision_matrices[label] for label in label_set])
                        class_means = torch.stack(list(self.class_representations.values())).squeeze(1)
                        class_precision_matrices = torch.stack(list(self.class_precision_matrices.values()))
                        number_of_classes = class_means.size(0)
                        number_of_targets = target_features.size(0)
                        """
                        Calculating the Mahalanobis distance between query examples and the class means
                        including the class precision estimates in the calculations, reshaping the distances
                        and multiplying by -1 to produce the sample logits
                        """
                        repeated_target = target_features.repeat(1, number_of_classes).view(-1, class_means.size(1))
                        repeated_class_means = class_means.repeat(number_of_targets, 1)
                        repeated_difference = (repeated_class_means - repeated_target)
                        repeated_difference = repeated_difference.view(number_of_targets, number_of_classes, repeated_difference.size(1)).permute(1, 0, 2)
                        first_half = torch.matmul(repeated_difference, class_precision_matrices)
                        sample_logits = torch.mul(first_half, repeated_difference).sum(dim=2).transpose(1,0) * -1
                        #################################################
                        #features = torch.cat([self.model.forward(combined_batch).unsqueeze(1), self.model.forward(combined_batch_aug).unsqueeze(1)], dim=1)
                        loss = self.criterion(features, combined_labels)
                        losses.update(loss, batch_y.size(0))
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
        self.after_train()
