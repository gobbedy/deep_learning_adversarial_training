#!/usr/bin/env python

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_rescheduler
import foolbox
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.modules.module import Module
from torch.nn import functional as F
#from torch.nn.functional import _Reduction
import argparse
from torch._six import int_classes as _int_classes
from torch.utils.data.sampler import Sampler, RandomSampler, BatchSampler
import math

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset_directory", type=str, help="directory where dataset resides", required=True)
args = parser.parse_args()

# original skeleton of code borrowed from:
# https://www.kaggle.com/negation/pytorch-logistic-regression-tutorial

# for imdb tutorial, used this tutorial:
# https://github.com/bentrevett/pytorch-sentiment-analysis


# Fixed seed for debugging
SEED = 1234

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class DataBatchSampler(Sampler):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    # TODO 0: add to github and clean up
    # NOTE: we now sample by class weight, but rounded down to nearest integer, aka "minimum class batch size"
    # this leaves remaining samples (code finds exactly how many). we pick num_remaining_samples_per_batch
    # of these remaining num_remaining_samples, on a purely probalistic basis of how many of each are left
    # So, 1. We shuffle the entire epoch, 2. We pick num_remaining_samples_per_batch[class_idx] from each class
    # from the epoch and sent to a sequential sampler, 3. Send the rest of the samples to sequential samplers for
    # each class, as before (except before we used random samplers which are now pointless)
    # TODO 1: this means in the last few batches there may be a batch with zero for some classes, then
    # it must be remediated in the code -- ie if the nearest neighbor class doesn't exist in the batch,
    # we just discard both the input sample altogether
    def __init__(self, dataset, batch_size, drop_last):
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integeral value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))

        self.batch_size = batch_size
        self.data_source = dataset.train_data # TODO: catch if train_data doesn't exist
        self.labels_source = dataset.train_labels #TODO: catch if train_labels doesn't exist
        self.drop_last = drop_last

        self.unique_labels = torch.unique(self.labels_source, sorted=True)
        self.num_classes = len(self.unique_labels)
        self.class_indices_batch_sampler_iterators = [None] * num_classes
        self.class_sample_indices = [None] * num_classes
        self.class_batch_minimum_size = [None] * num_classes
        #self.class_base_indices = [None] * self.num_classes
        #self.class_remaining_indices = [None] * self.num_classes
        #self.class_samples = [None] * num_classes

        self.unpadded_batch_size = 0
        for class_idx, unique_label in enumerate(self.unique_labels):

            # indices of class samples in the dataset as a whole
            self.class_sample_indices[class_idx] = torch.nonzero(self.labels_source == unique_label).view(-1)

            # minimum number of samples of this class in a batch
            class_weight = len(self.class_sample_indices[class_idx]) / len(self.data_source)
            self.class_batch_minimum_size[class_idx] = math.floor(batch_size * class_weight)

            # sum of all "minimum class sizes"
            self.unpadded_batch_size += self.class_batch_minimum_size[class_idx]

        '''
        unpadded_batch_size=0
        for class_idx, unique_label in enumerate(unique_labels):
            #self.class_samples[class_idx] = self.data_source[self.labels_source == unique_label]

            # indices of class samples in the dataset as a whole
            self.class_sample_indices[class_idx] = torch.nonzero(self.labels_source == unique_label)
            class_sample_indices_sampler = RandomSampler(self.class_sample_indices[class_idx])

            # OLD- REMOVE COMMENT SOON # TODO 1: handle when batch_size/num_classes is not a nice integer
            # OLD- REMOVE COMMENT SOON # TODO 2: handle when classes are not equi-distributed (then each sampler won't grab exactly batch_size/num_classes samples)

            class_weight = self.class_sample_indices[class_idx].size() / self.data_source.size()
            class_batch_minimum_size = math.floor(batch_size * class_weight)
            class_indices_batch_sampler = BatchSampler(class_sample_indices_sampler, class_batch_minimum_size, drop_last)

            # iterates over class sample indices
            self.class_indices_batch_sampler_iterators[class_idx] = iter(class_indices_batch_sampler)

            unpadded_batch_size += class_batch_minimum_size

        # remaining samples after class sampling
        num_remaining_samples_per_batch = batch_size - unpadded_batch_size
        num_remaining_samples = num_remaining_samples_per_batch * len(self) # note len(self) same as num_batches
        '''

    def __iter__(self):

        num_batches = len(self)
        class_base_indices = [None] * self.num_classes
        #self.class_epoch_indices = [None] * self.num_classes

        remaining_indices = torch.empty(0, dtype=torch.long)
        for class_idx, unique_label in enumerate(self.unique_labels):
            # self.class_samples[class_idx] = self.data_source[self.labels_source == unique_label]

            # TODO: move top 3 lines to init?
            class_size = len(self.class_sample_indices[class_idx])


            # how many samples will be used to meet minimum batch size for this class, aka "base" samples
            class_num_base_samples = self.class_batch_minimum_size[class_idx] * num_batches

            # how many samples will remaining after base samples
            #class_num_remaining_samples = class_size - class_num_base_samples



            # this epoch's random samples for this class
            class_epoch_indices = self.class_sample_indices[class_idx][torch.randperm(class_size)]

            class_base_indices[class_idx] = class_epoch_indices[0:class_num_base_samples]

            #self.class_remaining_indices[class_idx] = class_epoch_indices[class_num_base_samples:class_size]

            remaining_indices = torch.cat((remaining_indices, class_epoch_indices[class_num_base_samples:class_size]))

        remaining_indices = remaining_indices[torch.randperm(len(remaining_indices))].tolist()

        for batch_idx in range(len(self)):

            batch_indices = []
            for class_idx, class_indices_batch_sampler_iterator in enumerate(self.class_indices_batch_sampler_iterators):

                # randomly sample from the class indices
                # 'class_indices_batch_indices' is a batch of indices sampled from the class_sample_indices array
                # eg for MNIST with batch size 200, for each class this returns 20 indices between 0 and 5999 since
                # there are 60,000 training samples
                #class_indices_batch_indices = next(class_indices_batch_sampler_iterator)
                class_batch_base_size = self.class_batch_minimum_size[class_idx]
                #TODO: make this a list instead of tensor? since batch_indices is a list
                class_batch_base_indices = class_base_indices[class_idx][batch_idx*class_batch_base_size:(batch_idx+1)*class_batch_base_size].tolist()

                batch_indices.extend(class_batch_base_indices)


                # batch of sample indices -- now indices from the original dataset
                #class_sample_batch_indices = self.class_sample_indices[class_idx][class_indices_batch_indices]
                #batch_indices.extend(class_sample_batch_indices)

            num_remaining_in_batch = batch_size - self.unpadded_batch_size
            remaining_batch_indices = remaining_indices[batch_idx*num_remaining_in_batch:(batch_idx+1)*num_remaining_in_batch]

            batch_indices.extend(remaining_batch_indices)


            #print("batch_idx: " + str(batch_idx))
            #print("len(self): " + str(len(self)))
            yield batch_indices

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size



class CrossEntropyLoss_SoftLabels(Module):
    def __init__(self, dim=None):
        super(CrossEntropyLoss_SoftLabels, self).__init__()
        self.dim = dim

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    # predictions = output layer
    # soft_targets = labels
    # reduction this is what the mean does
    # adapted from: https://discuss.pytorch.org/t/how-should-i-implement-cross-entropy-loss-with-continuous-target-outputs/10720/18
    # Hongyi_Zhang answer
    def forward(self, predictions, soft_targets):
        logsoftmax = F.log_softmax(predictions, self.dim, _stacklevel=5)
        #return torch.mean(torch.sum(- soft_targets * logsoftmax, self.dim))
        return torch.mean(torch.sum(- soft_targets * logsoftmax, self.dim))

if __name__ == '__main__':

    '''
    def cross_entropy(pred, soft_targets):
        logsoftmax = nn.LogSoftmax(dim=0)
        return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 0))

    p_list = torch.tensor([0.3, 0.3, 0.4])
    q_list = torch.tensor([0.2, 0.7, 0.1])

    softmax = nn.LogSoftmax(dim=0)
    softmaxed_q_list = softmax(q_list)
    print("LogSoftmax 1 using pytorch method:")
    print(softmaxed_q_list)

    print("Cross entropy 1 using pytorch method:")
    cross_ent = cross_entropy(q_list, p_list)
    print(cross_ent)


    p_list = torch.tensor([0.6, 0.4, 0.0])
    q_list = torch.tensor([0.5, 0.0, 0.5])

    softmax = nn.LogSoftmax(dim=0)
    softmaxed_q_list = softmax(q_list)
    print("LogSoftmax 2 using pytorch method:")
    print(softmaxed_q_list)


    print("Cross entropy 2 using pytorch method:")
    cross_ent = cross_entropy(q_list, p_list)
    print(cross_ent)

    p_list = torch.tensor([[0.3, 0.3, 0.4],[0.6, 0.4, 0.0]])
    q_list = torch.tensor([[0.2, 0.7, 0.1],[0.5, 0.0, 0.5]])


    print("Cross entropy using new method:")
    new_CrossEntropy = CrossEntropyLoss_SoftLabels(dim=1)
    cross_ent = new_CrossEntropy(q_list, p_list)
    print(cross_ent)
    '''

    '''
    def cross_entropy(pred, soft_targets):
        logsoftmax = nn.LogSoftmax(dim=0)
        return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 0))


    p_list = torch.tensor([0.3, 0.3, 0.4])
    q_list = torch.tensor([0.2, 0.7, 0.1])

    cross_ent = 0
    softmax = nn.Softmax(dim=0)
    softmaxed_q_list = softmax(q_list)
    print("Softmax using pytorch method:")
    print(softmaxed_q_list)
    exp_sum = 0
    for idx, q in enumerate(q_list):
        exp_sum += np.exp(q)

    for idx, q in enumerate(q_list):
        softmaxed_q_list[idx] = np.exp(q) / exp_sum

    print("Softmax using hand method:")
    print(softmaxed_q_list)

    for idx, p in enumerate(p_list):
        cross_ent -= p_list[idx] * np.log(softmaxed_q_list[idx])

    print("Cross entropy using hand method:")
    print(cross_ent)

    print("Cross entropy using pytorch method:")
    cross_ent = cross_entropy(q_list, p_list)
    print(cross_ent)


    print("Cross entropy using new method:")
    new_CrossEntropy = CrossEntropyLoss_SoftLabels(dim=0)
    cross_ent = new_CrossEntropy(q_list, p_list)
    print(cross_ent)

    '''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = 'cpu'

    # Hyper Parameters

    # input to the linear layer is total number of pixels, represented in grayscale as a number from 0-255
    input_size = 28*28

    # number of labels -- here labels are 0-9
    num_classes = 10

    # epoch = forward and backward pass over all training examples (more exactly: over the same number of samples
    # as the training dataset since with SGD we may pick random samples and technically not use all training examples)
    num_epochs = 40

    # heuristically we pick a batch size with an adequate compromise between accurate gradient (larger sample size of
    # datapoints is better) and speed of computation (smaller batch size is better)
    batch_size = 100

    # tradeoff: precision of
    learning_rate = 0.1

    # set of 60,000 28*28 images + 60,000 digit labels


    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

    try:

        # note: transforms.ToTensor() converts PIL format to tensor, and also
        # normalizes grayscale values in [0-255] range to [0-1]
        # this normalization allows faster learning as signmoid function more likely to be in roughly linear range
        # TODO: ask prof Mao that this is correct interpretation
        train_dataset = dsets.MNIST(root=args.dataset_directory,
                                train=True,
                                transform=transforms.ToTensor(),
                                download=False)

    except RuntimeError:
        train_dataset = dsets.MNIST(root=args.dataset_directory,
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)

    # set of 10,000 28*28 images + 10,000 digit labels

    test_dataset = dsets.MNIST(root=args.dataset_directory,
                               train=False,
                               transform=transforms.ToTensor())


    # dataset loader: generator that yields input samples each based on rules: yields "batch_size" samples
    # on each call and yields random samples or works its way sequentially thru training dataset depending on "shuffle"
    train_data_batch_sampler = DataBatchSampler(dataset=train_dataset, batch_size=batch_size, drop_last=False)
    '''
    batch_sample_iterator = iter(train_data_batch_sampler)
    indices = []
    for idx, batch_indices in enumerate(batch_sample_iterator):
        indices.extend(batch_indices)
        print(idx)

    indices = np.unique(np.array(indices))
    real_indices = np.arange(60000)

    np.array_equal(indices, real_indices)
    '''

    #'''
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_sampler=train_data_batch_sampler,
                                               **kwargs)
    #'''

    '''
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              **kwargs)
    '''

    # batch size is irrelevant for test_loader, could set it to full test dataset
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              **kwargs)



    # Model
    class MLP_Regression(nn.Module):
        def __init__(self, input_size, num_classes):
            super(MLP_Regression, self).__init__()

            # crucial line:
            # calls register_parameters() (via __setattr__), which adds the linear model in the children modules list
            # the linear model registers its own parameters also via register_parameters()
            # later, LogisticRegression obj's self.parameters() will recursively find all its children modules' parameters
            # in our case, this will just be the parametesr of the linear layer

            # In short: this line is critical for keeping track of parameters for gradient descent

            # Note that the below forward function needs to call 'self.linear' and not its own instance of nn.Linear
            # as otherwise the model would lose track of parameters for gradient descent
            #self.linear = nn.Linear(in_features=input_size, out_features=num_classes)

            linear1_output_len=120
            linear2_output_len=84
            linear3_output_len=num_classes

            self.linear1 = nn.Linear(in_features=input_size, out_features=linear1_output_len)
            self.linear2 = nn.Linear(in_features=linear1_output_len, out_features=linear2_output_len)
            self.linear3 = nn.Linear(in_features=linear2_output_len, out_features=linear3_output_len)

        def forward(self, x):

            y1 = F.relu(self.linear1(x))
            y2 = F.relu(self.linear2(y1))
            #out = F.relu(self.linear3(y2))
            out = self.linear3(y2)

            return out


    model = MLP_Regression(input_size, num_classes)
    #model = model.cuda()
    model.to(device)

    # softmax + cross entropy loss
    #criterion = nn.CrossEntropyLoss()
    criterion = CrossEntropyLoss_SoftLabels(dim=1)
    criterion.to(device)

    #criterion_original = nn.CrossEntropyLoss()
    #criterion_original.to(device)

    # Note that technically this is not just Gradient Descent, not Stochastic Gradient Descent
    # What makes it 'stochastic' or 'not stochastic' is whether we use batch_size == dataset size or not
    # In reality, SGD is just a generalized form of GD (where batch size = dateset size), so it is not actually
    # incorrect to call it SGD when doing GD
    # model.parameters() in this case is the linear layer's parameters, aka the 'theta' of our algorithm
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # adaptive learning rate policy -- "schedules" when to decrease learning rate
    scheduler = lr_rescheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)

    #'''
    fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=10)
    #attack = foolbox.attacks.CarliniWagnerL2Attack(fmodel)
    attack = foolbox.attacks.FGSM(fmodel)
    '''
    #train_loader
    im0 = train_dataset.train_data[0]
    lbl0 = train_dataset.train_labels[0]
    
    im0 = im0.float() / 255  # because our model expects values in [0, 1]
    im0 = im0.view(28 * 28)
    im0 = im0.numpy()
    
    lbl0 = lbl0.numpy()
    

    print('label', lbl0)
    print('predicted class', np.argmax(fmodel.predictions(im0)))
    
    # apply attack on source image
    #attack = foolbox.attacks.CarliniWagnerL2Attack(fmodel)
    adversarial = attack(im0, lbl0)
    
    print('adversarial class', np.argmax(fmodel.predictions(adversarial)))


    plt.figure()
    
    plt.subplot(1, 3, 1)
    plt.title('Original')
    plt.imshow(im0.reshape(28, 28))  # division by 255 to convert [0, 255] to [0, 1]
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title('Adversarial')
    plt.imshow(adversarial.reshape(28, 28))  # ::-1 to convert BGR to RGB
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title('Difference')
    difference = (adversarial - im0).reshape(28, 28)
    plt.imshow(difference / abs(difference).max() * 0.2 + 0.5)
    plt.axis('off')
    
    plt.show()
    '''










    # Training the Model
    epoch=0
    while True:
        # train_loader keeps yielding batches until it reaches full epoch

        print("epoch: " + str(epoch))


        # TODO: smarter sampling
        for i, (images, labels) in enumerate(train_loader):

            print("batch: " + str(i))

            # 100 images x 784
            images = images.view(-1, 28 * 28)

            #'''
            #print("BEFORE COMPUTE BATCH OF ADV")
            true_boundary_images=torch.zeros(batch_size, 28*28)
            true_boundary_labels=torch.zeros(batch_size, num_classes)

            # TODO 2: start adversarial training much later in simulation
            # TODO 3: use fast gradient just as a proof of concept to get result
            #        + compare with goodfellow result (ie just train with adversarial)
            # TODO 4: check back with foolbox around dec 11 (3 weeks after they said it would be done in 2 weeks)
            #        for update on batch launch
            # TODO 5: try on Niagara with a single CPU for 10mins; if it reaches any reasonable fraction
            # of 6 batches completes, then we know we can get a huge speedup on Niagara
            # TODO 6: use same parallelization method as used this summer to send across many CPUs on Niagara
            # TODO 7: verify the logic below a second time (?)

            for idx, image in enumerate(images):
                adversarial = attack(image.numpy(), labels[idx].numpy())

                #adversarial = attack(image.numpy(), labels[idx].numpy(), binary_search_steps=9,
                #                     initial_const=1e-3, learning_rate=1e-2)

                # if no adversarial is found, just skip
                if adversarial is None:
                    print("Warning: no adversarial was found. Using original image as adversarial.")
                    adversarial = image.numpy()


                #adversarial_label = torch.from_numpy(fmodel.predictions(adversarial))
                adversarial_label = int(np.argmax(fmodel.predictions(adversarial)))

                # pytorch wants it in tensor format
                adversarial = torch.from_numpy(adversarial)

                # find nearest neighbor in target class
                target_label = adversarial_label
                target_images = images[labels==adversarial_label]

                distances = torch.norm(target_images - adversarial, p=2, dim=1)
                distances_sorted, sorted_indices = torch.sort(distances, 0)

                # TODO: use range(k) instead of zero here if choose k nearest neighbors
                nearest_neighbor_indices = sorted_indices[0]

                # NOTE: nearest_target_neighbors becomes an array instead of single image if choose k nearest neighbors
                nearest_target_neighbors=target_images[nearest_neighbor_indices]


                # find nearest neighbor in source class
                source_label = labels[idx]
                source_images = images[labels==source_label]

                distances = torch.norm(source_images - adversarial, p=2, dim=1)
                distances_sorted, sorted_indices = torch.sort(distances, 0)

                # TODO: use range(k) instead of zero here if choose k nearest neighbors
                nearest_neighbor_indices = sorted_indices[0]

                # TODO: nearest_source_neighbors becomes an array instead of single image if choose k nearest neighbors
                nearest_source_neighbors = source_images[nearest_neighbor_indices]


                # find "true boundary" image

                # TODO: use mean of all nearest neighbors if choose k nearest neighbors (where k<1)
                # aka the code below (has been tested!)
                # (torch.mean(nearest_target_neighbors,0) + torch.mean(nearest_source_neighbors,0)) / 2
                true_boundary_images[idx] = (nearest_target_neighbors + nearest_source_neighbors) / 2

                boundary_label_values = torch.tensor([0.5, 0.5])
                index = torch.tensor([target_label, source_label])
                true_boundary_labels[idx].scatter_(0, index, boundary_label_values)


            images = torch.cat((images, true_boundary_images), 0).to(device)
            #images = images.to(device)

            # above, we made use of 'labels' being integer for indexing;
            # now that we're making labels a

            soft_labels = torch.eye(num_classes)[labels]
            soft_labels = torch.cat((soft_labels, true_boundary_labels), 0).to(device)
            #soft_labels = soft_labels.to(device)

            #'''

            #soft_labels = torch.eye(num_classes)[labels].to(device)
            #images = images.to(device)

            #print("AFTER COMPUTE BATCH OF ADV")

            # 100 labels

            # prevent gradients from accumulating -- they should be computed "fresh" during each batch
            optimizer.zero_grad()

            # linear layer
            outputs = model(images)

            # softmax "layer" + cross entropy loss
            loss = criterion(outputs, soft_labels)

            #loss_original = criterion_original(outputs, labels)

            # computed loss gradients wrt to parameters
            loss.backward()

            # update parameters (here linear layer parameters) using learning rate + gradients
            optimizer.step()

            # TODO: change end condition?
            #if (i + 1) % 600 == 0:
            if True:
                lr=optimizer.param_groups[0]['lr']
                best_loss = scheduler.best
                print('Best end-of-epoch loss: %.7f, LR: %.4f, Epoch: [%d/%d], Step: [%d/%d], Loss: %.7f'
                      % (best_loss, lr, epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.data.item()))

        epoch+=1

        # adapt learning rate
        scheduler.step(loss)
        lr=optimizer.param_groups[0]['lr']

        # stop when loss has stopped decreasing for a long time -- ie when, on 3 separate occasions, it didn't decrease
        # from start-to-end-of-epoch 3 epochs in a row
        # NOTE TO SELF: BETTER CONDITION WOULD BE CHECKING GRADIENT OF LOSS ITSELF
        if lr < 0.001:
            break


    # Test the Model
    correct = 0
    total = 0


    for i, (images, labels) in enumerate(test_loader):


        images = images.view(-1, 28 * 28)
        images = images.to(device)

        labels = labels.to(device)

        # linear layer
        outputs = model(images)


        # note that we do not need softmax layer for the decision process, as softmax does not change the
        # order the selection (merely amplifies the difference between them) -- picking index corresponding to
        # max value is sufficient

        _, predicted = torch.max(outputs.data, 1)
        total += len(labels)
        correct += (predicted == labels).sum()

    print('Accuracy of the model on the 10000 test images: %.3f %%' % (100 * correct.item() / total))