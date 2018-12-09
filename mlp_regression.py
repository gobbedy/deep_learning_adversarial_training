#!/usr/bin/env python

import torch
#import torch.nn as nn
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
import dispy
import os
import functools
from mlp_model import MLP_Regression
import traceback
import os.path
from foolbox import Adversarial

ME_DIR = os.path.dirname(os.path.realpath(__file__))


#################################################################################################
############## 1. START OF IDEAS TO IMPROVE ACCURACY OR BUG FIXES ###############################
#################################################################################################

# TODO: new non-convexity algorithm -- why so many targets are "behind" the adversarial tho? doesnt seem right
# 1. Find xa as before
# 2. Find source, target as before
# 3. Note distance Rt between xa and target -- we know there is a half "n-sphere" around xa of radius R with no target class samples in it
# 4. Move from xa along direction from ORIGINAL to xa in small increments until hit boundary of sphere
#    -- at each increment test if still classified as target class.
#    3 scenarios: 1) every increment classified as target until sphere boundary, then we pick midpoint and deem it ground truth boundary -- use soft labels [NOTE: we could also deep it something closer to original class, but hard to quantify how much)
#                 2) at some increment, we hit original source class -- then we pick midpoint between this increment and xa and deem it 100% source class ("fill in")
#                 3) at some incrememnt, we hit a 3rd class -- DO WE JUST IGNORE THIS? No because if scenario 1 becomes "every increment classified as non-source class" -- then what do we pick as soft labels at sphere boundary?
#                    **so here can pick middle between increment and xa, and deem it 50%/50% between original class and 3rd class**


# 1. Find xa, xs, xt as before
# 2. Find ra, rt
# 3. Find xm




# "a" is vector from adversarial to target nearest neighbor
# a = nearest_target_neighbor - adversarial
#
# # "b" is vector from vector from original sample to adversarial
# b = adversarial - image

# R = torch.norm(a)




# TODO: fix test accuracy (why is last decimal always zero?)


# TODO: graph original loss, boundary loss, test accuracy (in latex?)
# TODO: save model, optimizer and scheduler to allow relaunching simulation
# TODO: switch to adam optimizer
# TODO: lower learning rate when switching to improve accuracy? -- for now just not changing it instead of increasing


#################################################################################################
############## END OF SPEEDUP IDEAS NOTES (LOWER PRIORITY FOR NOW ###############################
#################################################################################################





#################################################################################################
############## 2. START OF SPEEDUP IDEAS OR NON-ESSENTIAL FIXES (LOWER PRIORITY FOR NOW) ########
#################################################################################################

# TODO: switch back to boundary attack after figuring out this bug!! https://github.com/bethgelab/foolbox/issues/243
# TODO: do nearest neighbor search inside dispy compute function
# TODO: improve initialization of boundary attack after 1st epoch -- could drastically improve speed
# TODO: get rid of DataBatchSampler since using nearest neighbor for entire dataset
# TODO: compute class subsets only once
# TODO: ask pgiri if it's possible to launch a dispy job on multiple CPUs (ie request multiple CPUs per dispy job)
    # alternatively, manually tell dispynode how many CPUs there are, and "trick it" by telling it there are few CPUs
# TODO: use k-nearest neighbor instead of nearest neighbor (slower, less trainning-accurate, but maybe testing accurate)
# TODO: if keep batch neareset neighbor, discard when sample has no target neighbor(see the TODOs in DataBatchSampler)
# TODO: make sure that boundary attack returns original images quickly when point is already adversarial. no reason to suspect
#       if there's an issue refer to aug 29 comment: https://github.com/bethgelab/foolbox/issues/63

#################################################################################################
############## END OF SPEEDUP IDEAS NOTES (LOWER PRIORITY FOR NOW ###############################
#################################################################################################






#################################################################################################
############## 3. START OF OMP_NUM_THREADS DEBUGGING NOTES ######################################
#################################################################################################

# TODO: OMP_NUM_THREADS ISSUE MAY BE SOLVED -- NOW QUESTION IS WHY SOMETIMES DOESNT START
# two working ones so far
# /scratch/y/yymao/gobbedy/deep_learning/deep_learning/adversarial_project/Nov26_014130
# /scratch/y/yymao/gobbedy/deep_learning/deep_learning/adversarial_project/Nov26_020753

# ASK pgiri these questions:
# figure out OMP issue: perhaps I can manually tell dispynode that it has fewer nodes?
# alternatively, perhaps I can request multiple CPUs per job?
# OMP: Error #34: System unable to allocate necessary resources for OMP thread:
# OMP: System error #11: Resource temporarily unavailable
# OMP: Hint: Try decreasing the value of OMP_NUM_THREADS.



# TODO: put OMP_NUM_THREADS=1 back, not sure when i removed it!
# /scratch/y/yymao/gobbedy/deep_learning/deep_learning/adversarial_project/Nov26_015545
# Above never launched, trying again:
# /scratch/y/yymao/gobbedy/deep_learning/deep_learning/adversarial_project/Nov26_020753 -- THIS ONE WORKED!

# TODO try setting these to false
# threaded_rnd : bool
# threaded_gen : bool
# trying here: /scratch/y/yymao/gobbedy/deep_learning/deep_learning/adversarial_project/Nov26_014425
# NOTE: WAS REQUESTED WITHOUT OMP_NUM_THREADS=1
# WORKED BUT WAS VERY SLOW

#TODO: try requesting this node since it maybe accepts more threads (/scratch/y/yymao/gobbedy/deep_learning/deep_learning/adversarial_project/Nov26_005856/adversarial_project.log)
# nia0213 (-w nia0213)
# requested it, log will be here when it launches /scratch/y/yymao/gobbedy/deep_learning/deep_learning/adversarial_project/Nov26_014130
# job id is 652576
# NOTE: WAS REQUESTED WITHOUT OMP_NUM_THREADS=1

#################################################################################################
############## END OF OMP_NUM_THREADS DEBUGGING NOTES ###########################################
#################################################################################################



#################################################################################################
############## 3. START ALGORITHM IDEAS #########################################################
#################################################################################################

# TODO: use projection along direction from original sample to classification boundary (aka adversarial)?

# "a" is vector from adversarial to target nearest neighbor
# a = nearest_target_neighbor - adversarial
#
# # "b" is vector from vector from original sample to adversarial
# b = adversarial - image
#
# # we find projection of a onto b, ie projection of nearest target in direction of adversarial
# projection = (torch.dot(a,b) / torch.norm(b)**2) * b
#
# true_boundary_images[idx] = projection + adversarial

#################################################################################################
############## END ALGORITHM IDEAS ##############################################################
#################################################################################################




parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset_directory", type=str, help="directory where dataset resides. Also contains pre-trained model.", required=True)
#parser.add_argument('-m','--compute_nodes_pythonic', nargs='+', help='python-friendly compute node names', required=True)
args = parser.parse_args()

# original skeleton of code borrowed from:
# https://www.kaggle.com/negation/pytorch-logistic-regression-tutorial

# for imdb tutorial, used this tutorial:
# https://github.com/bentrevett/pytorch-sentiment-analysis


# Fixed seed for debugging
#SEED = 1234
#torch.manual_seed(SEED)
#torch.cuda.manual_seed(SEED)
#torch.backends.cudnn.deterministic = True

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

            num_remaining_in_batch = batch_size - self.unpadded_batch_size
            remaining_batch_indices = remaining_indices[batch_idx*num_remaining_in_batch:(batch_idx+1)*num_remaining_in_batch]

            batch_indices.extend(remaining_batch_indices)

            yield batch_indices

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size


def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        #stdv = 1. / math.sqrt(self.weight.size(1))
        stdv = 1. / 10
        m.weight.data.uniform_(-stdv, stdv)
        if m.bias is not None:
            m.bias.data.uniform_(-stdv, stdv)

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


def dispy_setup_adversarial(dirpath):

    global directory_path
    directory_path = dirpath

    global os
    import os
    os.environ["OMP_NUM_THREADS"] = "1"

    #global torch, numpy, traceback
    #import torch
    #import numpy
    #import foolbox
    #import sys
    #import traceback
    #sys.path.append(dirpath) # add to python path instead?
    #from mlp_model import MLP_Regression

    #global attack, images, labels
    #model = torch.load(dirpath + '/model.torch')
    #images = torch.load(dirpath + '/images.torch')
    #labels = torch.load(dirpath + '/labels.torch')


    #fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=10)
    #attack = foolbox.attacks.BoundaryAttack(fmodel)


    #attack = foolbox.attacks.CarliniWagnerL2Attack(fmodel)
    #attack = foolbox.attacks.FGSM(fmodel)

    return 0

def dispy_compute_adversarial(i):

    try:
        import torch
        import foolbox
        import sys
        import numpy as np
        #import os


        sys.path.append(directory_path)  # add to python path instead?
        from mlp_model import MLP_Regression

        model = torch.load(directory_path + '/model.torch')
        images = torch.load(directory_path + '/images.torch')
        labels = torch.load(directory_path + '/labels.torch')

        fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=10, device='cpu')
        attack = foolbox.attacks.BoundaryAttack(fmodel)
        #attack = foolbox.attacks.CarliniWagnerL2Attack(fmodel)
        #attack = foolbox.attacks.FGSM(fmodel)

        image = images[i].numpy()
        label = labels[i].numpy()
        #adversarial = attack(image, label, log_every_n_steps=999999, threaded_rnd=False, threaded_gen=False)
        adversarial = attack(image, label)


        #classification_label = int(np.argmax(fmodel.predictions(image)))
        #adversarial_label = int(np.argmax(fmodel.predictions(adversarial)))

        ###dispy.logger.info("source label: " + str(label) + ", adversarial_label: " + str(adversarial_label) + ", classification_label: " + str(classification_label))

        #os.environ.pop("OMP_NUM_THREADS")

    except Exception as e:
        #_dispy_logger.info(traceback.format_exc())
        return traceback.format_exc()
    else:
        return adversarial


if __name__ == '__main__':

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'

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
    batch_size = 5

    # tradeoff: precision of
    learning_rate = 0.01

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

    # train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
    #                                            batch_sampler=train_data_batch_sampler,
    #                                            **kwargs)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_sampler=train_data_batch_sampler)

    # batch size is irrelevant for test_loader, could set it to full test dataset
    # test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
    #                                           batch_size=batch_size,
    #                                           shuffle=False,
    #                                           **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)


    # load pretrained model if it exists
    pretrained_model_filepath = ME_DIR + '/pretrained_model2.torch'
    if os.path.isfile(pretrained_model_filepath):
        #model = torch.load(pretrained_model_filepath, map_location=lambda storage, loc: storage)
        model = torch.load(pretrained_model_filepath)
        model.to(device)
        adv_training_en=1
        learning_rate = 0.001
    else:
        model = MLP_Regression(input_size, num_classes)
        model.to(device)
        model.apply(weights_init)
        adv_training_en=0

    # softmax + cross entropy loss
    criterion = CrossEntropyLoss_SoftLabels(dim=1)
    criterion.to(device)

    # Note that technically this is not just Gradient Descent, not Stochastic Gradient Descent
    # What makes it 'stochastic' or 'not stochastic' is whether we use batch_size == dataset size or not
    # In reality, SGD is just a generalized form of GD (where batch size = dateset size), so it is not actually
    # incorrect to call it SGD when doing GD
    # model.parameters() in this case is the linear layer's parameters, aka the 'theta' of our algorithm
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    #optimizer.to(device)

    # adaptive learning rate policy -- "schedules" when to decrease learning rate
    scheduler = lr_rescheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)

    #'''
    fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=10, device=device)
    #attack = foolbox.attacks.BoundaryAttack(fmodel)
    #attack = foolbox.attacks.CarliniWagnerL2Attack(fmodel)
    attack = foolbox.attacks.FGSM(fmodel)


    '''
    ###### DISPY SETUP CLUSTER ################

    # save batch of images and labels
    images_filepath = ME_DIR + '/images.torch'
    labels_filepath = ME_DIR + '/labels.torch'

    # change working directory temporarily to force JobCluster command to dump in the proper output directory
    original_working_dir = os.getcwd()
    os.chdir(ME_DIR + '/dispy')

    # tell dispy where all the compute nodes are and set them up using setup command
    # cluster = dispy.JobCluster(dispy_compute_adversarial, nodes=args.compute_nodes_pythonic,
    #                           setup=functools.partial(dispy_setup_adversarial, ME_DIR))
    cluster = dispy.JobCluster(dispy_compute_adversarial,
                               setup=functools.partial(dispy_setup_adversarial, ME_DIR))
    # cluster = dispy.JobCluster(compute_optimal_portfolio, nodes=["nia1189.scinet.local", ], setup=setup)

    # return to original working dir to avoid any unintended effects from dir change
    os.chdir(original_working_dir)

    ###### END DISPY SETUP CLUSTER ################
    '''

    dataset_size = len(train_dataset)
    #subset_size = dataset_size // 100
    subset_size = 80
    rand_perm = torch.randperm(dataset_size)[0:subset_size]
    train_data_subset = (train_dataset.train_data.float()/255)[rand_perm]
    train_label_subset = train_dataset.train_labels[rand_perm]

    num_batches = subset_size // batch_size

    # Training the Model
    epoch=0
    #while True:
    for epoch in range(30):
        # train_loader keeps yielding batches until it reaches full epoch

        print("epoch: " + str(epoch))


        # TODO: smarter sampling
        #rand_perm = torch.randperm(subset_size)[0:batch_size]

        # shuffle data subset before epoch
        rand_perm = torch.randperm(subset_size)
        train_data_subset = train_data_subset[rand_perm]
        train_label_subset = train_label_subset[rand_perm]

        #for i, (images, labels) in enumerate(train_loader):
        for ldx in range(num_batches):

            images = train_data_subset[ldx * batch_size : (ldx + 1) * batch_size]
            labels = train_label_subset[ldx * batch_size : (ldx + 1) * batch_size]

            # 100 images x 784
            images = images.view(-1, 28 * 28)

            if not adv_training_en:
                soft_labels = torch.eye(num_classes)[labels].to(device)
                images = images.to(device)

                #images = torch.cat((images, images), 0).to(device)
                #soft_labels = torch.cat((soft_labels, soft_labels), 0).to(device)

            else:
                '''
                #print("BEFORE COMPUTE BATCH OF ADV")



                ########## FIND BATCH ADVERSARIALS USING DISPY ################

                # save batch
                torch.save(images, images_filepath)
                torch.save(labels, labels_filepath)

                # save model
                model_filepath = ME_DIR + '/model.torch'
                torch.save(model, model_filepath)


                jobs = []
                for j in range(len(images)):
                    # for i in range(10):
                    job = cluster.submit(j)  # it is sent to a node for executing 'compute'
                    jobs.append(job)

                #adversarials = torch.zeros(len(images))
                #print("Waiting for dispy jobs to complete...")
                adversarials = [None] * len(images)
                for idx, job in enumerate(jobs):
                    job()  # wait for job to finish
                    if type(job.result) == str:
                        print(job.result)
                        exit(1)
                    else:
                        #print("Done generating adversarial " + str(idx))
                        adversarials[idx] = job.result
                    #print(x)


                #cluster.close()


                ########## END FIND BATCH ADVERSARIALS USING DISPY ################
                '''

                print("batch: " + str(ldx))

                true_boundary_images=torch.zeros(batch_size, 28*28)
                true_boundary_labels=torch.zeros(batch_size, num_classes)


                #for idx, image in enumerate(images):

                for idx, image in enumerate(images):

                    #print("Generating adversarial " + str(idx))
                    #adversarial = attack(image.numpy(), labels[idx].numpy(), log_every_n_steps=999999)

                    #classification_label = int(np.argmax(fmodel.predictions(image.numpy())))
                    #adversarial_label = int(np.argmax(fmodel.predictions(adversarial)))
                    #print("source label: " + str(labels[idx]) + ", adversarial_label: " + str(adversarial_label) + ", classification_label: " + str(classification_label))

                    #adversarial = adversarials[idx]

                    # FOR TESTING OF SIMPLIFIED ALGORITHM
                    source_label = labels[idx]
                    #adversarial_label = int(np.argmax(fmodel.predictions(adversarial)))
                    #adversarial = torch.from_numpy(adversarial)
                    ##adversarial = attack(image.numpy(), source_label.numpy())
                    ##adversarial = torch.from_numpy(adversarial)

                    #'''
                    # get Goodfellow adversarial
                    amodel = attack._default_model
                    acriterion = attack._default_criterion
                    adistance = attack._default_distance
                    athreshold = attack._default_threshold
                    adv_obj = Adversarial(amodel, acriterion, image.numpy(), source_label.numpy(),
                                      distance=adistance, threshold=athreshold)
                    signed_gradient = attack._gradient(adv_obj)
                    adversarial = image.numpy() + signed_gradient * 0.01
                    adversarial = torch.from_numpy(adversarial)
                    #adversarial_label = int(np.argmax(fmodel.predictions(adversarial)))
                    #'''

                    true_boundary_images[idx] = adversarial
                    index = torch.tensor([source_label])
                    true_boundary_labels[idx] = torch.eye(num_classes)[index]

                    #true_boundary_images[idx] = torch.from_numpy(adversarial)
                    #boundary_label_values = torch.tensor([0.5, 0.5])
                    #index = torch.tensor([adversarial_label, source_label])
                    #true_boundary_labels[idx].scatter_(0, index, boundary_label_values)

                    #true_boundary_images[idx] = image
                    #index = source_label
                    #true_boundary_labels[idx] = torch.eye(num_classes)[index]


                    continue

                    source_label = labels[idx]

                    #adversarial = attack(image.numpy(), labels[idx].numpy(), binary_search_steps=9,
                    #                     initial_const=1e-3, learning_rate=1e-2)

                    # if original is misclassified, use original sample as regularizer
                    if np.array_equal(adversarial, image.numpy()):
                        print("Warning: original image already misclassified. Using original image as regularizer.")
                        true_boundary_images[idx] = image
                        index = torch.tensor([source_label])
                        true_boundary_labels[idx] = torch.eye(num_classes)[index]
                        continue


                    # if no adversarial is found, use original sample as regularaizre
                    # if adversarial is None:
                    #     print("Warning: no adversarial was found. Using original image as regularizer.")
                    #     true_boundary_images[idx] = image.numpy()
                    #     index = torch.tensor([source_label])
                    #     true_boundary_labels[idx] = torch.eye(num_classes)[index]
                    #     continue

                    if adversarial is None:
                        print("Warning: no adversarial was found! Using original image as regularizer.")
                        true_boundary_images[idx] = image
                        index = torch.tensor([source_label])
                        true_boundary_labels[idx] = torch.eye(num_classes)[index]
                        continue

                    adversarial_label = int(np.argmax(fmodel.predictions(adversarial)))

                    # pytorch wants it in tensor format
                    adversarial = torch.from_numpy(adversarial)

                    # find nearest neighbor in target class
                    target_label = adversarial_label
                    #target_images = images[labels==adversarial_label]
                    # TODO: put train_images[class_idx] outside model
                    target_images = train_dataset.train_data[train_data_batch_sampler.class_sample_indices[target_label]].view(-1, 28 * 28).float()/255

                    distances = torch.norm(target_images - adversarial, p=2, dim=1)
                    distances_sorted, sorted_indices = torch.sort(distances, 0)

                    rt = distances_sorted[0]

                    # TODO: use range(k) instead of zero here if choose k nearest neighbors
                    nearest_neighbor_indices = sorted_indices[0]

                    # NOTE: nearest_target_neighbors becomes an array instead of single image if choose k nearest neighbors
                    nearest_target_neighbors=target_images[nearest_neighbor_indices]

                    # check that region from adversarial to target are all classified as target label
                    target = nearest_target_neighbors





                    # find nearest neighbor in source class

                    #source_images = images[labels==source_label]
                    source_images = train_dataset.train_data[train_data_batch_sampler.class_sample_indices[source_label]].view(-1, 28 * 28).float()/255

                    distances = torch.norm(source_images - adversarial, p=2, dim=1)
                    distances_sorted, sorted_indices = torch.sort(distances, 0)

                    rs = distances_sorted[0]

                    # TODO: use range(k) instead of zero here if choose k nearest neighbors
                    nearest_neighbor_indices = sorted_indices[0]


                    # TODO: nearest_source_neighbors becomes an array instead of single image if choose k nearest neighbors
                    nearest_source_neighbors = source_images[nearest_neighbor_indices]


                    # find "true boundary" image

                    # TODO: use mean of all nearest neighbors if choose k nearest neighbors (where k<1)
                    # aka the code below (has been tested!)
                    # (torch.mean(nearest_target_neighbors,0) + torch.mean(nearest_source_neighbors,0)) / 2



                    num_increments = 100
                    incremental_shift = (target - adversarial) / num_increments
                    found_anomaly = 0
                    for jdx in range(num_increments):
                        check_sample = adversarial + jdx * incremental_shift
                        check_label = int(np.argmax(fmodel.predictions(check_sample.numpy())))
                        if check_label == source_label:
                            if jdx == 0:
                                print("ERROR: the adversarial is not actually adversarial.")
                                print("Target label=" + str(target_label) + ", source label=" + str(
                                    source_label) + ", check label=" + str(check_label))
                                exit(1)
                            elif jdx == 1:
                                print("Warning: target is in wrong direction. Using original image as regularizer.")
                                true_boundary_images[idx] = image
                                index = torch.tensor([source_label])
                                true_boundary_labels[idx] = torch.eye(num_classes)[index]
                                continue
                            else:
                                print("Warning: Filling in non-convexity! With jdx=" + str(jdx))
                                print("Target label=" + str(target_label) + ", source label=" + str(
                                    source_label) + ", check label=" + str(check_label))


                            '''
                            distance_travelled_from_adversarial = float(jdx) * rt
                            distance_to_move_boundary = distance_travelled_from_adversarial - rs
                            if distance_to_move_boundary > 0:
                                true_boundary_images[idx] = adversarial + distance_to_move_boundary * (target - adversarial)/torch.norm(target-adversarial)
                            else:
                                true_boundary_images[idx] = adversarial + distance_to_move_boundary * (adversarial - source_images) / torch.norm(adversarial - source_images)
                            '''


                            jdx=0
                            true_boundary_images[idx] = adversarial + (float(jdx) / 2) * incremental_shift
                            index = torch.tensor([source_label])
                            true_boundary_labels[idx] = torch.eye(num_classes)[index]
                            found_anomaly = 1
                            break
                        elif check_label != target_label:
                            if jdx == 0:
                                print("ERROR: the adversarial is not of the claimed target class.")
                                print("Target label=" + str(target_label) + ", source label=" + str(
                                    source_label) + ", check label=" + str(check_label))
                                exit(1)
                            elif jdx == 1:
                                print("ERROR: the adversarial is immediately next to a third class.")
                                print("Target label=" + str(target_label) + ", source label=" + str(
                                    source_label) + ", check label=" + str(check_label))
                                exit(1)

                            print("Very surpising! Found intercepting class. NEED TO IMPROVE ALGO FOR THIS CASE. With jdx=" + str(jdx))
                            print("Target label=" + str(target_label) + ", source label=" + str(
                                source_label) + ", check label=" + str(check_label))

                            # improve this: we have not actually found an intercepting sample, only an intercepting classification region.
                            # yet we are treating this point as an intercepting sample. can we do better?


    #                        true_boundary_images[idx] = adversarial + (float(jdx) / 2) * incremental_shift
                            '''
                            distance_travelled_from_adversarial = float(jdx) * rt
                            distance_to_move_boundary = distance_travelled_from_adversarial - rs
                            if distance_to_move_boundary > 0:
                                true_boundary_images[idx] = adversarial + distance_to_move_boundary * (target - adversarial)/torch.norm(target-adversarial)
                            else:
                                true_boundary_images[idx] = adversarial + distance_to_move_boundary * (adversarial - source_images) / torch.norm(adversarial - source_images)
                            '''
                            jdx = 0
                            true_boundary_images[idx] = adversarial + (float(jdx) / 2) * incremental_shift


                            boundary_label_values = torch.tensor([0.5, 0.5])
                            index = torch.tensor([check_label, source_label])
                            true_boundary_labels[idx].scatter_(0, index, boundary_label_values)
                            found_anomaly = 1
                            break

                    if found_anomaly:
                        # already computed boundary point and label in if/elif above, can continue with next sample
                        continue

                    print("Warning: Performing normal regularization")
                    rs=0
                    distance_to_move_boundary = rt - rs
                    if distance_to_move_boundary > 0:
                        true_boundary_images[idx] = adversarial + distance_to_move_boundary * (
                                    target - adversarial) / torch.norm(target - adversarial)
                    else:
                        true_boundary_images[idx] = adversarial + distance_to_move_boundary * (
                                    adversarial - source_images) / torch.norm(adversarial - source_images)

                    boundary_label_values = torch.tensor([0.5, 0.5])
                    index = torch.tensor([target_label, source_label])
                    true_boundary_labels[idx].scatter_(0, index, boundary_label_values)

                images = torch.cat((images, true_boundary_images), 0).to(device)
                #images = images.to(device)

                # above, we made use of 'labels' being integer for indexing;
                # now that we're making labels a

                original_labels = torch.eye(num_classes)[labels]

                soft_labels = torch.cat((original_labels, true_boundary_labels), 0).to(device)

                #'''

            # 100 labels

            # prevent gradients from accumulating -- they should be computed "fresh" during each batch
            optimizer.zero_grad()

            # linear layer
            #images.requires_grad=True
            outputs = model(images)

            # softmax "layer" + cross entropy loss
            loss = criterion(outputs, soft_labels)

            if adv_training_en:

                original_loss = criterion(outputs[0:batch_size], original_labels)
                boundary_loss = criterion(outputs[batch_size:], true_boundary_labels)

            #loss_original = criterion_original(outputs, labels)

            # compute loss gradients wrt to parameters


            loss.backward()

            #np.sign(images.grad)

            # update parameters (here linear layer parameters) using learning rate + gradients
            optimizer.step()

            #if adv_training_en:
            #if True:
            if (ldx + 1) % num_batches == 0:
                model.eval()
                total=0
                correct=0
                for k, (test_images, test_labels) in enumerate(test_loader):
                    test_images = test_images.view(-1, 28 * 28)
                    test_images = test_images.to(device)

                    test_labels = test_labels.to(device)

                    # linear layer
                    outputs = model(test_images)

                    _, predicted = torch.max(outputs.data, 1)
                    total += len(test_labels)
                    correct += (predicted == test_labels).sum()

                print('Test Accuracy: %.3f %%' % (100 * correct.item() / total))
                model.train()


            # TODO: change end condition?
            #if (i + 1) % 600 == 0:
            #if True:
                lr=optimizer.param_groups[0]['lr']
                best_loss = scheduler.best

                if adv_training_en:
                    print('Best end-of-epoch loss: %.7f, LR: %.4f, Epoch: [%d/%d], Step: [%d/%d], Loss: %.7f, Original Loss: %.7f, Boundary Loss: %.7f'
                          % (best_loss, lr, epoch + 1, num_epochs, ldx + 1, len(train_dataset) // batch_size, loss.data.item(), original_loss.data.item(), boundary_loss.data.item()))

                else:
                    print('Best end-of-epoch loss: %.7f, LR: %.4f, Epoch: [%d/%d], Step: [%d/%d], Loss: %.7f'
                          % (best_loss, lr, epoch + 1, num_epochs, ldx + 1, len(train_dataset) // batch_size, loss.data.item()))


        epoch+=1

        # adapt learning rate
        scheduler.step(loss)
        lr=optimizer.param_groups[0]['lr']


        if lr < 0.0001:

            model.eval()
            total = 0
            correct = 0
            for k, (test_images, test_labels) in enumerate(test_loader):
                test_images = test_images.view(-1, 28 * 28)
                test_images = test_images.to(device)

                test_labels = test_labels.to(device)

                # linear layer
                outputs = model(test_images)

                # note that we do not need softmax layer for the decision process, as softmax does not change the
                # order the selection (merely amplifies the difference between them) -- picking index corresponding to
                # max value is sufficient

                _, predicted = torch.max(outputs.data, 1)
                total += len(test_labels)
                correct += (predicted == test_labels).sum()

            print('Test Accuracy: %.3f %%' % (100 * correct.item() / total))
            model.train()

            if not adv_training_en:
                # reset scheduler and optimizer
                optimizer = torch.optim.SGD(model.parameters(), lr=lr * 10)
                scheduler = lr_rescheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)

                # save model to skip pre-training in future runs
                pretrained_model_filepath = args.dataset_directory + '/pretrained_model2.torch'
                torch.save(model, pretrained_model_filepath)

                exit()

                # enable adversarial training
                adv_training_en = 1

            else:
                break

    pretrained_model_filepath = args.dataset_directory + '/pretrained_model2.torch'
    torch.save(model, pretrained_model_filepath)