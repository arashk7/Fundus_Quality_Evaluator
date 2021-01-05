'''
Ahmad Karambakhsh
Fundus Image Quality Evaluator
using EfficientNet
ISBI train
'''
exp_name = 'fqe_isbi'  # default

# BASE_TRAIN_PATH = '/home/arash/Projects/Dataset/aptos2019-blindness-detection'
ISBI_TRAIN_PATH = 'E:\Dataset\DR\DeepDr\merged_tr_vl'
ISBI_TEST_PATH = 'E:\Dataset\DR\DeepDr\Onsite-Challenge1-2-Evaluation'
# APTOS_TRAIN_PATH = 'E:\Dataset\DR/aptos2019-blindness-detection'
# KAGGLEDR_TRAIN_PATH = 'E:\Dataset\DR\DiabeticRetinopathyDetection/train.zip'
chp_path = 'checkpoints_isbi'
task = 'train'
'''Parameteres'''
logger = False
limit_val_train = 0.02
img_size = 200
batch_size = 4
efficientnet = 0
learning_rate = 1e-4
patience = 1
balancing = True
gem_pool = True
num_fold = 5
'''Parameteres
smooth_l1_loss
adam optimizer

'''
'''aug'''
aug_rot = 120
'''aug
aug_norm = True
bichannel = False
'''

import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils import data
from torch.utils.data import random_split

import torchvision
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image
from sklearn.model_selection import KFold
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score, f1_score, accuracy_score

from collections import Counter

from torch.nn.parameter import Parameter
from sklearn.model_selection import KFold
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.callbacks import ModelCheckpoint

''' GeM Pooling'''


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


''' GeM Pooling'''


class GeM(torch.nn.Module):

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
            self.eps) + ')'


class FQEModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b' + str(efficientnet), num_classes=2, in_channels=3)
        if gem_pool:
            in_size = 1280  # eff 0
            if efficientnet == 3:
                in_size = 1536
            elif efficientnet == 5:
                in_size = 2048

            self.gem_pooling = GeM()
            self.dropout = torch.nn.Dropout(0.5)
            self.fc = torch.nn.Linear(in_size, 2)  # 0=1280, 3=1536, 4=2048

        self.metric = F.smooth_l1_loss
        self.val_acc = pl.metrics.Accuracy()

        self.preds = []
        self.labels = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=learning_rate)

    def quadratic_kappa_cpu(self, y_hat, y):
        return cohen_kappa_score(y_hat, y, weights='quadratic')

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.float()
        if gem_pool:

            x = self.model.extract_features(x)
            x = self.gem_pooling(x)
            x = x.view(batch_size, -1)
            x = self.dropout(x)
            x = self.fc(x)
        else:
            x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y[y <= 1] = 0
        y[y > 1] = 1
        y_hat = self(x)
        # y_hat = torch.tensor(y_hat.argmax(dim=1, keepdim=True),dtype=torch.float32,device='cuda:0')
        # y_hat[y_hat <= 1] = 0
        # y_hat[y_hat > 1] = 1
        eye = torch.eye(2).cuda()
        y = eye[y]

        loss = self.metric(y_hat, y)

        return {'loss': loss, 'log': {'train_step_loss': loss}}

    def training_epoch_end(self, outputs):
        loss = sum(x['loss'] for x in outputs) / len(outputs)
        return {'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y[y <= 1] = 0
        y[y > 1] = 1
        y_hat = self(x)
        # y_hat = torch.tensor(y_hat.argmax(dim=1, keepdim=True), dtype=torch.float32, device='cuda:0')

        # y_hat[y_hat <= 1] = 0
        # y_hat[y_hat > 1] = 1

        eye = torch.eye(2).cuda()
        yy = eye[y]

        loss = self.metric(yy, y_hat)
        pred = torch.reshape(y_hat.argmax(dim=1, keepdim=True), (batch_size, 1))

        accuracy = pred.eq(y.view_as(pred)).float().mean()

        self.preds += pred.tolist()
        self.labels += y.tolist()

        return {"val_loss": loss, "val_step_acc": accuracy}

    def validation_epoch_end(self, outputs):
        accuracy = sum(x['val_step_acc'] for x in outputs) / len(outputs)
        loss = sum(x['val_loss'] for x in outputs) / len(outputs)

        qkappa = torch.tensor(self.quadratic_kappa_cpu(self.preds, self.labels))

        print({'val_epoch_qkappa': qkappa.item(), 'val_epoch_acc': accuracy.item()})
        self.preds = []
        self.labels = []
        return {'log': {'val_loss': loss, 'val_epoch_acc': accuracy, 'val_epoch_qkappa': qkappa}}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y[y <= 1] = 0
        y[y > 1] = 1
        y_hat = self(x)
        # y_hat = torch.tensor(y_hat.argmax(dim=1, keepdim=True), dtype=torch.float32, device='cuda:0')
        # y_hat[y_hat <= 1] = 0
        # y_hat[y_hat > 1] = 1
        eye = torch.eye(2).cuda()
        yy = eye[y]

        loss = self.metric(y_hat, yy)
        pred = torch.reshape(y_hat.argmax(dim=1, keepdim=True), (batch_size, 1))
        accuracy = pred.eq(y.view_as(pred)).float().mean()
        self.preds += pred.tolist()
        self.labels += y.tolist()

        return {"test_step_loss": loss, "test_step_acc": accuracy}

    def test_epoch_end(self, outputs):
        accuracy = sum(x['test_step_acc'] for x in outputs) / len(outputs)
        loss = sum(x['test_step_loss'] for x in outputs) / len(outputs)
        qkappa = torch.tensor(self.quadratic_kappa_cpu(self.preds, self.labels))
        print({'test_epoch_qkappa': qkappa})
        self.preds = []
        self.labels = []
        return {'log': {'test_loss': loss, 'test_epoch_acc': accuracy, 'test_epoch_qkappa': qkappa}}


class Dataset_ISBI(data.Dataset):
    def __init__(self, csv_path, images_path, transform=None):
        ''' Initialise paths and transforms '''
        self.pd_set = pd.read_csv(csv_path, keep_default_na=False)  # Read The CSV and create the dataframe
        self.image_path = images_path  # Images Path
        self.transform = transform  # Augmentation Transforms

    def __len__(self):
        return len(self.pd_set)

    def labels(self, indices=None):
        '''

        :param indices:
        :return:
        return all the label information regards to indices
        '''
        labels = []
        if indices is None:
            for i in range(len(self.pd_set['image_quality'])):
                labels.append(int(self.pd_set['image_quality'][i]))
            return labels
        else:
            for i in indices:
                labels.append(int(self.pd_set['image_quality'][i]))
            return labels

    def __getitem__(self, idx):
        '''
        Receive element index, load the image from the path and transform it
        :param idx:
        Element index
        :return:
        Transformed image and its grade label
        '''
        img_id = self.pd_set['image_id'][idx]
        patient_id = self.pd_set['patient_id'][idx]
        file_path = os.path.join(str(patient_id), str(img_id) + '.jpg')

        label = int(self.pd_set['image_quality'][idx])

        if label>1:
            label =0
        path = os.path.join(self.image_path, file_path)
        path = path.replace('\\', '/')
        img = Image.open(path)  # Loading Image

        if self.transform is not None:
            img = self.transform(img)
        return img, label


def train_isbi(trainer, model, dataset, dataset_test, logger):
    torch.save(model.model, "init.ckpt")

    progressbar_callback = trainer.callbacks[1]

    '''KFolding'''
    kfold = KFold(num_fold, shuffle=True, random_state=1)
    for fold, (train_index, val_index) in enumerate(kfold.split(dataset)):
        model.model = torch.load("init.ckpt")
        # trainer.restore_weights(model)
        early_stopping = EarlyStopping(
            monitor='val_loss',
            min_delta=0.00,
            patience=patience,
            verbose=False,
            mode='min'
        )
        checkpoint_callback = ModelCheckpoint(
            monitor='val_epoch_qkappa',
            filepath=chp_path + '/offline_' + exp_name + '_fold_' + str(fold),
            save_top_k=1,
            mode='max')

        '''restore trainer callbacks '''
        trainer.callbacks.clear()
        trainer.callbacks.append(trainer.configure_early_stopping(early_stopping))
        trainer.callbacks.append(trainer.configure_checkpoint_callback(checkpoint_callback))
        trainer.callbacks.append(progressbar_callback)

        if logger:
            checkpoint_callback.filename = exp_name + '_' + experiment_id + '_fold_' + str(fold)

        '''balancing'''
        labels = dataset.labels(train_index)
        class_counts = Counter(labels)
        num_samples = len(labels)
        class_weights = [num_samples / class_counts[i] for i in range(len(class_counts))]
        weights = [class_weights[labels[i]] for i in range(int(num_samples))]
        '''balance train sampler'''
        train_sampler = WeightedRandomSampler(torch.DoubleTensor(weights), len(weights), replacement=True)

        '''validation sampler'''
        val_sampler = SubsetRandomSampler(val_index)

        '''db loaders'''
        train_loader = torch.utils.data.DataLoader(dataset,
                                                   sampler=train_sampler,
                                                   batch_size=batch_size)
        val_loader = torch.utils.data.DataLoader(dataset,
                                                 sampler=val_sampler,
                                                 batch_size=batch_size)

        test_loader = torch.utils.data.DataLoader(dataset_test,
                                                  batch_size=batch_size)

        '''start training'''
        trainer.fit(model, train_loader, val_loader)
        # print(checkpoint_callback.best_model_path)
        # trainer.restore(checkpoint_callback.best_model_path, on_gpu=True)
        model.load_from_checkpoint(checkpoint_callback.best_model_path)
        result = trainer.test(model, test_loader)

        kappa = result[0]['test_epoch_qkappa']
        kappa = round(kappa, 3)
        print('>>>>>>>>>>>>>>>>Test qkappa: ' + str(kappa))

        trainer.save_checkpoint(chp_path + '/' + experiment_id + '_kappa_' + str(kappa) + '_fold_' + str(fold) + '.pt')


transform_train = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.RandomApply([
    torchvision.transforms.RandomRotation((-aug_rot, aug_rot)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.4)],
    0.7), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                      ])

transform_test = transforms.Compose([transforms.Resize((img_size, img_size)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])
                                     ])

'''Dataset'''
isbi_dataset_train = Dataset_ISBI(os.path.join(ISBI_TRAIN_PATH, 'merged_tr_vl.csv'),
                                  ISBI_TRAIN_PATH,
                                  transform=transform_train)
isbi_dataset_test = Dataset_ISBI(os.path.join(ISBI_TEST_PATH, 'Onsite-Challenge1-2-Evaluation_full.csv'),
                                 ISBI_TEST_PATH,
                                 transform=transform_test)
# aptos_dataset_train = Dataset_APTOS(os.path.join(APTOS_TRAIN_PATH, 'train.csv'),
#                                     os.path.join(APTOS_TRAIN_PATH, 'train_images'),
#                                     transform=transform_train)
# kaggledr_dataset_train = Dataset_KAGGLEDR(os.path.join(KAGGLEDR_TRAIN_PATH, 'trainLabels.csv'),
#                                           os.path.join(KAGGLEDR_TRAIN_PATH, 'train'),
#                                           transform=transform_train)
experiment_id=exp_name
model = FQEModel()

trainer = None
if logger:
    neptune_logger = NeptuneLogger(
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiZDEwMDVjZGQtNzNlOS00ZDNmLTlmMjYtNGNhMzk0NmMwZWFkIn0=",
        project_name="arash.k/sandbox", experiment_name=exp_name)  # arash.k
    trainer = pl.Trainer(gpus=1, limit_val_batches=limit_val_train, limit_train_batches=limit_val_train,
                         limit_test_batches=limit_val_train,
                         logger=neptune_logger)
    experiment_id = neptune_logger.experiment.id
else:
    trainer = pl.Trainer(gpus=1, limit_val_batches=limit_val_train, limit_train_batches=limit_val_train,
                         limit_test_batches=limit_val_train)

print('>>>>>TRAIN KAGGLE Dataset')
# model = pretrain_kaggledr(trainer, model, kaggledr_dataset_train, logger)
print('>>>>>TRAIN APTOS Dataset')
# model = pretrain_aptos(trainer, model, aptos_dataset_train, logger)
print('>>>>>TRAIN ISBI Dataset')
train_isbi(trainer, model, isbi_dataset_train, isbi_dataset_test, logger)
