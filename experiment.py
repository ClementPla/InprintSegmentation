import os
from enum import auto, IntFlag

import albumentations as A
import cv2
import nntools.tracker.metrics as NNmetrics
import numpy as np
import torch
from nntools import SupervisedExperiment
from nntools.utils import reduce_tensor
import nntools.dataset as D
from networks import get_network


class DA(IntFlag):
    NONE = auto()
    GEOMETRIC = auto()
    COLOR = auto()
    COLOR_V2 = auto()

    @property
    def name(self):
        name = super(DA, self).name
        if name:
            return name
        else:
            return ', '.join([flag.name for flag in DA if flag in self])



da_funcs = {DA.NONE: [],
            DA.GEOMETRIC: [A.HorizontalFlip(p=0.5),
                           A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.10, rotate_limit=15,
                                              border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0)],
            DA.COLOR: [A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                       A.GaussianBlur(blur_limit=7, p=0.25), A.Sharpen(p=0.25),
                       A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=0, val_shift_limit=0)],
            (DA.COLOR | DA.GEOMETRIC): [A.RandomBrightnessContrast(brightness_limit=0.2,
                                                                   contrast_limit=0.2),
                                        A.GaussianBlur(blur_limit=7, p=0.25),
                                        A.Sharpen(p=0.25),
                                        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=0,
                                                             val_shift_limit=0),
                                        A.HorizontalFlip(p=0.5),
                                        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.10,
                                                           rotate_limit=15,
                                                           border_mode=cv2.BORDER_CONSTANT,
                                                           value=0, mask_value=0)]
            }


@D.nntools_wrapper
def to_rgb(image):
    if image.ndim == 2:
        image = np.expand_dims(image, 2)
    if image.shape[2]==1:
        image = np.repeat(image, 3, 2)

    return {'image':image}




class SegInprintExp(SupervisedExperiment):
    def __init__(self, config, id=None,
                 da_level=(DA.COLOR | DA.GEOMETRIC)):
        super(SegInprintExp, self).__init__(config, id)
        network = get_network(config['Network'])
        self.set_model(network)
        self.da_level = da_level

        dataset = D.SegmentationDataset(**self.c['Dataset'])
        v_size = self.c['Validation']['size']
        train_dataset, valid_dataset = D.random_split(dataset, [len(dataset) - v_size, v_size])
        test_dataset = D.SegmentationDataset(self.c['Test']['img_url'], shape=self.c['Dataset']['shape'],
                                             keep_size_ratio=True)

        # Data augmentation
        aug_func = da_funcs[da_level]
        aug = A.Compose(aug_func)
        ops = [aug, A.Normalize(mean=self.c['Preprocessing']['mean'],
                                std=self.c['Preprocessing']['std'],
                                always_apply=True)]

        if self.c['Preprocessing']['random_crop']:
            ops.append(A.CropNonEmptyMaskIfExists(*self.c['Preprocessing']['crop_size'], always_apply=True))

        composer_train = D.Composition()
        composer_train.add(*ops)
        train_dataset.set_composition(composer_train)

        composer_test = D.Composition()

        test_funcs = [to_rgb, A.PadIfNeeded(pad_height_divisor=2**4, pad_width_divisor=2**4, min_height=None, min_width=None),
                      A.Normalize(mean=self.c['Preprocessing']['mean'], std=self.c['Preprocessing']['std'],
                                  always_apply=True)]

        composer_test.add(*test_funcs)

        valid_dataset.set_composition(composer_test)
        test_dataset.set_composition(composer_test)

        self.set_train_dataset(train_dataset)
        self.set_valid_dataset(valid_dataset)
        self.set_test_dataset(test_dataset)
        """
        Define optimizers
        """
        self.set_optimizer(**self.c['Optimizer'])
        self.set_scheduler(**self.c['Learning_rate_scheduler'])
        self.log_artifacts(os.path.realpath(__file__))
        self.log_params(DA=self.da_level.name)
        print("Training size %i, validation size %i, test size %i" % (len(self.train_dataset),
                                                                      len(self.validation_dataset),
                                                                      len(self.test_dataset)))

    def end(self, model, rank):
        gpu = self.get_gpu_from_rank(rank)
        map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}
        model.load(self.tracker.network_savepoint, load_most_recent=True, map_location=map_location, strict=False,
                   filtername='best_valid')
        model.eval()
        self.test_dataset.return_indices = True
        test_loader, test_sampler = self.get_dataloader(self.test_dataset, shuffle=False, batch_size=1,
                                                        rank=rank)
        with torch.no_grad():
            for batch in test_loader:
                img = batch['image'].cuda(gpu)
                index = batch['indice']
                probas = model(img)
                preds = torch.argmax(probas, 1).float()
                preds = preds
                filenames = self.test_dataset.filename(index)
                for p, f in zip(preds, filenames):
                    p = p.cpu().numpy().astype(np.uint8)
                    h, w = p.shape
                    prediction = np.zeros((h, w, 3), dtype=np.uint8)
                    prediction[p == 1, 0] = 255
                    prediction[p == 2, 2] = 255
                    filename = os.path.join(self.tracker.save_paths['prediction_savepoint'], f).replace('tif', 'png')
                    cv2.imwrite(filename, prediction)

    def validate(self, model, valid_loader, iteration, rank=0, loss_function=None):
        model.eval()
        gpu = self.get_gpu_from_rank(rank)

        confMat = torch.zeros(self.n_classes, self.n_classes).cuda(gpu)
        for n, batch in enumerate(valid_loader):
            batch = self.batch_to_device(batch, rank)
            img = batch['image']
            gt = batch['mask']
            proba = model(img)
            preds = torch.argmax(proba, 1)
            confMat += NNmetrics.confusion_matrix(preds, gt, num_classes=self.n_classes, multilabel=False)

        if self.multi_gpu:
            confMat = reduce_tensor(confMat, self.world_size, mode='sum')
        confMat = NNmetrics.filter_index_cm(confMat, self.ignore_index)
        mIoU = NNmetrics.mIoU_cm(confMat)
        if self.is_main_process(rank):
            stats = NNmetrics.report_cm(confMat)
            stats['mIoU'] = mIoU
            self.log_metrics(step=iteration, **stats)
            if self.tracked_metric is None or mIoU >= self.tracked_metric:
                self.tracked_metric = mIoU
                filename = ('best_valid_iteration_%i_mIoU_%.3f' % (iteration, mIoU)).replace('.', '')
                self.save_model(model, filename=filename)

        model.train()
        return mIoU
