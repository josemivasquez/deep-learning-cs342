import torch
import numpy as np

from .models import Detector, save_model, load_model
from .utils import load_detection_data
from . import dense_transforms
import torch.utils.tensorboard as tb


def train(args):
    """
    Your code here, modify your HW3 code
    Hint: Use the log function below to debug and visualize your model
    """

    from datetime import datetime
    def namer(acc):
        st = str(acc) + '-' + datetime.now().strftime('%d-%m-%Y,%H-%M-%S')
        return st

    print('start training')

    # The device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # The model
    if not args.load:
        model = Detector().to(device)
    else:
        model = load_model().to(device)

    # The Loggers
    from os import path
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    # Hyperparams
    learning_rate = args.learning_rate
    n_epochs = args.n_epochs
    momentum = args.momentum
    weight_decay = args.weight_decay
    step_size = args.scheduler_step_size
    gamma = args.scheduler_gamma
    verb_step = args.verbose_step

    objective = args.objective
    objective_epoch = None
    bs = args.batch_size

    # Get Data
    train_data = load_detection_data('dense_data/train', batch_size=bs, num_workers=4,
                                     transform=dense_transforms.Compose(
                                         (
                                             dense_transforms.ColorJitter(0.8, 0.8, 0.8, 0.1),
                                             dense_transforms.RandomHorizontalFlip(),
                                             dense_transforms.ToTensor(),
                                             dense_transforms.ToHeatmap(),
                                     )
                                     ))
    valid_data = load_detection_data('dense_data/valid', batch_size=bs)

    # The loss fn
    # loss_fn = torch.nn.BCEWithLogitsLoss(loss_weigths(DENSE_CLASS_DISTRIBUTION)).to(device)

    # ------------------- COPY -------------------
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
    # ------------------- FINISH COPY ------------

    # The optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                momentum=momentum, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size,
                                                gamma=gamma)

    # Accuracy series
    global_step = 0
    epoch = 0
    while epoch < n_epochs:
        # Shuffle batches from the data loader
        for image, peaks, sizes in train_data:
            # Compute the out and the loss
            image = image.to(device)
            peaks = peaks.to(device)
            sizes = sizes.to(device)

            pred = model(image)

            # ------------------- COPY -------------------
            med = torch.sigmoid(pred * (1 - 2*peaks))
            loss = (loss_fn(pred, peaks) * med).mean() / med.mean()
            # ------------------- FINISH COPY ------------

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.log_dir is not None and global_step % 100 == 0:
                train_logger.add_scalar('loss', loss, global_step)

            global_step += 1

        scheduler.step()

        # Evaluate the model
        # train_acc, train_iou = confusion_matrix.global_accuracy, confusion_matrix.iou
        # valid_acc, valid_iou = accuracy_cal(valid_data)
        # train_acc, train_iou = 0, 0
        # valid_acc, valid_iou = 0, 0

        # Epoch logs
        # if args.log_dir is not None:
        #     train_logger.add_scalar('accuracy', train_acc, global_step)
        #     train_logger.add_scalar('iou', train_iou, global_step)
        #     valid_logger.add_scalar('accuracy', valid_acc, global_step)
        #     valid_logger.add_scalar('iou', valid_iou, global_step)

        # Epoch prints
        if epoch % verb_step == 0:
            print('Epoch %d, Loss %f' % (epoch, float(loss)))
            # print('Train accuracy: %0.4f' % train_acc)
            # print('Train iou: %0.4f' % train_iou)
            # print('Valid accuracy: %0.4f' % valid_acc)
            # print('Valid iou: %0.4f' % valid_iou)

        # # Epoch save
        # if valid_acc > objective:
        #     objective = valid_acc
        #     objective_epoch = epoch
        #     # save_model(model, namer(valid_acc))

        # # Epoch extension
        # if epoch == n_epochs - 1:
        #     if objective_epoch is not None and n_epochs - objective_epoch < args.extension:
        #         n_epochs += args.extension

        epoch += 1

    save_model(model)



def log(logger, imgs, gt_det, det, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    gt_det: ground-truth object-center maps
    det: predicted object-center heatmaps
    global_step: iteration
    """
    logger.add_images('image', imgs[:16], global_step)
    logger.add_images('label', gt_det[:16], global_step)
    logger.add_images('pred', torch.sigmoid(det[:16]), global_step)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', type=str)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-03)
    parser.add_argument('-ep', '--n_epochs', type=int, default=20)
    parser.add_argument('-mtm', '--momentum', type=float, default=0)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0)
    parser.add_argument('-load', type=bool, default=False)
    parser.add_argument('-sch_ss', '--scheduler_step_size', type=int, default=20)
    parser.add_argument('-sch_gamma', '--scheduler_gamma', type=float, default=0.1)
    parser.add_argument('-verb_step', '--verbose_step', type=int, default=1)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)

    parser.add_argument('-obj', '--objective', type=float, default=0.865)
    parser.add_argument('-ext', '--extension', type=int, default=10)
    parser.add_argument('-trains', '--n_trains', type=int, default=1)

    args = parser.parse_args()
    train(args)
