from .planner import Planner, save_model 
import torch
import torch.utils.tensorboard as tb
import numpy as np
from .utils import load_data
from . import dense_transforms

def train(args):
    from os import path

    print('start training')

    # The device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # The model
    model = Planner().to(device)

    # The loggers
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
    
    # The hyperparms
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
    train_data = load_data(batch_size=bs, num_workers=4,
                                     transform=dense_transforms.Compose((
                                             dense_transforms.ToTensor(),
                                             dense_transforms.RandomHorizontalFlip(),
                                             dense_transforms.ColorJitter(0.8, 0.8, 0.8, 0.1),
                                     )),
                          )
    valid_data = load_data(dataset_path='drive_data', batch_size=bs)

    # The loss function
    def loss_fn(pred, label):
        ys = label[:, 1]
        ys = ys * 500

        loss = (pred - label)**2
        loss = loss.sum(dim=1)
        loss * ys

        return loss.sum()

    #loss_fn = torch.nn.MSELoss()

    # The optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size,
                                                gamma=gamma)

    global_step = 0
    epoch = 0
    while epoch < n_epochs:
        for image, label in train_data:
            image = image.to(device)
            label = label.to(device)

            pred = model(image)

            loss = loss_fn(pred, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
        
        scheduler.step()

        print(epoch)
        epoch += 1
      
    save_model(model)
    """
    Your code here, modify your HW4 code
    Hint: Use the log function below to debug and visualize your model
    """


def log(logger, img, label, pred, global_step):
    """
    logger: train_logger/valid_logger
    img: image tensor from data loader
    label: ground-truth aim point
    pred: predited aim point
    global_step: iteration
    """
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    fig, ax = plt.subplots(1, 1)
    ax.imshow(TF.to_pil_image(img[0].cpu()))
    WH2 = np.array([img.size(-1), img.size(-2)])/2
    ax.add_artist(plt.Circle(WH2*(label[0].cpu().detach().numpy()+1), 2, ec='g', fill=False, lw=1.5))
    ax.add_artist(plt.Circle(WH2*(pred[0].cpu().detach().numpy()+1), 2, ec='r', fill=False, lw=1.5))
    logger.add_figure('viz', fig, global_step)
    del ax, fig

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
