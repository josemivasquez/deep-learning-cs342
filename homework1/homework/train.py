from .models import ClassificationLoss, model_factory, save_model, load_model
from .utils import accuracy, load_data
import torch


def train(args):
    def accuracy_cal(data) -> float:
        result_tensor = torch.Tensor(())
        for input_b, target_b in data:
            out_b = torch.argmax(model(input_b), dim=1)
            comp = target_b == out_b
            result_tensor = torch.cat((result_tensor, comp))

        rs = float(result_tensor.float().mean())
        return rs

    print('start training')
    # Hyperparams
    learning_rate = args.learning_rate
    n_epochs = args.n_epochs
    momentum = args.momentum
    weight_decay = args.weight_decay

    # Get Data
    train_data = load_data('data/train')
    valid_data = load_data('data/valid')

    # The model
    if not args.load:
        model = model_factory[args.model]()

    else:
        model = load_model(args.model)

    # The loss fn
    loss_fn = ClassificationLoss()

    # The optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,\
                                momentum=momentum, weight_decay=weight_decay)
    # Accuracy series
    train_acc_serie = []
    valid_acc_serie = []

    global_step = 0
    for epoch in range(n_epochs):
        # Shuffle batches from the data loader
        for input_b, target_b in train_data:
            # Compute the out and the loss
            out = model(input_b)
            loss = loss_fn(out, target_b)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

        # Evaluate the model
        # Train Data
        train_acc = accuracy_cal(train_data)
        train_acc_serie.append(train_acc)

        # Validation Data
        valid_acc = accuracy_cal(valid_data)
        valid_acc_serie.append(valid_acc)

        if epoch % 2 == 0:
            print('Epoch %d, Loss %f' % (epoch, float(loss)))
            print('Train accuracy: %f' % train_acc)
            print('Valid accuracy: %f' % valid_acc)
            print()

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')
    parser.add_argument('-lr', '--learning_rate', type=float)
    parser.add_argument('-ep', '--n_epochs', type=int)
    parser.add_argument('-mtm', '--momentum', type=float)
    parser.add_argument('-wd', '--weight_decay', type=float)
    parser.add_argument('-load', type=bool, default=False)
    # Put custom arguments here
    args = parser.parse_args()
    train(args)
