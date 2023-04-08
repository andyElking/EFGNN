import torch
import torch.nn as nn
import torch.optim as optim


# This is partially copied from one of the practicals (why fix what isn't broken...). Thanks!!

def train(model, data, num_epochs, use_edge_index=False, variable_lrs=None, layer_lr=0.001, alpha_lr=0.0001, wgt_dec=0,
          print_reports=True, noise_off_ep=-2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up the loss and the optimizer
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam([{"params": model.layer_params, "lr": layer_lr}, {"params": model.alpha_params, "lr": alpha_lr}],
                     weight_decay=wgt_dec)

    # layer_optimizer = optim.Adam(model.layer_params, lr=layer_lr, weight_decay=wgt_dec)
    # alpha_optimizer = optim.Adam(model.alpha_params, lr=alpha_lr, weight_decay=wgt_dec)

    # A utility function to compute the accuracy
    def get_acc(outs, y, mask):
        return (outs[mask].argmax(dim=1) == y[mask]).sum().float() / mask.sum()

    tt = {
        'alpha': [],
        'acc_val': [],
        'acc_test': [],
        'loss': [],
        'acc_train': []
    }

    best_acc_val = -1
    best_acc_test = 0
    for epoch in range(num_epochs):
        if (variable_lrs is not None) and epoch in variable_lrs:
            for i in [0, 1]:
                opt.param_groups[i]["lr"] = variable_lrs[epoch][i]
            if print_reports:
                print(f"Changed lrs to {variable_lrs[epoch]}")

        if epoch == noise_off_ep:
            model.noise_mult = 0
        data.x = data.x.to(device)
        data.y = data.y.to(device)
        data.edge_index = data.edge_index.to(device)
        try:
            data.edge_attr = data.edge_attr.to(device)
        except AttributeError:
            data.edge_attr = None
        try:
            data.edge_attr_matrix = data.edge_attr_matrix.to(device)
        except AttributeError:
            data.edge_attr_matrix = None
        opt.zero_grad()
        model.train()
        outs = model(data)
        loss = loss_fn(outs[data.train_mask], data.y[data.train_mask])
        loss.backward()
        opt.step()

        if epoch % 10 == 0:
            model.eval()
            outs = model(data).detach()
            model.train()
            acc_val = get_acc(outs, data.y, data.val_mask)
            acc_test = get_acc(outs, data.y, data.test_mask)
            acc_train = get_acc(outs, data.y, data.train_mask)
            tt['loss'].append(loss.item())
            tt['acc_val'].append(acc_val.item())
            tt['alpha'].append([float(f"{i:.3f}") for i in model.alpha.tolist()])
            tt['acc_train'].append(acc_train.item())
            if acc_val >= acc_val:
                best_acc_val = acc_val
                best_acc_test = acc_test
            if print_reports:
                print(f'[Epoch {epoch + 1}/{num_epochs}] Loss: {loss} | Train: {acc_train:.3f} | Val: {acc_val:.3f} | '
                      f'Test: {acc_test:.3f}')

    return best_acc_val, best_acc_test, tt
