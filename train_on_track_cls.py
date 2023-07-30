
import wandb
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pickle

from matplotlib import image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import TensorDataset

from cfgs.on_track_cls import OnTrackClsCfg
from dp_util.early_stopping import EarlyStopper
from models.on_track_cls import OnTrackClsNet
from dp_spec.img import preprocess_img
from dp_util.dataset import CustomDataset
from dp_util.training import valid

dir_auto_img = r'D:\file\data\racing\auto_img'


if __name__ == "__main__":
    cfg = OnTrackClsCfg(test_mode=False, dir_data='./')

    wandb.init(project=cfg.project, entity=cfg.entity,
               group=f'{cfg.user}_{cfg.model}', job_type="train",
               name= cfg.version)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device: {device}')

    # prepare train, valid and test dataloader
    ##################################################
    list_path = []
    list_lbl = []

    for file in os.listdir(os.path.join(dir_auto_img, 'on')):
        list_path.append(os.path.join(dir_auto_img, 'on', file))
        list_lbl.append(['on'])

    for file in os.listdir(os.path.join(dir_auto_img, 'off')):
        list_path.append(os.path.join(dir_auto_img, 'off', file))
        list_lbl.append(['off'])

    encoder = OneHotEncoder(handle_unknown='ignore')
    list_lbl_array = encoder.fit_transform(list_lbl).toarray()

    path_train, path_test, y_train, y_test = train_test_split(list_path, list_lbl_array, test_size=0.2, stratify=list_lbl,
                                                              shuffle=True)
    path_train, path_valid, y_train, y_valid = train_test_split(path_train, y_train, test_size=0.2, stratify=y_train,
                                                                shuffle=True)

    ts_y_train = torch.tensor(y_train, device=device)
    ts_y_valid = torch.tensor(y_valid, device=device)
    ts_y_test = torch.tensor(y_test, device=device)

    list_x_y_z = []
    for x, y, z in zip([torch.tensor(preprocess_img(image.imread(x)),
                                     dtype=torch.float32) for x in path_train], ts_y_train, path_train):
        list_x_y_z.append((x, y, z))
    train_loader = torch.utils.data.DataLoader(dataset=CustomDataset(list_x_y_z),
                                               batch_size=cfg.BATCH_SIZE,
                                               shuffle=True)
    list_x_y_z = []
    for x, y, z in zip([torch.tensor(preprocess_img(image.imread(x)),
                                     dtype=torch.float32) for x in path_valid], ts_y_valid, path_valid):
        list_x_y_z.append((x, y, z))
    valid_loader = torch.utils.data.DataLoader(dataset=CustomDataset(list_x_y_z),
                                               batch_size=cfg.BATCH_SIZE,
                                               shuffle=False)

    list_x_y_z = []
    for x, y, z in zip([torch.tensor(preprocess_img(image.imread(x)),
                                     dtype=torch.float32) for x in path_test], ts_y_test, path_test):
        list_x_y_z.append((x, y, z))
    test_loader = torch.utils.data.DataLoader(dataset=CustomDataset(list_x_y_z),
                                              batch_size=cfg.BATCH_SIZE,
                                              shuffle=False)

    # train and valid the model by early stopping
    ##################################################
    net = OnTrackClsNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=cfg.LR)
    es = EarlyStopper(cfg.md_name, smaller_is_better=True, patience=30, sf=cfg.SF)

    for epoch in range(500):  # loop over the dataset multiple times

        running_loss = 0.0
        total = 0

        for data in train_loader:
            inputs, labels, _ = data
            inputs = inputs.to(device)
            inputs = torch.unsqueeze(inputs, dim=1)

            total += labels.size(0)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        if not es.early_stop:

            train_avg_loss = running_loss / total
            valid_avg_loss, _ = valid(net, valid_loader, criterion,
                                      device, verbose=False)
            wandb.log({'train_avg_loss': train_avg_loss, 'valid_avg_loss': valid_avg_loss})

            es(valid_avg_loss, net)
        else:
            break

    test_avg_loss, _ = valid(net, test_loader, criterion, device)
    wandb.log({'test_avg_loss': test_avg_loss})

    with open(f'ont_hot_encoder_{cfg.version}.pkl', 'wb') as output:
        pickle.dump(encoder, output)
