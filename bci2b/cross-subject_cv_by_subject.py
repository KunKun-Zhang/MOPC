import numpy as np
import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import random
import argparse
import utils
from utils import data_augment
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import EEGNet, EEGNet_features



current_file_path = os.path.abspath(__file__)
current_folder_path = os.path.dirname(current_file_path)

def seed_torch(seed=2024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)    # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_data(args, session_type):
    logger.debug("=========source domain data===========")
    persons=[1, 2, 3, 4, 5, 6, 7, 8, 9]  # 
    source_person = [i for i in persons if i != args.person_id]

    data_X_train_all = {}
    data_Y_train_all = {}
    ea_bar_list = []
    
    for idex, person_i in enumerate(source_person):
        if session_type == 'TE':
            person_session = [f"B0{person_i}0{i}T.gdf" if i not in [4, 5] else f"B0{person_i}0{i}E.gdf" for i in range(1, 6)]
        elif session_type == 'T':
            person_session = [f"B0{person_i}0{i}T.gdf" for i in range(1, 4)]
        
        person_data = []
        person_label = []
        
        for idx, session_file_path in enumerate(person_session):
            logger.debug("session {} loding ...".format(session_file_path))
            
            file_path = current_folder_path + "/BCI2b/" + session_file_path
            if 'T.gdf' in session_file_path:
                mi_left, mi_right = utils.get_data_bci2b_class_2(file_path, args.channel_roi)
            elif 'E.gdf' in session_file_path:
                mi_left, mi_right = utils.get_data_bci2b_class_2_E(file_path, args.channel_roi)
            
            data1_train = data_augment(mi_left, args)
            data2_train = data_augment(mi_right, args)
            data_X_train = np.vstack((data1_train, data2_train))

            def Y_get(num_1, num_2):
                Y1 = np.full(shape=(num_1,), fill_value=0, dtype='float64')
                Y2 = np.full(shape=(num_2,), fill_value=1, dtype='float64')

                data_Y = np.concatenate((Y1, Y2))
                return data_Y
            data_Y_train = Y_get(len(np.array(data1_train)), len(np.array(data2_train)))
            logger.debug("session_train_data.shape: %s", data_X_train.shape)
            logger.debug("session_train_label.shape: %s", data_Y_train.shape)
            
            if idx == 0:
                person_data = data_X_train
                person_label = data_Y_train
            else:
                person_data = np.vstack((person_data, data_X_train))            # (400, 3, 1000)
                person_label = np.concatenate((person_label, data_Y_train))     # (400,)
        
        person_data = utils.data_norm(person_data)                         # min-max 归一化
        person_data, ea_bar_i = utils.preprocess_ea(person_data)           # !!!! EA
        ea_bar_list.append(ea_bar_i)

        data_X_train_all[f'{idex}'] = person_data
        data_Y_train_all[f'{idex}'] = person_label

    # 保存所有训练集的 EA_bar 用于测试集
    r_bars = {}
    ea_bar_array = np.array(ea_bar_list)
    train_ea_bar = np.mean(ea_bar_array, axis=0)
    r_bars['train_ea_bar'] = train_ea_bar

    return data_X_train_all, data_Y_train_all, r_bars


def transform_dataloader(TORCH_X, TORCH_Y, batch_size):
    # 增加一维, input:[batch, 1, channel, samples]
    TORCH_X = np.expand_dims(TORCH_X, 1)
    X_train = torch.from_numpy(TORCH_X).to(torch.float32)
    y_train = torch.from_numpy(TORCH_Y)
  
    dataset = Data.TensorDataset(X_train, y_train)
    dataloader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last = True)
    return dataloader


def model_train(X_TRAIN, Y_TRAIN, X_VALID, Y_VALID, model_save_path, args):
    # set device
    device = torch.device("cuda:" + args.gpu_id if torch.cuda.is_available() else "cpu")
    num, chs, samples = X_TRAIN.shape
    
    dataloader_train = transform_dataloader(X_TRAIN, Y_TRAIN, args.batch_size)
    dataloader_valid = transform_dataloader(X_VALID, Y_VALID, args.batch_size)
    
    if args.model_type == "EEGNet":
        model = EEGNet.EEGNet(num_classes=args.classes, chans=chs, samples=samples, kernLength=250//2, dropoutRate=0.25).to(device)
    elif args.model_type == 'EEGNet_features':
        model = EEGNet_features.EEGNet(num_classes=args.classes, chans=chs, samples=samples, kernLength=250//2, dropoutRate=0.25).to(device)

    utils.init_weights(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_func = nn.CrossEntropyLoss().to(device)
    # optimizer 按照余弦退火的方式调整学习率
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-4)
    
    best_acc = 0 
    best_epoch = 0
    train_correct = 0
    val_acc_all = [] 
    train_acc_all = []
    loss_item_train = []  
    loss_item_val = []
    # Early Stopping parameters
    patience = args.patience
    trigger_times = 0
 
    for epoch in range(args.epochs):
        
        # 训练阶段
        model.train()
        train_correct = 0
        total_train = 0
        epoch_loss = 0
        for step, (b_x, b_y) in enumerate(dataloader_train):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            optimizer.zero_grad()
            output = model(b_x)
            loss = loss_func(output, b_y.long())
            loss.backward()
            optimizer.step()
            
            _, pred = torch.max(output, 1)
            epoch_loss += loss.item() * b_x.size(0)
            train_correct += (pred == b_y).sum().item()
            total_train += b_y.size(0)
        
        scheduler.step()
        epoch_loss /= total_train
        train_acc = train_correct / total_train
        loss_item_train.append(epoch_loss)
        train_acc_all.append(train_acc)
        
        logger.debug(f'Epoch [{epoch + 1}/{args.epochs}], Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.4f}')
        
        # 验证阶段
        model.eval() 
        val_correct = 0
        total_val = 0
        val_loss = 0
        
        with torch.no_grad():
            for b_x, b_y in dataloader_valid:
                b_x = b_x.to(device)
                b_y = b_y.to(device)
                output = model(b_x)
                loss = loss_func(output, b_y.long())
                
                _, pred = torch.max(output, 1)
                val_loss += loss.item() * b_x.size(0)
                val_correct += (pred == b_y).sum().item()
                total_val += b_y.size(0)
        
        val_loss /= total_val
        val_acc = val_correct / total_val
        loss_item_val.append(val_loss)
        val_acc_all.append(val_acc)

        logger.debug(f'Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}')

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            trigger_times = 0
            # 保存最佳模型
            torch.save(model.state_dict(), model_save_path)
            logger.debug(f'Best model saved at epoch {epoch + 1}.')
        else:
            trigger_times += 1
            if trigger_times >= patience:
                logger.debug(f'Early Stopping triggered at epoch {epoch + 1}.')
                break
    
    logger.debug(f'Training complete. Best Validation Acc: {best_acc:.4f}, Best Epoch: {best_epoch + 1}')

    return loss_item_train, loss_item_val, train_acc_all, val_acc_all, best_epoch + 1, best_acc


# 计算 ITR
def itr_cal(itr_acc, itr_data_len, type_class):
    itr = 60*((np.log2(type_class)+itr_acc*np.log2(itr_acc)) + (1-itr_acc)*np.log2((1-itr_acc)/(type_class-1)))/(int(itr_data_len)/250)
    return itr


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='MI_find_hyper')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001)')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--seed', type=int, default=2024, metavar='S', help='random seed')
    parser.add_argument('--epochs', type=int, default=1000, help='max epoch')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--patience', type=int, default=1000, help='early-stopping patience (1000 = no early stopping)')
    
    parser.add_argument('--train_samples', type=int, default=1000, help='EEG MI training data length')
    parser.add_argument('--fold', type=int, default=5, help="ensemble models")  
    parser.add_argument('--classes', type=int,default=2, help="N classes")
    parser.add_argument('--person_id', type=int, default=1 ,help="Choose the person")
    
    parser.add_argument('--channel_roi', type=str, default='all' ,choices=['all', 'A'],help="Choose a data augmentation method you want to explore")
    parser.add_argument('--augmentation', type=str, default='none' ,choices=['none', 'overlap'],help="Choose a augmentation method you want to explore")
    parser.add_argument('--overlap_stride', type=int, default=50, help='overlap_points')
    parser.add_argument('--model_type', type=str, default='EEGNet_features' ,choices=['EEGNet_features', 'EEGNet'],help="Choose a backbone you want to explore")
    parser.add_argument('--model_save_path', type=str, default='/run_cross_subject_{}_{}_cv_class_2_T_and_E_by_subject/', help="output directory (format string: model_type, train_samples)")
    parser.add_argument('--sessions', type=str, default='TE', choices=['T', 'TE'], help="source sessions pooled for training: 'TE' = all five sessions (paper), 'T' = the three training sessions only")

    args = parser.parse_args()
    args.model_save_path = args.model_save_path.format(args.model_type, args.train_samples)
    os.makedirs(current_folder_path + args.model_save_path, exist_ok=True)
    
    logger_path = current_folder_path + args.model_save_path + f'{args.person_id}_run.log'
    logger = utils._logger(logger_path)
    logger.debug(args)
    logger.debug('use {}'.format("cuda" if torch.cuda.is_available() else "cpu"))
    
    data_X_train, data_Y_train, ea_bars = get_data(args, args.sessions)   # paper: all five sessions of every source subject

    ea_path = current_folder_path + args.model_save_path + '{}_ea_bars_{}.npz'.format(args.person_id, args.train_samples)
    np.savez(ea_path, **ea_bars)

    logger.debug("===================start training====================")
    seed_torch(args.seed)
    itr_all=[]
    acc_all=[]
    
    persons = range(8)
    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    for idx, (train_index, val_index) in enumerate(kf.split(persons)):
        fold = idx + 1
        logger.debug(f"=========begin train the {fold}-fold===========")
        
        X_train_fold, X_val_fold, y_train_fold, y_val_fold = [], [], [], []

        for idx in train_index:
            person_data = data_X_train[f"{idx}"]
            person_label = data_Y_train[f"{idx}"]
            X_train_fold.extend(person_data)
            y_train_fold.extend(person_label)

        for idx in val_index:
            person_data = data_X_train[f"{idx}"]
            person_label = data_Y_train[f"{idx}"]
            X_val_fold.extend(person_data)
            y_val_fold.extend(person_label)

        X_train_fold = np.array(X_train_fold)
        X_val_fold = np.array(X_val_fold)
        y_train_fold = np.array(y_train_fold)
        y_val_fold = np.array(y_val_fold)
        
        logger.debug("%d-fold - train_peron: %s", fold, train_index)
        logger.debug("%d-fold - X_train_fold.shape: %s", fold, X_train_fold.shape)
        logger.debug("%d-fold - y_train_fold.shape: %s", fold, y_train_fold.shape)
        logger.debug("%d-fold - val_peron: %s", fold, val_index)
        logger.debug("%d-fold - X_val_fold.shape: %s", fold, X_val_fold.shape)
        logger.debug("%d-fold - y_val_fold.shape: %s", fold, y_val_fold.shape)
        
        model_name = current_folder_path + args.model_save_path + str(args.person_id) + '_' + str(fold) + '.pth'
        losstrain, lossval, train_acc_all, val_acc_all, Best_EPOCH, Best_ACC = model_train(
            X_TRAIN=X_train_fold, Y_TRAIN=y_train_fold, X_VALID=X_val_fold, Y_VALID=y_val_fold, model_save_path=model_name, args=args)
        itr_i=itr_cal(Best_ACC, args.train_samples, args.classes)
        itr_all.append(itr_i)
        acc_all.append(Best_ACC)
        logger.debug("{}-fold validation ACC : {}, ITR : {} , best epoch : {}\n".format(fold, Best_ACC, itr_i, Best_EPOCH))
    
    logger.debug("\n===================================================")
    logger.debug("mean_itr : {}, mean_acc : {}".format(np.mean(itr_all), np.mean(acc_all)))
    logger.debug("===================end training====================")
    
        