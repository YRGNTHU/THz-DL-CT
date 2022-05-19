import argparse
import offset_new as offset
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle as pkl
import os
from torch.utils.data import random_split
from skimage.transform import iradon_sart, iradon
from dataloader import THz_dataloader
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from model_k_last import THz_SR_net
from os.path import splitext
from pathlib import Path
from scipy.stats import loguniform
from resnet import resnet152 as ResNet
from sklearn import preprocessing

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
base_root_dir = '' #please input your base root directory

def weighted_mse_loss(pred, target, weight):
    loss = nn.MSELoss()
    return loss(pred, target)

def arg_parser():
    parser = argparse.ArgumentParser(description='THz Reconstruction')
    parser.add_argument('--t_k_size', type=int, default=3, help='time domain kernel size')
    parser.add_argument('--sp_k_size', type=int, default=3, help='spatial domain kernel size')
    parser.add_argument('--batchSize', type=int, default=4, help='training batch size')
    parser.add_argument('--testBatchSize', type=int, default=4, help='testing batch size')
    parser.add_argument('--nEpochs', type=int, default=20, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
    parser.add_argument('--cuda', default=True, action='store_true', help='use cuda?')
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--ch',type=int,default=32,help='number of channel. Default=8')
    parser.add_argument('--pad_num', type=int, default=0, help='the number of noise padding')
    parser.add_argument('--scale', type=float, default=0.02, help='scale for label imbalance')
    parser.add_argument('--use_method', type=str, default='vgg16_klast')
    parser.add_argument('--out_dir', type=str, default=os.path.join(base_root_dir, 'output_dir/cross_validation/polarbear/'), help='')
    opt = parser.parse_args()

    return opt

def set_device_and_seed(if_cuda, seed):
    if if_cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    torch.manual_seed(seed)
    device = torch.device("cuda" if if_cuda else "cpu")
    return device



def gen_data_loader(root_dir, train_dir, target_dir, valid_train_dir, valid_target_dir, sp_test_data_dir, sp_test_target_dir, noise_file, opt):
    print('===> Loading datasets')
    all_dataset = THz_dataloader(root_dir, train_dir, target_dir, offset, noise_file, opt['pad_num'], False)
    valid_dataset = THz_dataloader(root_dir, valid_train_dir, valid_target_dir, offset, noise_file, opt['pad_num'], False)
    sp_test_dataset = THz_dataloader(root_dir, sp_test_data_dir, sp_test_target_dir, offset, noise_file, opt['pad_num'], False)
    training_data_loader = DataLoader(dataset=all_dataset, batch_size=opt['batchSize'], shuffle=True)
    testing_data_loader = DataLoader(dataset=valid_dataset, batch_size=opt['testBatchSize'], shuffle=False)

    sp_test_data_loader = DataLoader(dataset=sp_test_dataset, batch_size=1, shuffle= False)
    return training_data_loader, testing_data_loader, sp_test_data_loader


def build_model(device, sp_k_size=3, t_k_size=3, out_ch_tconv1=64, if_data_parallel=True):
    print("===> Building model")
    model = THz_SR_net(sp_k_size, t_k_size, out_ch_tconv1=out_ch_tconv1)
    if if_data_parallel and torch.cuda.device_count()>1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = model.cuda()
        model = nn.DataParallel(model)
    else:
        model = model.to(device)
    return model


def train(epoch, model, device, opt, training_data_loader, scale, optimizer, criterion):
    epoch_loss = 0
    loss_f = os.path.join(opt['out_dir'], "t{}_k{}/loss.txt".format(opt['t_k_size'],opt['sp_k_size']))
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = batch[0].to(device), batch[1].to(device)
        order = np.arange(input.shape[2])
        np.random.shuffle(order)
        mask = (target == 0.5).float()
        weight = torch.abs(mask - 1) + mask * scale
        for angle in order:
            optimizer.zero_grad()
            out = model(input[:,:,angle,:,:])
            loss = criterion(out[:, 0, :, 0], target[:,angle,:], weight[:, angle, :])
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()


        print("===> Epoch[{}]({}/{}): Loss: {:.6f}".format(epoch, iteration, len(training_data_loader), epoch_loss/iteration))



    f = open(loss_f,'a+')
    f.write("===> Epoch {} Complete: Training Loss: {:.6f}\n".format(epoch, epoch_loss / len(training_data_loader)))
    f.close()


    print("===> Epoch {} Complete: Avg. Loss: {:.6f}".format(epoch, epoch_loss / len(training_data_loader)))

    return epoch_loss/len(training_data_loader)

def test(epoch, model, criterion, opt, device, testing_data_loader):
    total_err = 0
    loss_f = os.path.join(opt['out_dir'], "t{}_k{}/loss.txt".format(opt['t_k_size'],opt['sp_k_size']))
    with torch.no_grad():
        idx = 0
        for batch in testing_data_loader:
            input, target = batch[0].to(device), batch[1].to(device)
            mask = (target == 0.5).float()
            weight = torch.abs(mask - 1) + mask * opt['scale']
            for angle in range(input.shape[2]):
                prediction = model(input[:,:,angle,:,:])
                err = criterion(prediction[:, 0, :, 0], target[:,angle,:], weight[:, angle, :])
                total_err += err.item()

    f = open(loss_f,'a+')
    f.write("===> Valid Loss: {:.6f}\n".format(total_err / len(testing_data_loader)))
    f.close()


    print("===> Avg. Loss: {:.6f}".format(total_err / len(testing_data_loader)))

    return total_err/len(testing_data_loader)


def checkpoint(epoch, model, model_out_base_path, opt):
    model_out_path = os.path.join(model_out_base_path, "model_epoch_ch{}_b{}_lr{}_{}.pth".format(opt['ch'], opt['batchSize'], opt['lr'], epoch))
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

def sp_test(epoch, model, opt, device, sp_test_data_loader, image_range, model_out_base_path):
    loss = 0
    count = 0
    sp_loss = 0
    loss_f = os.path.join(model_out_base_path, "sp_loss.txt".format(opt['t_k_size'],opt['sp_k_size']))
    with torch.no_grad():
        for batch in sp_test_data_loader:
            input, target = batch[0].to(device), batch[1].to(device)
            number = int(splitext(batch[2][0])[0].split('_')[2])
            tmp = []
            for angle in range(input.shape[2]):
                prediction = model(input[:, :, angle, :, :])
                prediction = prediction.cpu().detach().numpy()
                tmp.append(prediction)
            result = np.squeeze(np.stack(tmp)).astype(np.double)
            result_sart = iradon_sart(result.T, theta = np.arange(30)* 6)
            result_sart = result_sart[45:-45, 45:-45]
            target = np.squeeze(target.cpu().detach().numpy()).astype(np.double)
            target_sart = iradon_sart(target.T, theta = np.arange(30)* 6)
            target_sart = target_sart[45:-45, 45:-45]
            result_sart_binary = km_separate(result_sart)
            target_sart_binary = km_separate(target_sart)

            # do normalization

            shape1, shape2 = result_sart.shape[0], result_sart.shape[1]

            result_sart_ = np.expand_dims(result_sart.reshape(-1), axis=0)
            result_sart_ = preprocessing.normalize(result_sart_)
            result_sart_rms = result_sart_.squeeze().reshape(shape1, shape2)

            target_sart_ = np.expand_dims(target_sart.reshape(-1), axis=0)
            target_sart_ = preprocessing.normalize(target_sart_)
            target_sart_rms = target_sart_.squeeze().reshape(shape1, shape2)

            if number >= image_range[0] and number <= image_range[1]:
                count += 1
                single_loss = np.sum(np.abs(result_sart_binary - target_sart_binary))/(result_sart_binary.shape[0]* result_sart_binary.shape[1])
                loss += single_loss

                single_sp_loss = np.sum((result_sart_rms - target_sart_rms)**2)
                sp_loss += single_sp_loss

        img_file = os.path.join(model_out_base_path, 'tmp_res')
        if not os.path.exists(img_file):
            os.makedirs(img_file)
        plt.imsave(os.path.join(img_file, 'result_sart_{}_{}.png'.format(number, epoch)), result_sart)
        plt.imsave(os.path.join(img_file, 'target_sart_{}_{}.png'.format(number, epoch)), target_sart)

        plt.imsave(os.path.join(img_file, 'result_sart_binray_{}_{}.png'.format(number, epoch)), result_sart_binary)
        plt.imsave(os.path.join(img_file, 'target_sart_binary_{}_{}.png'.format(number, epoch)), target_sart_binary)

    f = open(loss_f, 'a+')
    f.write("===>{} epoch: Avg. sp rms Loss: {:.6f}\n".format(epoch, sp_loss/ count))
    f.write("===>{} epoch: Avg. sp kmeans Loss: {:.6f}\n".format(epoch, loss/ count))
    print("===>{} epoch: Avg. sp Loss: {:.6f}".format(epoch, loss/ count))


    return loss/ count

def km_separate(data): #This function only uses for the old version of DLCT.
    km = KMeans(2)
    km_img = km.fit_predict(data.reshape(-1,1)).reshape(data.shape[0],data.shape[1])
    if np.sum(km_img==0) < np.sum(km_img==1):
        km_img = -(km_img-1)
    return km_img

if __name__ == "__main__":

    print('Start run code')
    opt = vars(arg_parser())
    opt['out_dir'] = os.path.join(opt['out_dir'], opt['use_method'] + '_lr' + str(opt['lr']))
    model_out_base_path = os.path.join(opt['out_dir'], "t{}_k{}".format(opt['t_k_size'],opt['sp_k_size']))
    if not os.path.exists(model_out_base_path):
        os.makedirs(model_out_base_path)
    param_path = os.path.join(model_out_base_path, "param.txt")

    f = open(param_path, 'a+')
    f.write(str(opt))
    f.close()


    if_data_parallel = False
    start_epoch_num = 1

    opt['root_dir'] = os.path.join(base_root_dir, 'raw')
    opt['train_dir'] = ["dna_raw","eevee_raw","robot_raw","deer_raw", "insidehole_raw", "box_raw"]
    opt['target_dir'] = ['finish_gt_dna',"finish_gt_eevee","finish_gt_robot","finish_gt_deer", "finish_gt_insidehole", "finish_gt_box"]
    opt['valid_train_dir'] = ['skull_raw']
    opt['valid_target_dir'] = ['finish_gt_skull']
    opt['sp_test_data_dir'] = ['polarbear_raw']
    opt['sp_test_target_dir'] = ['finish_gt_polarbear']
    opt['noise_file'] = os.path.join(base_root_dir, 'code/noise.npy')
    opt['test_image_range'] = [32, 250]
    opt['if_data_parallel'] = if_data_parallel
    scale = opt['scale']
    opt['param_path'] = param_path

    print("set device and seed")

    print("building model")
    config = opt
    device = set_device_and_seed(if_cuda=config['cuda'], seed = config['seed'])
    model = build_model(device, opt['sp_k_size'], opt['t_k_size'], opt['ch'], if_data_parallel)


    print("generate data")
    training_data_loader, testing_data_loader, sp_test_data_loader = gen_data_loader(config['root_dir'], config['train_dir'], config['target_dir'], config['valid_train_dir'], config['valid_target_dir'], config['sp_test_data_dir'], config['sp_test_target_dir'], config['noise_file'], opt)

    print("set optimizer and criterion")
    optimizer = optim.Adam(model.parameters(), lr=opt['lr'])
    criterion = weighted_mse_loss


    print("start training")


    for epoch in range(start_epoch_num, opt['nEpochs'] + start_epoch_num):
        train(epoch, model, device, opt, training_data_loader, scale, optimizer, criterion)
        test(epoch, model, criterion, opt, device, testing_data_loader)
        sp_test(epoch, model, opt, device, sp_test_data_loader, config['test_image_range'], model_out_base_path)
        checkpoint(epoch, model, model_out_base_path, opt)

    print("Done training and testing")
