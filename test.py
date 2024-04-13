import argparse
import torch
import pickle
import os, copy
from dataset.dataloader import get_dataloader, get_transform
from dataset.dataset import SingleDomainData, SingleClassData
from model.model import MutiClassifier, MutiClassifier_, resnet18_fast, resnet50_fast, ConvNet
from optimizer.optimizer import get_optimizer, get_scheduler
from laplace import get_hessian, estimate_variance, predict_lap
from torch.distributions.multivariate_normal import MultivariateNormal
from loss_lap import Entropy
from loss.OVALoss import OVALoss
from train.test import eval
from util.log import log, save_data
from torch.nn import functional as F
from torch.utils.data import DataLoader
from util.ROC import generate_OSCR
from util.util import ForeverDataIterator, ConnectedDataIterator, split_classes
import numpy as np
import random

def compute_laplace(args, model, train_loader):
        # compute the laplace approximations
        hessians = get_hessian(args, model, train_loader)
        var0 = torch.tensor(1e-3).float().cuda() # 5e-4, 0.001, 0.01, 0.2656
        M_W, U, V = estimate_variance(args, var0, hessians)
        print(f'Saving the hessians...')
        M_W, U, V = M_W.detach().cpu().numpy(), U.detach().cpu().numpy(), \
                            V.detach().cpu().numpy()
        return M_W, U, V
        # np.save(os.path.join(args.save_dir, args.dataset + "_llla.npy"), [M_W, U, V])

def predict_lap(args, model, val_loader, M_W, U, V, n_samples=100, apply_softmax=True):
    M_W, U, V = torch.from_numpy(M_W).cuda(), \
                torch.from_numpy(U).cuda(), \
                torch.from_numpy(V).cuda()
    model.eval()

    py = []
    preds = []
    targets = []
    max_probs = []
    
    from tqdm.auto import tqdm 
    for iter in tqdm(range(len(val_loader[0]))):
        x, y = next(val_loader[0])
        x, y = x.cuda(), y.cuda()
        x = x.cuda(args.gpu)
        targets.append(y)
        phi = model.net(x).view(x.shape[0], -1)

        mu_pred = phi @ M_W
        Cov_pred = torch.diag(phi @ U @ phi.t()).view(-1, 1, 1) * V.unsqueeze(0)

        post_pred = MultivariateNormal(mu_pred, Cov_pred)

        # MC-integral
        py_ = 0

        for _ in range(n_samples):
            f_s = post_pred.rsample() / 1. # {(0.1, 0.5), (0.235, 0.4)}
            py_ += torch.softmax(f_s, 1) if apply_softmax else f_s
        
        py_ /= n_samples
        max_prob, pred = torch.max(py_, dim=1)
        preds.append(pred)
        max_probs.append(max_prob.view(-1))
        py.append(py_)
    
    py = torch.cat(py, dim=0)
    preds = torch.cat(preds, dim=0).detach().cpu()
    max_probs = torch.cat(max_probs, dim=0).detach().cpu().numpy()
    targets = torch.cat(targets, dim=0)
    correct = preds.eq(targets).numpy() # boolean vector
    entropy = Entropy(py.detach().cpu(), reduction='sum').numpy()

    out = {}
    out['entropies'] = entropy
    out['max_probs'] = max_probs
    out['correct'] = correct
    out['targets'] = targets

    return out
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', default='PACS')
    parser.add_argument('--source-domain', nargs='+', default=['photo', 'cartoon', 'art_painting'])
    parser.add_argument('--target-domain', nargs='+', default=['sketch'])
    parser.add_argument('--known-classes', nargs='+', default=['dog', 'elephant', 'giraffe', 'horse', 'guitar', 'house',])
    parser.add_argument('--unknown-classes', nargs='+', default=['person'])

    
    # parser.add_argument('--dataset', default='OfficeHome')
    # parser.add_argument('--source-domain', nargs='+', default=['Art', 'Clipart', 'Product'])
    # parser.add_argument('--target-domain', nargs='+', default=['RealWorld'])
    # parser.add_argument('--known-classes', nargs='+', default=['Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 
    #     'Bottle', 'Bucket', 'Calculator', 'Calendar', 'Candles', 
    #     'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 
    #     'Desk_Lamp', 'Drill', 'Eraser', 'Exit_Sign', 'Fan', 
    #     'File_Cabinet', 'Flipflops', 'Flowers', 'Folder', 'Fork', 
    #     'Glasses', 'Hammer', 'Helmet', 'Kettle', 'Keyboard',  
    #     'Knives', 'Lamp_Shade', 'Laptop', 'Marker', 'Monitor', 
    #     'Mop', 'Mouse', 'Mug', 'Notebook', 'Oven',
        
    #     ])
    # parser.add_argument('--unknown-classes', nargs='+', default=[      
    #     'Pan', 'Paper_Clip', 'Pen', 'Pencil', 'Postit_Notes', 
    #     'Printer', 'Push_Pin', 'Radio', 'Refrigerator', 'Ruler', 
    #     'Scissors', 'Screwdriver', 'Shelf', 'Sink', 'Sneakers', 
    #     'Soda', 'Speaker', 'Spoon', 'TV', 'Table', 
    #     'Telephone', 'ToothBrush', 'Toys', 'Trash_Can', 'Webcam'
    #     ])

    # parser.add_argument('--dataset', default='DigitsDG')
    # parser.add_argument('--source-domain', nargs='+', default=['mnist', 'mnist_m', 'svhn'])
    # parser.add_argument('--target-domain', nargs='+', default=['syn'])
    # parser.add_argument('--known-classes', nargs='+', default=['0', '1', '2', '3', '4', '5'])
    # parser.add_argument('--unknown-classes', nargs='+', default=['6', '7', '8', '9'])

    parser.add_argument('--no-crossval', action='store_true')
    parser.add_argument('--gpu', default=0)
    parser.add_argument('--batch-size', type=int, default=8)

    parser.add_argument('--net-name', default='resnet50')
    parser.add_argument('--optimize-method', default="SGD")
    parser.add_argument('--schedule-method', default='StepLR')
    parser.add_argument('--num-epoch', type=int, default=10000)
    parser.add_argument('--eval-step', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--meta-lr', type=float, default=0.01)
    parser.add_argument('--nesterov', action='store_true')
    parser.add_argument('--without-bcls', action='store_true')
    parser.add_argument('--share-param', action='store_true')

    parser.add_argument('--save-dir', default='save')
    parser.add_argument('--save-name', default='demo')
    parser.add_argument('--save-best-test', action='store_true')
    parser.add_argument('--save-later', action='store_true')

    parser.add_argument('--num-epoch-before', type=int, default=0)
    
    args = parser.parse_args()

    # It can be used to replace the following code, but the editor may take it as an error.
    # locals().update(vars(args))

    # It can be replaced by the preceding code.
    dataset = args.dataset
    source_domain = sorted(args.source_domain)
    target_domain = sorted(args.target_domain)
    known_classes = sorted(args.known_classes)
    unknown_classes = sorted(args.unknown_classes)
    crossval = not args.no_crossval   
    gpu = args.gpu
    batch_size = args.batch_size
    net_name = args.net_name
    optimize_method = args.optimize_method
    schedule_method = args.schedule_method
    num_epoch = args.num_epoch
    eval_step = args.eval_step
    lr = args.lr
    meta_lr = args.meta_lr
    nesterov = args.nesterov
    without_bcls = args.without_bcls
    share_param = args.share_param
    save_dir = args.save_dir
    save_name = args.save_name   
    save_later = args.save_later
    save_best_test = args.save_best_test
    num_epoch_before = args.num_epoch_before

    torch.set_num_threads(1)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dataset == 'PACS':
        train_dir = 'data_list/pacs/pacs_data/pacs_data_train'
        val_dir = 'data_list/pacs/pacs_data/pacs_data_crossval'
        test_dir = ['data_list/pacs/pacs_data/pacs_data_train', 'data_list/pacs/pacs_data/pacs_data_crossval']
        sub_batch_size = batch_size // 2
        small_img = False
    elif dataset == 'OfficeHome':
        train_dir = ''
        val_dir = ''
        test_dir = ''
        sub_batch_size = batch_size // 4
        small_img = False
    elif dataset == "DigitsDG":
        train_dir = ''
        val_dir = ''
        test_dir = ''
        sub_batch_size = batch_size // 2
        small_img = True
        

    log_path = os.path.join(save_dir, 'log', save_name + '_train.txt')
    param_path = os.path.join(save_dir, 'param', save_name + '.pkl')
    model_val_path = os.path.join(save_dir, 'model', 'val', save_name + '.tar')
    model_test_path = os.path.join(save_dir, 'model', 'test', save_name + '.tar')
    renovate_step = int(num_epoch*0.6) if save_later else 0

    log('GPU: {}'.format(gpu), log_path)

    log('Loading path...', log_path)

    log('Save name: {}'.format(save_name), log_path)
    log('Save best test: {}'.format(save_best_test), log_path)
    log('Save later: {}'.format(save_later), log_path)

    with open(param_path, 'wb') as f: 
        pickle.dump(vars(args), f, protocol=pickle.HIGHEST_PROTOCOL)

    log('Loading dataset...', log_path)

    num_domain = len(source_domain)
    num_classes = len(known_classes)

    class_index = [i for i in range(num_classes)]
    group_length = (num_classes-1) // 10 + 1

    if dataset == "OfficeHome" and len(unknown_classes) == 0:
        group_length = 6

    log('Group length: {}'.format(group_length), log_path)
    
    group_index = [i for i in range((num_classes-1)//group_length + 1)]
    num_group = len(group_index)

    domain_specific_loader = []
    for domain in source_domain:       
        dataloader_list = []
        if num_classes <= 10:
            for i, classes in enumerate(known_classes):
                scd = SingleClassData(root_dir=train_dir, domain=domain, classes=classes, domain_label=-1, classes_label=i, transform=get_transform("train", small_img=small_img))
                loader = DataLoader(dataset=scd, batch_size=sub_batch_size, shuffle=True, drop_last=True, num_workers=1)
                dataloader_list.append(loader)
        else:
            classes_partition = split_classes(classes_list=known_classes, index_list=class_index, n=group_length)
            for classes, class_to_idx in classes_partition:
                sdd = SingleDomainData(root_dir=train_dir, domain=domain, classes=classes, domain_label=-1, get_classes_label=True, class_to_idx=class_to_idx, transform=get_transform("train", small_img=small_img))
                loader = DataLoader(dataset=sdd, batch_size=sub_batch_size, shuffle=True, drop_last=True, num_workers=1)
                dataloader_list.append(loader)

        domain_specific_loader.append(ConnectedDataIterator(dataloader_list=dataloader_list, batch_size=batch_size))
    
    if crossval:
        val_k = get_dataloader(root_dir=val_dir, domain=source_domain, classes=known_classes, batch_size=batch_size, get_domain_label=False, get_class_label=True, instr="val", small_img=small_img, shuffle=False, drop_last=False, num_workers=4)
    else:
        val_k = None
    
    test_k = get_dataloader(root_dir=test_dir, domain=target_domain, classes=known_classes, batch_size=batch_size, get_domain_label=False, get_class_label=True, instr="test", small_img=small_img, shuffle=False, drop_last=False, num_workers=4)
    if len(unknown_classes) > 0:
        test_u = get_dataloader(root_dir=test_dir, domain=target_domain, classes=unknown_classes, batch_size=batch_size, get_domain_label=False, get_class_label=False, instr="test", small_img=small_img, shuffle=False, drop_last=False, num_workers=4)   
    else:
        test_u = None

    log('DataSet: {}'.format(dataset), log_path)
    log('Source domain: {}'.format(source_domain), log_path)
    log('Target domain: {}'.format(target_domain), log_path)
    log('Known classes: {}'.format(known_classes), log_path)
    log('Unknown classes: {}'.format(unknown_classes), log_path)
    log('Batch size: {}'.format(batch_size), log_path)
    log('CrossVal: {}'.format(crossval), log_path)
    log('Loading models...', log_path)

#################################################################### model loading
    muticlassifier = MutiClassifier
    model = muticlassifier(net=resnet50_fast(), num_classes=num_classes, feature_dim=2048)
    model.to(device=device)
    
    M_W, U, V = compute_laplace(args, model, domain_specific_loader)

    out = predict_lap(args, model,domain_specific_loader, M_W, U, V)
    
    
    
    