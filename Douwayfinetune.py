import numpy as np
import torch
from torch.autograd import Variable
import os
import glob
import h5py
import pdb
import random
import time
import configs
import backbone
from data.datamgr import SimpleDataManager
from data import feature_loader
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.baselinevae import DisentangleNet
from methods.douwayvae import DWNet
from io_utils import model_dict, parse_args, get_resume_file, get_best_file, get_assigned_file 
from utils import l2_norm
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import entropy
from scipy.special import softmax
from confusionmatrix import plot_confusion_matrix
def aug_features(all_cls_feature, mu, logvar, feature_map, model, aug_per_sample=2, n_way=5, n_shot=1, feat_dim=512):
    all_cls_feature = all_cls_feature.view(n_way, n_shot, feat_dim)
    # print(aug_per_sample)
    mu = mu.view(n_way, n_shot, feat_dim)
    logvar = logvar.view(n_way, n_shot, feat_dim)
    aug_features = torch.zeros((n_way, aug_per_sample*n_shot, feat_dim))
    aug_y= torch.from_numpy(np.repeat(range( n_way ), aug_per_sample*n_shot))
    aug_y = Variable(aug_y).cuda()
    for cls in range(n_way):  
        cls_feature = all_cls_feature[cls,:,:]
        cls_feature = cls_feature.repeat(aug_per_sample, 1)
        cls_mu = mu[cls,:,:]
        #cls_mu = cls_mu.mean(0, True)
        cls_mu = cls_mu.repeat(aug_per_sample, 1)
        cls_logvar = logvar[cls,:,:]
        #cls_logvar = cls_logvar.mean(0, True)
        cls_logvar = cls_logvar.repeat(aug_per_sample, 1)
        #cls_invar_feature = torch.randn_like(cls_feature) 
        cls_invar_feature,_ = model.reparameterize(cls_mu, cls_logvar)
        #cls_feature_map = feature_map[cls,:,:,:]
        #cls_feature_map = cls_feature_map.unsqueeze(0)
        #cls_invar_feature = torch.randn_like(cls_feature) * 0.2
        aggr_feature = cls_feature + cls_invar_feature  
        #recon_feature = model.forward_d(aggr_feature)
        #recon_loss = model.superloss(recon_feature, cls_feature_map)
        #cls_aug_feature = model.cls_fc(recon_feature)
        cls_aug_feature = aggr_feature
        aug_features[cls, :, :] = cls_aug_feature
    aug_features = aug_features.view(n_way*aug_per_sample*n_shot, -1)
    return aug_features.detach().cuda(), aug_y

# def channeltrans(x):
#     k = 1.8
#     x= torch.sign(x)/((torch.log(1/abs(x)+1))**k)
#     return x

def finetune_backbone(model, img_data_file, n_way, n_support, feat_dim, aug=False, n_query=10):

    class_list = img_data_file.keys()
    select_class = random.sample(class_list, n_way)
    select_class = sorted(select_class, reverse=True)
    img_all = []
    #print(select_class)
    for cl in select_class:
        #print(cl)
        img_data = img_data_file[cl]
        img_data = np.array(img_data)
        perm_ids = np.random.permutation(len(img_data)).tolist()
        perm_ids = perm_ids + perm_ids
        img_all.append( [ np.squeeze( img_data[perm_ids[i]]) for i in range(n_support+n_query) ] )     # stack each batch
    
    # samples images for support set and query set
    img_all = torch.from_numpy(np.array(img_all))
    img_all = Variable(img_all).cuda()
    [c, h, w] = img_all[0][0].shape

    x_support = img_all[:, :n_support,:,:,:]
    x_query = img_all[:,n_support:,:,:,:]

    x_support   = x_support.contiguous().view(n_way* n_support, c, h, w)
    x_query     = x_query.contiguous().view(n_way* n_query, c, h, w)

    y_support = torch.from_numpy(np.repeat(range( n_way ), n_support ))
    y_support = Variable(y_support).cuda()
    time_start = time.time()
    z_support,mu, logvar,mu2,logvar2,feature_map = model(x_support) 
    #print(z_support.size())
    z_query,_, _,_,_,_ = model(x_query) 
    #z_support = model(x_support).detach().cuda()
    z_support = z_support.view(n_way*n_support, -1).detach().cuda()
    # agg_support = agg_support.view(n_way*n_support, -1).detach().cuda()
    z_query = z_query.view(n_way*n_query, -1).detach().cuda()
    # agg_query = agg_query.view(n_way*n_query, -1).detach().cuda()
    #z_mean = torch.mean(z_support, dim=1)
    #_, z_mean = l2_norm(z_mean)
    # z_support = channeltrans((z_support))
    # agg_support = channeltrans((agg_support))
    # z_query = channeltrans((z_query))
    # agg_query = channeltrans((agg_query))
    #z_query = model(x_query).detach().cuda()
      #z_query = z_query.view(n_way, n_query, -1)

    #aug_z = recon_z.view(n_way*n_support, -1).detach().cuda()
    #aug_y = y_support.clone()
    #z_support_all = z_support.view(n_way*n_support, -1)
    feat_dim = 640
    y_query = np.repeat(range( n_way ), n_query )
    aug_z, aug_y = aug_features(z_support, mu, logvar, feature_map, model, n_shot=n_support, aug_per_sample=4, feat_dim=feat_dim)
    aug_z2, aug_y2 = aug_features(z_support, mu2, logvar2, feature_map, model, n_shot=n_support, aug_per_sample=4, feat_dim=feat_dim)
    # aug_z3, aug_y3 = aug_features(z_support, mu3, logvar3, feature_map, model, n_shot=n_support, aug_per_sample=4, feat_dim=feat_dim)
    # aug_z4, aug_y4 = aug_features(z_support, mu4, logvar4, feature_map, model, n_shot=n_support, aug_per_sample=4, feat_dim=feat_dim)
    #aug_z, aug_y = trans_features(aug_z, aug_y, z_query, z_support)
    #cls_invar = model.reparameterize(mu, logvar)
    #aug_z, aug_y = aug_beta_features(z_support, cls_invar, feature_map, model, n_shot=n_support)
    #print(z_support.shape,z_support_1.shape)
    aug_z = None
    if aug_z is not None:
        z_support_all = torch.cat((z_support,aug_z,aug_z2))
        y_support_all = torch.cat((y_support,aug_y,aug_y2))
    #   z_support_all = torch.cat((z_support_all,z_support_all_1))
    #   y_support_all = torch.cat(( y_support_all,y_support_all_1))
    else:
        z_support_all = z_support
        y_support_all = y_support
    #z_support_all = z_support
    #y_support_all = y_support
    # train classifier with augmened features
    # print(1/(time.time()-time_start))
    linear_clf = backbone.distLinear(feat_dim, n_way).cuda()
    ## initialize weights for linear_clf
    set_optimizer = torch.optim.SGD(linear_clf.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)

    loss_function = torch.nn.CrossEntropyLoss()
    loss_function = loss_function.cuda()
    #    
    batch_size = 4
    support_size = z_support_all.shape[0]
    model.train()
    acc_1 = []
    acc_2 = []
    m = 0
    
    for epoch in range(200):
        rand_id = np.random.permutation(support_size)
        for i in range(0, support_size , batch_size):
           set_optimizer.zero_grad()
           selected_id = torch.from_numpy( rand_id[i: min(i+batch_size, support_size) ]).cuda()
           z_batch = z_support_all[selected_id]
           y_batch = y_support_all[selected_id] 
           scores  = linear_clf(z_batch)


        #    scores_2 = linear_clf(z_query)
        #    pred_2 = torch.argmax(scores_2, 1)
        #    acc_2.append(np.mean(np.array(pred_2.cpu().data) == y_query)*100)
        #    pred_2 = torch.argmax(scores, 1)
        #    acc_2.append(np.mean(np.array(pred_2.cpu().data) == np.array(y_batch.cpu().data))*100)
        #    print(pred_2,y_batch)
           
           
           loss = loss_function(scores,y_batch)
           loss.backward()
           set_optimizer.step()
        # print(acc_2)
    #NWPU   
    # class_name = {1:"Airport",3:"Basketball court",8:"Circular farmland",11:"Dense residential",13:"Forest",16:"Ground track field",19:"Intersection",23:"Medium residential",28:"Parking lot",32:"River"} 
    # class_name = {1:"Airport",3:"Basket",8:"Circularf",11:"Denser",13:"Forest",16:"Groundt",19:"Inters",23:"Mediumr",28:"Parkingl",32:"River"} 
    #AID
    # class_name = {5:"Center",6:"Church",11:"Forest",12:"Industrial",23:"River",24:"School",25:"SparseResidential",26:"Square",28:"StorageTanks",29:"Viaduct"} 
    # class_name = {5:"Center",6:"Church",11:"Forest",12:"Indus",23:"River",24:"School",25:"Sparser",26:"Square",28:"Storaget",29:"Viaduct"} 
    #UCMerced
    # class_name = {3:"Beach",9:"Golfcourse",13:"Mobilehomepark",16:"River",18:"Sparseresidential",20:"Tenniscourt"}
    class_name = {3:"Beach",9:"Golfc",13:"Mobileh",16:"River",18:"Spaser",20:"Tennisc"}  
    model.eval()
    scores = linear_clf(z_query)
    pred = torch.argmax(scores, 1)
    #pdb.set_trace()
    # name = []
    # for i in select_class:
    #     name.append(class_name[i])
    # class_names = np.array(name)
    
    acc = np.mean(np.array(pred.cpu().data) == y_query)*100
    # plot_confusion_matrix(y_query,np.array(pred.cpu().data),classes=class_names, normalize=True)
    return acc
    # return np.array(y_query),np.array(pred.cpu().data),class_names
    

if __name__ == '__main__':
    params = parse_args('finetune_sample')

    image_size = 84
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    # os.environ['RANK'] = '0'
    # os.environ['WORLD_SIZE'] = '1'
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12345'
    split = "novel"
    loadfile = configs.data_dir[params.dataset] + split + '.json'

    checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(params.save_dir, params.dataset, params.model, params.method)
    #params.train_aug = True
    if params.train_aug:
        checkpoint_dir += '_aug'
    checkpoint_dir += "_" + params.split
    checkpoint_dir += '_%.2f'%(params.kl_weight)
    if params.assign_name is not None:
        modelfile   = get_assigned_file(checkpoint_dir,params.assign_name)
#    elif params.method in ['baseline', 'baseline++'] :
#        modelfile   = get_resume_file(checkpoint_dir) #comment in 2019/08/03 updates as the validation of baseline/baseline++ is added
    else:
        modelfile   = get_best_file(checkpoint_dir)
    if params.save_iter != -1:
        novel_file = os.path.join( checkpoint_dir.replace("checkpoints","features"), split + "_" + str(params.save_iter)+ ".hdf5") 
    else:
        novel_file = os.path.join( checkpoint_dir.replace("checkpoints","features"), split + ".hdf5") 
    print(checkpoint_dir)
    novel_datamgr     = SimpleDataManager(image_size, batch_size =16)
    novel_loader      = novel_datamgr.get_data_loader( loadfile, aug = False)
    img_data_file = feature_loader.init_img_loader(novel_loader)
    # datamgr         = SimpleDataManager(image_size, batch_size = 8)
    # data_loader      = datamgr.get_data_loader(loadfile, aug = False, shuffle=False)
    # img_data_file = feature_loader.init_img_loader(data_loader)
    
    #base_file = os.path.join( checkpoint_dir.replace("checkpoints","features"), "base.hdf5") #defaut split = novel, but you can also test base or val classes
    #base_data_file = feature_loader.init_loader(base_file)

    #model           = DisentangleNet( model_dict[params.model], params.num_classes, kl_weight=params.kl_weight, loss_type = params.loss_type)
    model           = DWNet( model_dict[params.model], params.num_classes, kl_weight=params.kl_weight, loss_type = params.loss_type)
    # torch.cuda.set_device(parse_args.local_rank)
    # torch.distributed.init_process_group(backend='nccl')
    model = model.cuda()  # 在使用DistributedDataParallel之前，需要先将模型放到GPU上
    # model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    tmp = torch.load(modelfile)
    state = tmp['state']
    state_keys = list(state.keys())
    for i, key in enumerate(state_keys):
        if "module." in key:
            newkey = key.replace("module.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'  
            state[newkey] = state.pop(key)
    #    else:
    #        state.pop(key)

    model = model.cuda()        
     
    dirname = os.path.dirname(novel_file)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)

    iter_num = 600
    acc_all = []
    model.load_state_dict(state, strict=False)
    #visualize_intra_cls_var(model)
    for i in range(iter_num):
    #     y_query,pred,class_names = finetune_backbone(model, img_data_file, params.train_n_way, params.n_shot, 640)
    #     # print(y_query.shape)
    #     if i ==0:
    #         y = y_query
    #         pre = pred
    #     else:
    #         y = np.hstack((y,y_query))
    #         pre = np.hstack((pre,pred))
    # print(y.shape)
    # plot_confusion_matrix(y,pre,classes=class_names, normalize=False)     
        acc = finetune_backbone(model, img_data_file, params.train_n_way, params.n_shot, 640)
        acc_all.append(acc)
        if i%20 == 0:
            print('Iter: %d, Acc : %f, Avg Acc: %f'% (i, acc, np.mean(np.array(acc_all))))
    
    acc_all = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)

    print('%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num, acc_mean, 1.96* acc_std/np.sqrt(iter_num)))


    # with open('./record/results.txt' , 'a') as f:
    #     timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime()) 
    #     aug_str = '-aug' if params.train_aug else ''
    #     if False :
    #         exp_setting = '%s-%s-%s-%s %sshot %sway_test' %(params.dataset,  params.model, params.method, aug_str, params.n_shot, params.test_n_way )
    #     else:
    #         exp_setting = '%s-%s-%s%s %sshot %sway_train %sway_test aug%s' %(params.dataset, params.model, params.method, aug_str , params.n_shot , params.train_n_way, params.test_n_way, str(params.aug_per_sample) )
    #     acc_str = '%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num, acc_mean, 1.96* acc_std/np.sqrt(iter_num))
    #     f.write( 'Time: %s, Setting: %s, Acc: %s \n' %(timestamp,exp_setting,acc_str)  )
