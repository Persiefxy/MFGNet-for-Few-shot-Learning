from concurrent.futures import thread
from ensurepip import version
from multiprocessing import reduction
from turtle import ycor
import SLbackbone
import utils
import pdb
import os
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np
import torch.nn.functional
from utils import l2_norm, _get_log_pz_qz_prodzi_qzCx
import torch.distributed as dist
from SeqAttention import MultiHeadAttention
import torch.nn.functional as F
class Conv_block(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv = nn.Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x

class KDLoss(nn.Module):
    def __init__(self, temp_factor):
        super(KDLoss, self).__init__()
        self.temp_factor = temp_factor
        self.kl_div = nn.KLDivLoss(reduction="sum")

    def forward(self, input, target):
        log_p = torch.log_softmax(input/self.temp_factor, dim=1)
        q = torch.softmax(target/self.temp_factor, dim=1)
        loss = self.kl_div(log_p, q)*(self.temp_factor**2)/input.size(0)
        return loss

class SPLLoss(nn.Module):
    def __init__(self):
        super(SPLLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduction='none')
    def forward(self, input,targeta,targetb,threshold,lam):
        super_loss = mixup_criterion(self.loss,input,targeta,targetb,lam)
        # with open('max.txt','a') as f:
        #     f.write(str(int(targeta[torch.argmax(super_loss)]))+','+str(int(targetb[torch.argmax(super_loss)]))+'\n')
        # with open('min.txt','a') as f:
        #     f.write(str(int(targeta[torch.argmin(super_loss)]))+','+str(int(targetb[torch.argmin(super_loss)]))+'\n')
        # print(super_loss)
        mask = (targeta!=targetb)
        m = 0
        for i in mask:
            if i == True:
                m+=1
        # threshold = torch.min(super_loss)+thre*(torch.max(super_loss)-torch.min(super_loss))
        v,u = self.spl_loss(super_loss,threshold)
        t = 0
        for i in range(len(u)):
            t += math.log(u[i]**u[i])+math.log(v[i]**v[i])-threshold*v[i]
        # print((super_loss * v).mean())
        return (sum(super_loss * v*mask)/m)

    def spl_loss(self, super_loss,threshold):
        u = torch.tensor(super_loss)
        v = torch.tensor(super_loss)
        for i in range(len(v)):
            v[i] = (1+math.exp(-threshold))/(1+math.exp(super_loss[i]-threshold))
        for i in range(len(u)):
            u[i] = 1+math.exp(-threshold)-v[i]
        return v,u

def mixup_criterion(criterion, pred, y_a, y_b, lam):
        #print(criterion(pred, y_a),criterion(pred, y_b))
        # lam = lam.cuda()
        # for i in range(pred.shape[0]):
        #     out[i] = lam[i] * criterion(pred[i], y_a[i]) + (1 - lam[i]) * criterion(pred[i], y_b)
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
class DWNet(nn.Module):
    def __init__(self, model_func,  num_class, kl_weight=1, loss_type = 'Curricular', aug_weight=1.0, use_conv=False, rank=0, world_size=0, avg=True):
        super(DWNet, self).__init__()
        self.backbone = model_func(flatten=False)
        self.aug_weight = aug_weight
        self.DBval = True
        self.rank = rank
        self.world_size = world_size
        self.use_conv = use_conv
        if not use_conv:
            channel = 640
            pool_size = 5
            feature_dim = 640
            self.cls_fc = nn.Sequential(
                nn.AvgPool2d(pool_size),
                SLbackbone.Flatten()      
            )
            self.cls_fc168 = nn.Sequential(
                nn.AvgPool2d(10),
                SLbackbone.Flatten()      
            )
            self.cls_fc252 = nn.Sequential(
                nn.AvgPool2d(15),
                SLbackbone.Flatten()      
            )
            self.cls_fc336 = nn.Sequential(
                nn.AvgPool2d(20),
                SLbackbone.Flatten()      
            )
        else:
            channel = 64
            pool_size = 5
            feature_dim = 1600
            self.cls_fc = nn.Sequential(
                SLbackbone.Flatten()      
            )

        cls_feature_dim = feature_dim
        self.encoder =  nn.Sequential(
            Conv_block(channel, channel, kernel=(3,3), padding=(1,1)),
            Conv_block(channel, channel, kernel=(3,3), padding=(1,1)),
            Conv_block(channel, channel, kernel=(3,3), padding=(1,1)),
            SLbackbone.Flatten(),
        )
        self.encoder2 =  nn.Sequential(
            Conv_block(channel, channel, kernel=(3,3), padding=(1,1)),
            Conv_block(channel, channel, kernel=(3,3), padding=(1,1)),
            Conv_block(channel, channel, kernel=(3,3), padding=(1,1)),
            SLbackbone.Flatten(),
        )
        # self.encoder3 =  nn.Sequential(
        #     Conv_block(channel, channel, kernel=(3,3), padding=(1,1)),
        #     Conv_block(channel, channel, kernel=(3,3), padding=(1,1)),
        #     Conv_block(channel, channel, kernel=(3,3), padding=(1,1)),
        #     SLbackbone.Flatten(),
        # )
        # self.encoder4 =  nn.Sequential(
        #     Conv_block(channel, channel, kernel=(3,3), padding=(1,1)),
        #     Conv_block(channel, channel, kernel=(3,3), padding=(1,1)),
        #     Conv_block(channel, channel, kernel=(3,3), padding=(1,1)),
        #     SLbackbone.Flatten(),
        # )
        self.vae_mean = nn.Linear(channel*pool_size*pool_size, feature_dim)
        self.vae_var = nn.Linear(channel*pool_size*pool_size, feature_dim)
        self.vae_mean2 = nn.Linear(channel*pool_size*pool_size, feature_dim)
        self.vae_var2 = nn.Linear(channel*pool_size*pool_size, feature_dim)
        # self.decoder_fc = nn.Linear(feature_dim, channel*pool_size*pool_size)
        # self.decoder =  nn.Sequential(
        #         Conv_block(channel, channel, kernel=(3,3), padding=(1,1)),
        #         Conv_block(channel, channel, kernel=(3,3), padding=(1,1)),
        #         Conv_block(channel, channel, kernel=(3,3), padding=(1,1)),
        # )
        self.TPLinear = nn.Sequential( nn.Linear(640,160)
                                ) 
        self.rotate_classifier = nn.Sequential( nn.Linear(640,4)
                                )
        self.size_classifier = nn.Sequential( nn.Linear(640,4)
                                ) 
        self.diff_classifier = nn.Sequential(nn.Linear(640,3))
        self.superloss = torch.nn.MSELoss()
        if loss_type == 'softmax':
            self.classifier = nn.Linear(cls_feature_dim, num_class)
            self.classifier.bias.data.fill_(0)
        elif loss_type == 'Curricular': #Baseline ++
            self.classifier = SLbackbone.MXCurricularLinear(cls_feature_dim, num_class)
        elif loss_type == 'dist':
            self.classifier = SLbackbone.distLinear(cls_feature_dim, num_class)
        elif loss_type == 'Arcface':
            self.classifier = SLbackbone.ArcMarginProduct(cls_feature_dim, num_class)
        else:
            self.classifier = SLbackbone.NormLinear(cls_feature_dim, num_class, radius=10)
        self.smloss1 = torch.nn.CrossEntropyLoss(reduction='none')
        self.smloss = torch.nn.CrossEntropyLoss()
        self.kl_weight = kl_weight
        self.ranking_loss = nn.MarginRankingLoss(margin=5.0)
        self.atten =  MultiHeadAttention(1, 640, 640, 640, dropout=0.5)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)  
        eps = torch.randn_like(std)
        # remove abnormal points
        return mu + eps*std,eps
    
    # def forward_d(self, aggr_feature):
    #     if not self.use_conv:
    #         channel = 640
    #         pool_size = 5
    #     else:
    #         channel = 64
    #         pool_size = 5
    #     recon_feature = self.decoder_fc(aggr_feature).view(-1, channel, pool_size, pool_size)
    #     recon_feature = self.decoder(recon_feature)
    #     return recon_feature

    def forward(self, x,target=None):
        if target is None:
            #print(1)
            feature_map = self.backbone(x)
            # x_5 = self.fc(x_5)
            cls_feature = self.cls_fc(feature_map)
            # cls_feature = self.atten(cls_feature,cls_feature,cls_feature)
            #if self.use_wide:
            #    feature_map = self.downsample(feature_map)
            bs = x.size(0)
            if not self.use_conv:
                channel = 640
            else:
                channel = 64
            encoder_map = feature_map + cls_feature.view(bs, channel, 1, 1).detach()
            encoder_feature = self.encoder(encoder_map)
            encoder_feature2 = self.encoder2(encoder_map)
            # encoder_feature3 = self.encoder3(encoder_map)
            # encoder_feature4 = self.encoder4(encoder_map)
            mu = self.vae_mean(encoder_feature)
            logvar = self.vae_var(encoder_feature)
            mu2 = self.vae_mean2(encoder_feature2)
            logvar2 = self.vae_var2(encoder_feature2)
            # mu3 = self.vae_mean2(encoder_feature3)
            # logvar3 = self.vae_var2(encoder_feature3)
            # mu4 = self.vae_mean2(encoder_feature4)
            # logvar4 = self.vae_var2(encoder_feature4)
            return cls_feature, mu, logvar, mu2,logvar2,feature_map
            # return cls_feature, mu, logvar,feature_map
        else:
            feature_map,target_a ,target_b,lam = self.backbone(x,target)
            cls_feature = self.cls_fc(feature_map)
            # cls_feature = self.atten(cls_feature,cls_feature,cls_feature)
            # x_5 = self.fc(x_5)
            #if self.use_wide:
            #    feature_map = self.downsample(feature_map)
            bs = x.size(0)
            if not self.use_conv:
                channel = 640
            else:
                channel = 64
            # print(feature_map.size())
            encoder_map = feature_map + cls_feature.view(bs, channel, 1, 1).detach()
            encoder_feature = self.encoder(encoder_map)
            encoder_feature2 = self.encoder2(encoder_map)
            # encoder_feature3 = self.encoder3(encoder_map)
            # encoder_feature4 = self.encoder4(encoder_map)
            mu = self.vae_mean(encoder_feature)
            logvar = self.vae_var(encoder_feature)
            mu2 = self.vae_mean2(encoder_feature2)
            logvar2 = self.vae_var2(encoder_feature2)
            # mu3 = self.vae_mean2(encoder_feature3)
            # logvar3 = self.vae_var2(encoder_feature3)
            # mu4 = self.vae_mean2(encoder_feature4)
            # logvar4 = self.vae_var2(encoder_feature4)
            return cls_feature,mu, logvar, mu2,logvar2,feature_map,target_a,target_b,lam
            # return cls_feature,mu, logvar, feature_map,target_a,target_b,lam

    def train_all(self, epoch, train_loader, optimizer, tb_logger,thre,n_data=None):
        print_freq = 800
        cls_avg_loss = 0   
        kl_avg_loss = 0
        aug_avg_loss = 0
        diff_avg_loss = 0
        sr_avg_loss = 0
        sd_avg_loss = 0
        beta = self.kl_weight 
        acc_ =[]
        acc_1 =[]
        for i, (x, y) in enumerate(train_loader):
            x = Variable(x.cuda())
            y = Variable(y.cuda()) 
            bs = x.size(0)
            cls_feature, mu, logvar,mu2,logvar2,feature_map,target_a,target_b,lam= self.forward(x,y)
            # cls_feature, mu, logvar,feature_map,target_a,target_b,lam= self.forward(x,y)
            # cls_feature,mu, logvar,mu2,logvar2,feature_map = self.forward(x)
            # tscores = self.classifier.forward(x_5)
            # fscores = self.classifier.forward(cls_feature,target_a,target_b,lam)
            fscores = self.classifier.forward(cls_feature)
            
            # classification loss
            # spLLoss = SPLLoss()
            # tcls_loss = mixup_criterion(self.smloss,tscores,target_a,target_b,lam)
            # fcls_loss = self.smloss(fscores, y)
            # print(fcls_loss)
            # fcls_loss = spLLoss(fscores, target_a,target_b,thre,lam)
            # fcls_loss = mixup_criterion(spLLoss,fscores,target_a,target_b,lam,thre)
            fcls_loss = mixup_criterion(self.smloss,fscores,target_a,target_b,lam)
            # fcls_loss = fcls_loss.mean()
            cls_invar_featurea,expa = self.reparameterize(mu, logvar)
            cls_invar_featureb,expb = self.reparameterize(mu2, logvar2)
            # cls_invar_featurec,expc = self.reparameterize(mu3, logvar3)
            # cls_invar_featured,expd = self.reparameterize(mu4, logvar4)
            # diff_loss =((torch.dist(expa,expb,p=2)))/((torch.dist(cls_invar_featurea,cls_invar_featureb,p=2)))
            diff_loss = (1-(torch.cosine_similarity(cls_invar_featurea,cls_invar_featureb,dim=1)))/(1-(torch.cosine_similarity(expa,expb,dim=1)))
            # diff_loss2 = (1-(torch.cosine_similarity(cls_invar_featurec,cls_invar_featured,dim=1)))/(1-(torch.cosine_similarity(expc,expd,dim=1)))
            # diff_loss3 = (1-(torch.cosine_similarity(cls_invar_featurea,cls_invar_featurec,dim=1)))/(1-(torch.cosine_similarity(expa,expc,dim=1)))
            diff_loss = diff_loss.mean()
            # diff_loss2 = diff_loss2.mean()
            # diff_loss3 = diff_loss3.mean()
            # print(diff_loss)
            # print(torch.dist(cls_invar_featurea,cls_invar_featureb,p=2))
            aggr_featurea = cls_feature + cls_invar_featurea.detach()
            aggr_featureb = cls_feature + cls_invar_featureb.detach()
            aggr_scorea = self.classifier.forward(aggr_featurea)
            aggr_scoreb = self.classifier.forward(aggr_featureb)
            # aggr_featurec = cls_feature + cls_invar_featurec.detach()
            # aggr_featured = cls_feature + cls_invar_featured.detach()
            # aggr_scorec = self.classifier.forward(aggr_featurec)
            # aggr_scored = self.classifier.forward(aggr_featured)
            kdloss = KDLoss(4)
            fcls_loss = kdloss(fscores, aggr_scorea.detach())+kdloss(fscores, aggr_scoreb.detach())+fcls_loss
            # fcls_loss = kdloss(fscores, aggr_scorea.detach())+fcls_loss
            cls_loss = fcls_loss
            
            # differential loss
            # origin = []
            # agrea = []
            # agreb = []
            # for m in range(cls_feature.size()[0]):
            #     origin += [torch.tensor(0)]
            #     agrea += [torch.tensor(1)]
            #     agreb += [torch.tensor(2)]
            # origin = Variable(torch.stack(origin,0)).cuda()
            # agrea = Variable(torch.stack(agrea,0)).cuda()
            # agreb = Variable(torch.stack(agreb,0)).cuda()
            # inputs__ = torch.cat((cls_feature,aggr_featurea,aggr_featureb),dim=0)
            # targets__ = Variable(torch.cat((origin,agrea,agreb),dim=0)).cuda()
            # diff_outputs = self.diff_classifier(inputs__)
            # sdiff_loss = self.smloss(diff_outputs,targets__)
            
            
            #rotation loss
            bs = x.size(0)
            inputs_ = []
            a_ = []
            targets_ = []
            indices = np.arange(bs)
            np.random.shuffle(indices)
            split_size = int(bs/4)
            for j in indices[0:split_size]:
                x90 = x[j].transpose(2,1).flip(1)
                x180 = x90.transpose(2,1).flip(1)
                x270 =  x180.transpose(2,1).flip(1)
                inputs_ += [x[j], x90, x180, x270]
                targets_ += [y[j] for _ in range(4)]
                a_ += [torch.tensor(0),torch.tensor(1),torch.tensor(2),torch.tensor(3)]
            inputs = Variable(torch.stack(inputs_,0)).cuda()
            targets = Variable(torch.stack(targets_,0))
            a_ = Variable(torch.stack(a_,0)).cuda()
            # out,  _, _,_,targeta,targetb,lamr= self.forward(inputs,targets)
            out,  _, _,_,_,_= self.forward(inputs)
            rotate_outputs = self.rotate_classifier(out)
            rloss =  self.smloss(rotate_outputs,a_)
            rscores = self.classifier.forward(out)
            rcls_loss = self.smloss(rscores,targets)
            rloss = rloss+rcls_loss
            
            # trotate_outputs = self.rotate_classifier(tout)
            # rtloss =  self.smloss(trotate_outputs,a_)
            # rtscores = self.classifier.forward(tout)
            # rtcls_loss = self.smloss(rtscores,targets)
            # rloss += rtloss
            # rcls_loss += rtcls_loss
            
            #resize loss
            # split_size = int(bs/4)
            # if split_size!=0:
            #     x168 = F.interpolate(x,size=(168,168), mode='bilinear', align_corners=True)
            #     out168 = self.backbone(x168[0:split_size])
            #     out168 = self.cls_fc168(out168)
            #     # output168 = self.classifier.forward(out168)
            #     size_output168 = self.size_classifier(out168)
                
            #     x252 = F.interpolate(x,size=(252,252), mode='bilinear', align_corners=True)
            #     out252 = self.backbone(x252[0:split_size])
            #     out252 = self.cls_fc252(out252)
            #     # output252 = self.classifier.forward(out252)
            #     size_output252 = self.size_classifier(out252)
                
            #     # x336 = F.interpolate(x,size=(336,336), mode='bilinear', align_corners=True)
            #     # out336 = self.backbone(x336[0:split_size])
            #     # out336 = self.cls_fc336(out336)
            #     # # output336 = self.classifier.forward(out336)
            #     # size_output336 = self.size_classifier(out336)
                
            #     out84 = self.backbone(x[0:split_size])
            #     out84 = self.cls_fc(out84)
            #     # output84 = self.classifier.forward(out84)
            #     size_output84 = self.size_classifier(out84)
            #     sscores_84 = self.classifier.forward(out84)
            #     scls_loss84 = self.smloss(sscores_84,y[0:split_size])
            #     sscores_168 = self.classifier.forward(out168)
            #     scls_loss168 = self.smloss(sscores_168,y[0:split_size])
            #     sscores_252 = self.classifier.forward(out252)
            #     scls_loss252 = self.smloss(sscores_252,y[0:split_size])
            #     # sscores_336 = self.classifier.forward(out336)
            #     # scls_loss336 = self.smloss(sscores_336,y[0:split_size])
            #     scls_loss = (scls_loss168+scls_loss252+scls_loss84)/3
            #     sloss =  self.smloss(size_output168,torch.ones(split_size).cuda().long())+self.smloss(size_output84,torch.zeros(split_size).cuda().long())+self.smloss(size_output252,2*torch.ones(split_size).cuda().long())+scls_loss
            # else:
            #     size_output84 = self.size_classifier(out84)
            #     sscores_84 = self.classifier.forward(out84)
            #     scls_loss84 = self.smloss(sscores_84,y[0:split_size])
            #     sscores_168 = self.classifier.forward(out168)
            #     scls_loss168 = self.smloss(sscores_168,y[0:split_size])
            #     sscores_252 = self.classifier.forward(out252)
            #     scls_loss252 = self.smloss(sscores_252,y[0:split_size])
            #     scls_loss = (scls_loss168+scls_loss252+scls_loss84)/3
            #     sloss = scls_loss
            
            # feature aug loss
            if self.aug_weight > 0:
                aug_cls_featurea = cls_feature + cls_invar_featurea
                aug_cls_featureb = cls_feature + cls_invar_featureb
                # aug_cls_featurec = cls_feature + cls_invar_featurec
                # aug_cls_featured = cls_feature + cls_invar_featured
                aug_scoresa  = self.classifier.forward(aug_cls_featurea)
                aug_scoresb  = self.classifier.forward(aug_cls_featureb)
                # aug_scoresc  = self.classifier.forward(aug_cls_featurec)
                # aug_scoresd  = self.classifier.forward(aug_cls_featured)
                # aug_cls_loss = self.smloss(aug_scoresa, y)+self.smloss(aug_scoresb, y)
                # aug_cls_loss = spLLoss(aug_scores, target_a,target_b,thre,lam)
                # aug_cls_loss = mixup_criterion(spLLoss,aug_scores,target_a,target_b,lam,thre)
                # aug_cls_loss = mixup_criterion(self.smloss,aug_scoresa,target_a,target_b,lam)
                aug_cls_loss = (mixup_criterion(self.smloss, aug_scoresa, target_a, target_b, lam)+mixup_criterion(self.smloss, aug_scoresb, target_a, target_b, lam))/2
                # aug_cls_loss = aug_cls_loss.mean()

            else:
                aug_cls_loss = torch.zeros(1).cuda()

            # kl_loss
            #kl_loss = -0.5*torch.sum(1+logvar-logvar.exp()-mu.pow(2)) / bs
            log_pz, log_qz, log_prod_qzi, log_q_zCx = _get_log_pz_qz_prodzi_qzCx(cls_invar_featurea, (mu, logvar), n_data, is_mss=False)
            log_pz2, log_qz2, log_prod_qzi2, log_q_zCx2 = _get_log_pz_qz_prodzi_qzCx(cls_invar_featureb, (mu2, logvar2), n_data, is_mss=False)
            # log_pz3, log_qz3, log_prod_qzi3, log_q_zCx3 = _get_log_pz_qz_prodzi_qzCx(cls_invar_featurec, (mu3, logvar3), n_data, is_mss=False)
            # log_pz4, log_qz4, log_prod_qzi4, log_q_zCx4 = _get_log_pz_qz_prodzi_qzCx(cls_invar_featured, (mu4, logvar4), n_data, is_mss=False)
            #I[z;x] = KL[q(z,x)||q(x)q(z)] = E_x[KL[q(z|x)||q(z)]]
            mi_loss = (log_q_zCx - log_qz).mean()+(log_q_zCx2 - log_qz2).mean()
            # mi_loss = (log_q_zCx - log_qz).mean()
            # TC[z] = KL[q(z)||\prod_i z_i]
            tc_loss = (log_qz - log_prod_qzi).mean()+(log_qz2 - log_prod_qzi2).mean()
            # tc_loss = (log_qz - log_prod_qzi).mean()
            # dw_kl_loss is KL[q(z)||p(z)] instead of usual KL[q(z|x)||p(z))]
            dw_kl_loss = (log_prod_qzi - log_pz).mean()+(log_prod_qzi2 - log_pz2).mean()
            # dw_kl_loss = (log_prod_qzi - log_pz).mean()
            #loss = cls_loss + recon_loss + kl_loss * self.kl_weight + aug_cls_loss * self.aug_weight
            # loss = cls_loss +(mi_loss  + dw_kl_loss + tc_loss * beta)  + aug_cls_loss * self.aug_weight+rloss+1/diff_loss+sloss
            loss = cls_loss +(mi_loss  + dw_kl_loss + tc_loss * beta)  + aug_cls_loss * self.aug_weight+(1/diff_loss)+rloss
            #loss = cls_loss + recon_loss + kl_loss * self.kl_weight
            # print(beta)
            kl_loss = (mi_loss + dw_kl_loss + tc_loss)/2
            # pred = torch.argmax(fscores, dim=1)
            # acc = torch.mean((pred == y).float())
            # acc_ += [acc]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cls_avg_loss = cls_avg_loss+cls_loss.item()
            kl_avg_loss = kl_avg_loss+kl_loss.item()  
            aug_avg_loss = aug_avg_loss+aug_cls_loss.item()
            sr_avg_loss = sr_avg_loss+rloss.item()
            # sr_avg_loss = sr_avg_loss+sloss.item()
            diff_avg_loss = diff_avg_loss+diff_loss.item()
            sd_avg_loss = sd_avg_loss+diff_loss.item()
            if i%print_freq==0 and self.rank == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | Cl Loss {:f} |  Kl Loss {:f} | Aug Loss {:f}| Diff Loss {:f}| SR loss{:f}| SelfDL {:f}'.format(epoch, i, len(train_loader), cls_avg_loss/float(i+1), kl_avg_loss/float(i+1), aug_avg_loss/float(i+1),diff_avg_loss/float(i+1),sr_avg_loss/float(i+1),sd_avg_loss/float(i+1)))
                curr_step = epoch*len(train_loader) + i
                tb_logger.add_scalar('Cl Loss', cls_avg_loss/float(i+1), curr_step)
                tb_logger.add_scalar('KL Loss', kl_avg_loss/float(i+1), curr_step)
                tb_logger.add_scalar('Aug Loss', aug_avg_loss/float(i+1), curr_step)
        # print('acc:{}'.format(sum(acc_)/len(acc_)))



    def analysis_loop(self, val_loader, record = None):
        cls_class_file  = {}
        #classifier = self.classifier.weight.data
        for i, (x,y) in enumerate(val_loader):
            x = x.cuda()
            x_var = Variable(x)
            feats = self.backbone.forward(x_var)
            # print(feats.size())
            feats = self.cls_fc(feats)
            cls_feats = feats.data.cpu().numpy()
            labels = y.cpu().numpy()
            for f, l in zip(cls_feats, labels):
                if l not in cls_class_file.keys():
                    cls_class_file[l] = []
                cls_class_file[l].append(f)
        for cl in cls_class_file:
            cls_class_file[cl] = np.array(cls_class_file[cl])
        DB, intra_dist, inter_dist = DBindex(cls_class_file)
        #sum_dist = get_dist(classifier)
        print('DB index (cls) = %4.2f, intra_dist (cls) = %4.2f, inter_dist (cls) = %4.2f' %(DB, intra_dist, inter_dist))
        return 1/DB #DB index: the lower the better


                            

def DBindex(cls_data_file):
    #For the definition Davis Bouldin index (DBindex), see https://en.wikipedia.org/wiki/Davies%E2%80%93Bouldin_index
    #DB index present the intra-class variation of the data
    #As baseline/baseline++ do not train few-shot classifier in training, this is an alternative metric to evaluate the validation set
    #Emperically, this only works for CUB dataset but not for miniImagenet dataset

    class_list = cls_data_file.keys()
    cls_num= len(class_list)
    cls_means = []
    stds = []
    DBs = []
    intra_dist = []
    inter_dist = []
    for cl in class_list:
        cls_means.append( np.mean(cls_data_file[cl], axis = 0) )
        stds.append( np.sqrt(np.mean( np.sum(np.square( cls_data_file[cl] - cls_means[-1]), axis = 1))))

    mu_i = np.tile( np.expand_dims( np.array(cls_means), axis = 0), (len(class_list),1,1) )
    mu_j = np.transpose(mu_i,(1,0,2))
    mdists = np.sqrt(np.sum(np.square(mu_i - mu_j), axis = 2))
    
    for i in range(cls_num):
        DBs.append( np.max([ (stds[i]+ stds[j])/mdists[i,j]  for j in range(cls_num) if j != i ]) )
        intra_dist.append(stds[i])
        inter_dist.append(np.mean([mdists[i,j] for j in range(cls_num) if j != i]))

    return np.mean(DBs), np.mean(intra_dist), np.mean(mdists)
