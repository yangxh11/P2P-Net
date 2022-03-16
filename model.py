import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from anchors import generate_default_anchor_maps, hard_nms
from clustering import PartsResort


class PMG(nn.Module):
    def __init__(self, model, feature_size, num_ftrs, classes_num, topn):
        super(PMG, self).__init__()

        self.backbone = model
        self.num_ftrs = num_ftrs
        self.topn = topn
        self.im_sz = 448
        self.pad_side = 224
        self.PR = PartsResort(self.topn, self.num_ftrs//2)

        self.proposal_net = ProposalNet(self.num_ftrs)
        _, edge_anchors, _ = generate_default_anchor_maps()
        self.edge_anchors = (edge_anchors+self.pad_side).astype(np.int)
        
        # mlp for regularization
        self.reg_mlp1 = nn.Sequential(
            nn.Linear(self.num_ftrs//2 * self.topn, self.num_ftrs//2),
            nn.ELU(inplace=True),
            nn.Linear(self.num_ftrs//2, self.num_ftrs//2)
        )
        self.reg_mlp2 = nn.Sequential(
            nn.Linear(self.num_ftrs//2 * self.topn, self.num_ftrs//2),
            nn.ELU(inplace=True),
            nn.Linear(self.num_ftrs//2, self.num_ftrs//2)
        )
        self.reg_mlp3 = nn.Sequential(
            nn.Linear(self.num_ftrs//2 * self.topn, self.num_ftrs//2),
            nn.ELU(inplace=True),
            nn.Linear(self.num_ftrs//2, self.num_ftrs//2)
        )

        # stage 1
        self.conv_block1 = nn.Sequential(
            BasicConv(self.num_ftrs//4, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs//2, kernel_size=3, stride=1, padding=1, relu=True),
            nn.AdaptiveMaxPool2d(1)
        )
        self.classifier1 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs//2),
            nn.Linear(self.num_ftrs//2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(feature_size, classes_num),
        )

        # stage 2
        self.conv_block2 = nn.Sequential(
            BasicConv(self.num_ftrs//2, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs//2, kernel_size=3, stride=1, padding=1, relu=True),
            nn.AdaptiveMaxPool2d(1)
        )
        self.classifier2 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs//2),
            nn.Linear(self.num_ftrs//2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(feature_size, classes_num),
        )

        # stage 3
        self.conv_block3 = nn.Sequential(
            BasicConv(self.num_ftrs, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs//2, kernel_size=3, stride=1, padding=1, relu=True),
            nn.AdaptiveMaxPool2d(1)
        )
        self.classifier3 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs//2),
            nn.Linear(self.num_ftrs//2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(feature_size, classes_num),
        )
        
        # concat features from different stages
        self.classifier_concat = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs//2 * 3),
            nn.Linear(self.num_ftrs//2 * 3, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

    def forward(self, x, is_train=True):
        _, _, f1, f2, f3 = self.backbone(x)

        batch = x.shape[0]
        rpn_score = self.proposal_net(f3.detach())
        all_cdds = [np.concatenate((x.reshape(-1, 1), 
                    self.edge_anchors.copy(),
                    np.arange(0, len(x)).reshape(-1, 1)), 
                    axis=1) for x in rpn_score.data.cpu().numpy()]
        top_n_cdds = np.array([hard_nms(x, self.topn, iou_thresh=0.25) for x in all_cdds])
        top_n_index = top_n_cdds[:, :, -1].astype(np.int)
        top_n_index = torch.from_numpy(top_n_index).long().to(x.device)
        top_n_prob = torch.gather(rpn_score, dim=1, index=top_n_index)
        
        # re-input salient parts
        part_imgs = torch.zeros([batch, self.topn, 3, 224, 224]).to(x.device)
        x_pad = F.pad(x, (self.pad_side, self.pad_side, self.pad_side, self.pad_side), mode='constant', value=0)
        for i in range(batch):
            for j in range(self.topn):
                [y0, x0, y1, x1] = top_n_cdds[i, j, 1:5].astype(np.int)
                part_imgs[i:i + 1, j] = F.interpolate(x_pad[i:i + 1, :, y0:y1, x0:x1], 
                                                        size=(224, 224), mode='bilinear',
                                                        align_corners=True)
        
        part_imgs = part_imgs.view(batch*self.topn, 3, 224, 224)
        _, _, f1_part, f2_part, f3_part = self.backbone(part_imgs.detach())
        f1_part = self.conv_block1(f1_part).view(batch*self.topn, -1)
        f2_part = self.conv_block2(f2_part).view(batch*self.topn, -1)
        f3_part = self.conv_block3(f3_part).view(batch*self.topn, -1)
        yp1 = self.classifier1(f1_part)
        yp2 = self.classifier2(f2_part)
        yp3 = self.classifier3(f3_part)
        yp4 = self.classifier_concat(torch.cat((f1_part, f2_part, f3_part), -1))


        # resort parts
        feature_points = f3_part.view(batch, self.topn, -1)
        parts_order = self.PR.classify(feature_points.data.cpu().numpy(), is_train)
        parts_order = torch.from_numpy(parts_order).long().to(x.device)
        parts_order = parts_order.unsqueeze(2).expand(batch, self.topn, self.num_ftrs//2)

        f1_points = torch.gather(f1_part.view(batch, self.topn, -1), dim=1, index=parts_order)
        f1_m = self.reg_mlp1(f1_points.view(batch, -1))
        f2_points = torch.gather(f2_part.view(batch, self.topn, -1), dim=1, index=parts_order)
        f2_m = self.reg_mlp2(f2_points.view(batch, -1))
        f3_points = torch.gather(f3_part.view(batch, self.topn, -1), dim=1, index=parts_order)
        f3_m = self.reg_mlp3(f3_points.view(batch, -1))

        # stage-wise classification
        f1 = self.conv_block1(f1).view(batch, -1)
        f2 = self.conv_block2(f2).view(batch, -1)
        f3 = self.conv_block3(f3).view(batch, -1)
        y1 = self.classifier1(f1)
        y2 = self.classifier2(f2)
        y3 = self.classifier3(f3)
        y4 = self.classifier_concat(torch.cat((f1, f2, f3), -1))

        return y1, y2, y3, y4, yp1, yp2, yp3, yp4, top_n_prob, f1_m, f1, f2_m, f2, f3_m, f3
    
    
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ProposalNet(nn.Module):
    def __init__(self, depth):
        super(ProposalNet, self).__init__()
        self.down1 = nn.Conv2d(depth, 128, 3, 1, 1)
        self.down2 = nn.Conv2d(128, 128, 3, 2, 1)
        self.down3 = nn.Conv2d(128, 128, 3, 2, 1)
        self.ReLU = nn.ReLU()
        self.tidy1 = nn.Conv2d(128, 6, 1, 1, 0)
        self.tidy2 = nn.Conv2d(128, 6, 1, 1, 0)
        self.tidy3 = nn.Conv2d(128, 9, 1, 1, 0)
        # proposals: 14x14x6, 7x7x6, 4x4x9

    def forward(self, x):
        batch_size = x.size(0)
        d1 = self.ReLU(self.down1(x))
        d2 = self.ReLU(self.down2(d1))
        d3 = self.ReLU(self.down3(d2))
        t1 = self.tidy1(d1).view(batch_size, -1)
        t2 = self.tidy2(d2).view(batch_size, -1)
        t3 = self.tidy3(d3).view(batch_size, -1)
        return torch.cat((t1, t2, t3), dim=1)
