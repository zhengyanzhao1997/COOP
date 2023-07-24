# -*- coding: utf-8 -*-
#
# Copyright (C) 2022 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import math
from tools.utils import makepath, to_cpu, to_np, to_tensor, create_video

from tools.utils import aa2rotmat, rotmat2aa, rotmul, rotate 
from tools.model_utils import full2bone, full2bone_aa, parms_6D2full, full2bone_aa_nohead, full2bone_aa_rh
import chamfer_distance as chd
import os
import sys
from pytorch3d.structures import Meshes
from tools import utils_loss


class CoopOptim(nn.Module):

    def __init__(self,
                 sbj_model,
                 rh_model,
                 obj_model,
                 cfg,
                 device,
                 verbose = False
                 ):
        super(CoopOptim, self).__init__()

        self.device = device
        self.dtype = torch.float32
        self.cfg = cfg
        self.body_model_cfg = cfg.body_model

        self.sbj_m = sbj_model
        self.rh_m = rh_model
        self.obj_m = obj_model

        self.config_optimizers()
        self.cwd = os.path.dirname(sys.argv[0])


        self.rh_ids_sampled = torch.from_numpy(np.load('./consts/valid_rh_idx_99.npy'))
        self.feet_verts_ids = to_tensor(np.load('./consts/feet_ids_f.npy'), dtype=torch.long)
        self.rh_verts_ids = to_tensor(np.load('./consts/MANO_SMPLX_vertex_ids.pkl',allow_pickle=True)['right_hand'], dtype=torch.long)
        self.mano_tips_ids =  to_tensor(np.array([744, 320, 443, 554, 671]), dtype=torch.long)
        self.smplx_tips = self.rh_verts_ids[self.mano_tips_ids]


        self.four_contact_front =  to_tensor(np.array([8497, 5829]), dtype=torch.long)
        self.four_concat_hind = to_tensor(np.array([8706, 8929]), dtype=torch.long)
        self.four_points = torch.cat([self.four_contact_front,self.four_concat_hind],dim=-1)


        self.body_part = np.load('./consts/body_part.npy',allow_pickle=True)[()]
        self.r_leg_v = to_tensor(self.body_part['right_hand_v'],dtype=torch.long)
        self.l_leg_v = to_tensor(self.body_part['left_hand_v'],dtype=torch.long)
        self.b_v = to_tensor(self.body_part['body_v'],dtype=torch.long)
        self.l_f_h_v = to_tensor(self.body_part['l_f_h_v'],dtype=torch.long)

        self.verbose = verbose

        self.ch_dist = chd.ChamferDistance()


        self.f1 = [697, 698, 699, 700, 712, 713, 714, 715, 737, 738, 739, 740, 741, 743, 744, 745, 746, 748, 749,
                    750, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768,
                    31, 267, 125, 704, 266, 126, 10, 8, 9, 240]

        self.f2 = [46, 47, 48, 49, 155, 156, 164, 165, 166, 167, 189, 194, 195, 223, 224, 237, 238, 245, 280, 281, 298, 300, 301, 317, 320, 323, 324, 325, 326,
                    327, 328, 329, 330, 331, 332, 333, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354,
                    355, 140, 139, 171, 173, 172, 170, 137, 138, 168, 169, 150, 132, 62, 65, 128, 93, 63, 64, 129]

        self.f3 = [356, 357, 358, 359, 372, 373, 375, 376, 385, 386, 387, 396, 397, 398, 402, 403, 410, 413, 429, 433, 434, 435, 436, 437, 438,
                    439, 440, 441, 442, 443, 444, 452, 453, 454, 455, 456, 459, 460, 461, 462, 463, 464, 465, 466, 467,
                    370, 379, 378, 380, 75, 288, 228]

        self.f4 = [468, 469, 470, 471, 485, 486, 496, 497, 506, 507, 513, 514, 524, 545, 546, 547, 548, 549,
                    550, 551, 552, 553, 555, 563, 564, 565, 566, 567, 570, 572, 573, 574, 575, 576, 577, 578,
                    579, 489, 488, 197, 141, 76]

        self.f5 = [580, 581, 582, 583, 601, 602, 614, 615, 624, 625, 630, 631, 641, 663, 664, 665, 666, 667,
                    668, 670, 672, 680, 681, 682, 683, 684, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695,
                    596, 607, 606, 595, 604, 605]

        self.f0 = [73, 96, 98, 99, 772, 774, 775, 777]

    def config_optimizers(self):
        bs = 1
        self.bs = bs
        device = self.device
        dtype = self.dtype

        self.opt_params = {
            'global_orient'     : torch.randn(bs, 1* 3, device=device, dtype=dtype, requires_grad=True),
            'body_pose'         : torch.randn(bs, 20*3, device=device, dtype=dtype, requires_grad=True),
            'left_hand_pose'    : torch.randn(bs, 15*3, device=device, dtype=dtype, requires_grad=True),
            'right_hand_pose'   : torch.randn(bs, 15*3, device=device, dtype=dtype, requires_grad=False),
            'transl'            : torch.zeros(bs, 1, device=device, dtype=dtype, requires_grad=True),}

        self.opt_params_hands = {
            'global_orient'   : torch.randn(bs, 1* 3, device=device, dtype=dtype, requires_grad=True),
            'hand_pose'       : torch.randn(bs, 15*3, device=device, dtype=dtype, requires_grad=True),
            'transl'         : torch.zeros(bs, 3, device=device, dtype=dtype, requires_grad=True),
        }

        lr = self.cfg.get('smplx_opt_lr', 5e-3)
        
        self.opt = optim.Adam([self.opt_params[k] for k in ['global_orient','body_pose','transl']] + [self.opt_params_hands[k] for k in ['global_orient','hand_pose','transl']] , lr=lr)

        # self.opt_h = optim.Adam([self.opt_params_hands[k] for k in ['global_orient','hand_pose','transl']], lr=lr)
        
        self.optimizers = [self.opt]

        self.num_iters = [400]

        self.LossL1 = nn.L1Loss(reduction='mean')
        self.LossL2 = nn.MSELoss(reduction='mean')


    def reput_transl(self, start_params):
        
        fullpose_aa = rotmat2aa(start_params['fullpose_rotmat']).reshape(1, -1)
        start_params_aa = full2bone_aa_nohead(fullpose_aa, start_params['transl'])

        opt_params = {k:aa2rotmat(v) for k,v in start_params_aa.items() if k!='transl'}
        opt_params['transl'] = start_params['transl']

        output = self.sbj_m(**opt_params, return_full_pose = True)
        verts = output.vertices

        feet_vertices = verts[:, self.feet_verts_ids][:,:,1]
        start_params_aa['transl'][:,1] -= feet_vertices.mean(1)
        return start_params_aa


    def init_params(self, start_params):

        # fullpose_aa = rotmat2aa(start_params['fullpose_rotmat']).reshape(1, -1)

        # start_params_aa = full2bone_aa(fullpose_aa, start_params['transl'])

        start_params_aa = self.reput_transl(start_params)

        for k in self.opt_params.keys():
            if k == 'body_pose':
                self.opt_params[k].data = torch.repeat_interleave(start_params_aa[k][:,:-3], self.bs, dim=0)
            elif k == 'transl':
                self.opt_params[k].data = torch.repeat_interleave(start_params_aa[k][:,1:2], self.bs, dim=0)
            else:
                self.opt_params[k].data = torch.repeat_interleave(start_params_aa[k], self.bs, dim=0)
    
        #self.reput_transl()


    def get_smplx_verts(self, batch, output):

        B = batch['transl'].shape[0]

        if batch['gender']==0:
            net_params = output['cnet']['m_params']
        else:
            net_params = output['cnet']['f_params']

        self.sbj_params = net_params

        self.init_params(net_params)

        with torch.no_grad():
            sbj_output = self.sbj_m(**net_params)
            v = sbj_output.vertices.reshape(-1, 10475, 3)
            self.feet_vertexs = v[:, self.feet_verts_ids]

            point1_xy_ori = v[0][8687][[0,2]]
            point2_xy_ori = v[0][8898][[0,2]]
            self.ori_dis = torch.sqrt(torch.pow(point1_xy_ori-point2_xy_ori, 2).sum(-1))
            self.four_feet_points = v[:,self.four_points]


        self.sbj_params['body_pose'] = self.sbj_params['body_pose'][:,:-1]
        return v


    def init_rh_params(self, start_params):

        fullpose_aa = rotmat2aa(start_params['fullpose_rotmat']).reshape(1, -1)
        start_params_aa = full2bone_aa_rh(fullpose_aa, start_params['transl'])
        for k in self.opt_params_hands.keys():
            self.opt_params_hands[k].data = torch.repeat_interleave(start_params_aa[k], self.bs, dim=0)


    def get_rh_verts(self, batch, output):

        B = batch['transl_obj_RH'].shape[0]
        net_params = output['cnet']['params']
        obj_params_gt = {'transl': batch['transl_obj_RH'],
                         'global_orient': batch['global_orient_obj_RH']}

        obj_output = self.obj_m(**obj_params_gt)

        self.obj_verts = obj_output.vertices
        self.sbj_params_hands = net_params

        self.init_rh_params(net_params)

        with torch.no_grad():
            sbj_output = self.rh_m(**net_params)
            v = sbj_output.vertices.reshape(-1, 778, 3)
            verts_sampled = v
            self.rh_vertices = v
            self.rh_wrist_transl = sbj_output.joints[:,0]

        return v


    def cdist(self,x,y):
        if len(x) > 0:
            return torch.cdist(x, y, p=2).min(-1).values.sum()
        else:
            return 0
            
    # def cdist(self,x,y):
    #     if len(x) > 0:
    #         d_lists = torch.cdist(x, y, p=2).min(-1).values
    #         if d_lists.min() < 0.001:
    #             return d_lists.mean()
    #         else:
    #             return d_lists.min()
    #     else:
    #         return 0

    def calc_rh_loss(self, batch, net_output, stage, obj_mesh, itr):

        opt_params = self.opt_params_hands

        output = self.rh_m(**opt_params, return_full_pose = True)
        verts = output.vertices

        self.rh_wrist_transl = output.joints[:,0]
        self.rh_vertices = verts

        losses = {}

        concat_loss = 0
        obj_v = batch['verts'].view(-1,3)
        concat_map = batch['contact_map_obj'].view(-1)
        concat_loss += self.cdist(obj_v[concat_map == 1], verts[:,self.f2])
        concat_loss += self.cdist(obj_v[concat_map == 2], verts[:,self.f3])
        concat_loss += self.cdist(obj_v[concat_map == 3], verts[:,self.f5])
        concat_loss += self.cdist(obj_v[concat_map == 4], verts[:,self.f4])
        concat_loss += self.cdist(obj_v[concat_map == 5], verts[:,self.f1])
        concat_loss = concat_loss * 0.1
        if concat_loss != 0:
            losses['concat_loss'] = concat_loss

        mesh = Meshes(verts=verts, faces=torch.from_numpy(self.rh_m.faces[None,:].astype(np.int64)).to(self.device))
        hand_normal = mesh.verts_normals_packed().view(1, -1, 3)
        nn_dist, nn_idx = utils_loss.get_NN(self.obj_verts, verts)
        outer = nn_dist > 0.00001
        nn_dist = nn_dist * outer
        interior = utils_loss.get_interior(hand_normal, verts, self.obj_verts, nn_idx).type(torch.bool)
        if len(interior) > 0:
            penetr_dist = nn_dist[interior].sum()
        else:
            penetr_dist = 0

        o_mesh = Meshes(verts = self.obj_verts, faces = torch.from_numpy(obj_mesh.f[None,:].astype(np.int64)).cuda())
        o_normal = o_mesh.verts_normals_packed().view(1, -1, 3)
        o_nn_dist, o_nn_idx = utils_loss.get_NN(verts, self.obj_verts)
        o_outer = o_nn_dist > 0.00001
        o_nn_dist = o_nn_dist * o_outer
        o_interior = utils_loss.get_interior(o_normal, self.obj_verts, verts, o_nn_idx).type(torch.bool)
        if len(interior) > 0:
            o_penetr_dist = o_nn_dist[o_interior].sum()
        else:
            o_penetr_dist = 0

        penet = 10 * (penetr_dist  + o_penetr_dist)
        if penet != 0:
            losses['penet'] = penet

        losses['global_orient'] = 0.01 * self.LossL2(self.sbj_params_hands['global_orient'].detach().reshape(-1), self.opt_params_hands['global_orient'].reshape(-1))
        losses['hand_pose'] = 0.001 * self.LossL2(self.sbj_params_hands['hand_pose'].detach().reshape(-1), self.opt_params_hands['hand_pose'].reshape(-1))
        losses['transl'] = 0.01 * self.LossL1(self.opt_params_hands['transl'],self.sbj_params_hands['transl'].detach())
        loss_total = torch.sum(torch.stack([torch.mean(v) for v in losses.values()]))

        losses['loss_total'] = loss_total

        return losses, verts, output


    def trans_global2loc(self, global_t,local_pose):
            bs = local_pose.shape[0]
            global_t = global_t.reshape(bs,1,3,3)
            global_pose = local_pose.view(bs, -1, 3, 3)
            x = global_pose[:,0]
            for i in [3,6,9,14,17,19]:
                x = torch.matmul(x, global_pose[:,i])
            output = torch.matmul(torch.inverse(x.unsqueeze(1)),global_t)
            return output


    def get_rotate_roz(self,r,m,rotation=False):
        R = r # /180*math.pi
        roata = torch.Tensor([[math.cos(R),-math.sin(R),0],
                                [math.sin(R),math.cos(R),0],
                                [0,0,1]]).to(torch.float32).to(self.device)
        if rotation:
            return rotmul(roata.reshape(1, 3, 3).transpose(1,2), m)
        else:
            return torch.matmul(m,roata)


    def glob2rel(self, wrist_transl,global_orient,rh_vertices,obj_transl, r = None):

        global_orient_rotmat = aa2rotmat(global_orient)

        if r:
            wrist_transl = self.get_rotate_roz(r,wrist_transl)
            global_orient_rotmat = self.get_rotate_roz(r,global_orient_rotmat,rotation=True)
            rh_vertices = self.get_rotate_roz(r,rh_vertices)

        R = torch.tensor(
            [[1., 0., 0.],
            [0., 0., -1.],
            [0., 1., 0.]]).reshape(1, 3, 3).transpose(1,2).to(self.device)

        wrist_transl = to_tensor(rotate(wrist_transl, R)) + obj_transl
        rh_vertices = to_tensor(rotate(rh_vertices, R)) + obj_transl
        ### right fix !
        global_orient_rotmat = to_tensor(rotmul(R, global_orient_rotmat))

        return wrist_transl, global_orient_rotmat, rh_vertices


    def calc_body_loss(self, batch, net_output, stage, itr, r):

        global_orient_rotmat = aa2rotmat(self.opt_params['global_orient'])
        body_pose_rotmat = aa2rotmat(self.opt_params['body_pose'])
        pose_cat = torch.cat([global_orient_rotmat,body_pose_rotmat],dim=1)


        rh_wrist_transl, wrist_orient_rotamat, rh_vertices = self.glob2rel(self.rh_wrist_transl.detach(),self.sbj_params_hands['global_orient'].clone().detach(),self.rh_vertices.detach(),batch['transl_obj'], r)
        #rh_wrist_transl, wrist_orient_rotamat, rh_vertices = self.glob2rel(self.rh_wrist_transl.detach(),self.sbj_params_hands['global_orient'].detach(),self.rh_vertices.detach(),batch['transl_obj'])

        rel_wrist_orient_rotmat = self.trans_global2loc(wrist_orient_rotamat,pose_cat)

        self.opt_params['right_hand_pose'] = self.opt_params_hands['hand_pose'].clone().detach()
        opt_params = {k:aa2rotmat(v) for k,v in self.opt_params.items() if k not in ['transl','body_pose','global_orient']}
        opt_params['body_pose'] = torch.cat([body_pose_rotmat,rel_wrist_orient_rotmat],dim=1)
        opt_params['global_orient'] = global_orient_rotmat
        opt_params['transl'] = torch.cat([self.sbj_params['transl'][:,0:1],self.opt_params['transl'],self.sbj_params['transl'][:,2:]],dim=1)

        output = self.sbj_m(**opt_params, return_full_pose = True)
        verts = output.vertices
        joints = output.joints

        losses = {}

        # ---------gaze opt--------------
        eyes_position = (joints[:, 24] +  joints[:, 23]) / 2

        gt_gaze = eyes_position - batch['transl_obj']

        pre_gaze = verts[:, 2007] - eyes_position

        gt_gaze = gt_gaze/gt_gaze.norm(dim=-1, keepdim=True)

        pre_gaze = pre_gaze/pre_gaze.norm(dim=-1, keepdim=True)

        losses['gaze_loss'] = self.LossL1(gt_gaze.detach(), pre_gaze)
        # ------------------------------


        # ---------leg penent opt--------------
        losses['left_hands_pen'] = self.get_hands_body_pen(verts[:,self.l_leg_v],verts[:],self.l_f_h_v)

        if batch['transl_obj'][0,1] <= 0.5:
            
            losses['right_hands_pen'] = self.get_hands_body_pen(verts[:,self.r_leg_v],verts[:],self.b_v)
        # ------------------------------


        # losses['cnet_wrist_transl'] = 20 * self.LossL1(rh_wrist_transl, joints[:, 21])
        losses['cnet_rh_vertices'] = 20 * self.LossL1(rh_vertices, verts[:, self.rh_verts_ids])

        body_loss = {k: self.LossL2(rotmat2aa(self.sbj_params[k]).detach().reshape(-1), self.opt_params[k].reshape(-1)) for k in
                     ['body_pose']}

        # feet control
        
        body_loss['ff'] = self.LossL2(rotmat2aa(self.sbj_params['body_pose'][:,9:11]).detach().reshape(-1), self.opt_params['body_pose'][:,27:33].reshape(-1))
        body_loss['foot_concat'] = verts[:, self.feet_verts_ids, 1].abs().sum()
        body_loss["grnd_contact"] = 5 * (verts[:,:,1].min() < - 0.01) * (- verts[:,:,1].min())

        point1_xy = verts[0][8687][[0,2]]
        point2_xy = verts[0][8898][[0,2]]
        now_dis = torch.sqrt(torch.pow(point1_xy-point2_xy, 2).sum(-1))
        body_loss['slid'] = 0.5 * torch.abs(now_dis - self.ori_dis)

        # ------------balance------------------
        if itr > 200:
            body_loss['balance'] = 5 * self.g_loss(joints[0][0][[0,2]],verts[0])
        # -----------------------------------------

        losses.update(body_loss)

        loss_total = torch.sum(torch.stack([torch.mean(v) for v in losses.values()]))
        losses['loss_total'] = loss_total

        return losses, verts, output, rh_vertices , rh_wrist_transl


    def g_loss(self,joints_xy,verts):

        fornt_v = verts[self.four_contact_front][:,[0,2]]

        concat_hind = torch.where(verts[self.four_concat_hind][:,1] < 0.01)[0]

        if len(concat_hind) == 0:
            point1_xy = fornt_v[0]
            point2_xy = fornt_v[1]

        elif len(concat_hind) == 2:
            point1_xy = verts[[self.four_contact_front[0].item(),self.four_concat_hind[0].item()]][:,[0,2]].mean(0)
            point2_xy = verts[[self.four_contact_front[1].item(),self.four_concat_hind[1].item()]][:,[0,2]].mean(0)

        else:
            if concat_hind[0] == 0:
                point2_xy = verts[[self.four_contact_front[0].item(),self.four_contact_front[1].item(),self.four_concat_hind[0].item()]][:,[0,2]].mean(0)
                point1_xy = verts[[self.four_contact_front[0].item(),self.four_concat_hind[0].item()]][:,[0,2]].mean(0)
            else:
                point1_xy = verts[[self.four_contact_front[0].item(),self.four_contact_front[1].item(),self.four_concat_hind[1].item()]][:,[0,2]].mean(0)
                point2_xy = verts[[self.four_contact_front[1].item(),self.four_concat_hind[1].item()]][:,[0,2]].mean(0)


        return self.get_line_dis(joints_xy,point1_xy,point2_xy)


    def get_line_dis(self, joints_xy, point1_xy, point2_xy):

        x0,y0 = joints_xy[0],joints_xy[1]

        x1,y1 = point1_xy[0],point1_xy[1]

        x2,y2 = point2_xy[0],point2_xy[1]

        k = (y2-y1)/(x2-x1)

        A = k

        B = -1

        C = y1-k*x1

        feet_x = (B*B*x0-A*B*y0-A*C) / (A*A+B*B)

        feet_y = (A*A*y0-A*B*x0-B*C) / (A*A+B*B)

        feet = torch.Tensor([feet_x,feet_y]).to(self.device)

        feet_dis1 = torch.pow(feet-point1_xy, 2).sum(-1)

        feet_dis2 = torch.pow(feet-point2_xy, 2).sum(-1)

        line_dis = torch.pow(point1_xy-point2_xy, 2).sum(-1)

        if feet_dis1 > line_dis:
            return torch.sqrt(torch.pow(joints_xy-point2_xy, 2).sum(-1))

        elif feet_dis2 > line_dis:
            return torch.sqrt(torch.pow(joints_xy-point1_xy, 2).sum(-1))
        
        else:
            return torch.abs(A*x0+B*y0+C)/torch.sqrt(A*A+B*B)


        # dis_1 = torch.sqrt(torch.pow(joints_xy-point1_xy, 2).sum(-1))
        # dis_2 = torch.sqrt(torch.pow(joints_xy-point2_xy, 2).sum(-1))
        # if dis < dis_1 and dis < dis_2:
        #     print(1)
        # else:
        #     print(0)
        # return min(dis,dis_1,dis_2)


    def fitting(self, batch, net_output_hands, net_output_body, obj_mesh, r):

        cnet_verts = self.get_smplx_verts(batch, net_output_body)

        self.get_rh_verts(batch, net_output_hands)


        _, _, orh_vertices = self.glob2rel(self.rh_wrist_transl.detach(),self.sbj_params_hands['global_orient'].clone().detach(),self.rh_vertices.detach(),batch['transl_obj'], r)
        
        for stg, optimizer in enumerate(self.optimizers):
            for itr in range(self.num_iters[stg]):
                optimizer.zero_grad()
                losses_rh, _, _ = self.calc_rh_loss(batch, net_output_hands, stg, obj_mesh, itr)
                losses_body, opt_verts, _, rh_vertices, wrist_transl = self.calc_body_loss(batch, net_output_body, stg, itr , r)
                if itr == 0:
                    cnet_verts = opt_verts
                    orh_vertices = rh_vertices
                loss_total = losses_rh['loss_total'] + losses_body['loss_total']
                loss_total.backward(retain_graph=True)
                optimizer.step()
                # if self.verbose and itr % 50 == 0:
                #     print(self.create_loss_message(losses_rh, stg, itr))
                #     print(self.create_loss_message(losses_body, stg, itr))

        #opt_results = {k:aa2rotmat(v.detach()) for k,v in self.opt_params.items() if v != 'transl'}
        #opt_results['transl'] = self.opt_params['transl'].detach()
        #opt_results['fullpose_rotmat'] = opt_output.full_pose.detach()
        opt_results = {}
        opt_results['cnet_verts'] = cnet_verts
        opt_results['opt_verts'] = opt_verts
        opt_results['rh_verts'] = rh_vertices
        opt_results['orh_vertices'] = orh_vertices
        opt_results['wrist_transl'] = wrist_transl
        return opt_results


    @staticmethod
    def create_loss_message(loss_dict, stage=0, itr=0):
        ext_msg = ' | '.join(['%s = %.2e' % (k, v) for k, v in loss_dict.items() if k != 'loss_total'])
        return f'Stage:{stage:02d} - Iter:{itr:04d} - Total Loss: {loss_dict["loss_total"]:02e} | [{ext_msg}]'


    def get_hands_body_pen(self,hands_v,body_v,ex_v):
        mesh = Meshes(verts=body_v, faces=torch.from_numpy(self.sbj_m.faces[None,:].astype(np.int64)).to(self.device))
        hand_normal = mesh.verts_normals_packed().view(1, -1, 3)
        nn_dist, nn_idx = utils_loss.get_NN(hands_v, body_v[:,ex_v])
        interior = utils_loss.get_interior(hand_normal[:,ex_v], body_v[:,ex_v], hands_v, nn_idx).type(torch.bool)
        if len(interior) > 0:
            penetr_dist = nn_dist[interior].sum()
        else:
            penetr_dist = 0
        return penetr_dist