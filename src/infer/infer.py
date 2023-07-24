import os
import shutil
import sys
sys.path.append('.')
sys.path.append('..')
import json
import numpy as np
import torch
import mano
from smplx import SMPLXLayer
import math
from datetime import datetime
from pytorch3d.structures import Meshes
import glob, time
from psbody.mesh import MeshViewers, Mesh
from psbody.mesh.colors import name_to_rgb
from tools.objectmodel import ObjectModel

from tools.utils import makepath, makelogger, to_cpu, to_np, to_tensor
from tools.utils import aa2rotmat, rotmat2aa, rotmul, rotate
from tools.utils import get_relation_map, get_relation_map_new

from omegaconf import OmegaConf

from pct.two_stage import concatMap, Point2Hand
from transformers_model.motion_model import pretrain_actor_rel
from data.dataloader_union import LoadData, build_dataloader

from tools.utils import aa2rotmat, rotmat2aa, d62rotmat
from tools.model_utils import full2bone, full2bone_aa, parms_6D2full, parms_6D2full_addrh_pretrain, full2bone_nohead, parms_6D2full_rh
from tools.vis_tools import sp_animation, get_ground, points_to_spheres
from tools.optim_union import CoopOptim

cdir = os.path.dirname(sys.argv[0])
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
gpu_name = 'cuda:0'

class Tester:

    def __init__(self, cfg, cfg_contact, cfg_hand, inference=False):

        self.dtype = torch.float32
        self.cfg = cfg
        self.cfg_contact = cfg_contact
        self.cfg_hand = cfg_hand

        self.is_inference = inference
        self.cwd = os.path.dirname(sys.argv[0])
        self.joints_num = 36
        torch.manual_seed(cfg.seed)

        starttime = datetime.now().replace(microsecond=0)
        makepath(cfg.work_dir, isfile=False)

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.empty_cache()

        self.device = torch.device(gpu_name if torch.cuda.is_available() else "cpu")

        gpu_brand = torch.cuda.get_device_name(cfg.cuda_id) if use_cuda else None
        gpu_count = cfg.num_gpus

        self.data_info = {}
        self.load_data(cfg, inference)


        self.predict_offsets = cfg.get('predict_offsets', False)
        self.use_exp = cfg.get('use_exp', 0)

        self.rhand_model = mano.load(model_path='./models/mano',
                                    model_type='mano',
                                    num_pca_comps=45,
                                    use_pca=False,
                                    batch_size=1 if self.is_inference else cfg.datasets.batch_size,
                                    flat_hand_mean=True).to(self.device)

        smplx_model_path = './models/smplx'

        self.body_model = SMPLXLayer(
            model_path=smplx_model_path,
            gender='neutral',
            num_pca_comps=45,
            flat_hand_mean=True,
        ).to(self.device)

        self.female_model = SMPLXLayer(
            model_path=smplx_model_path,
            gender='female',
            num_pca_comps=45,
            flat_hand_mean=True,
        ).to(self.device)

        self.male_model = SMPLXLayer(
            model_path=smplx_model_path,
            gender='male',
            num_pca_comps=45,
            flat_hand_mean=True,
        ).to(self.device)

        self.object_model = ObjectModel().to(self.device)

        # Create the network
        relation_map = get_relation_map_new()
        self.contact_network = concatMap(**cfg_contact.network.coop_model).to(self.device)
        self.hand_network = Point2Hand(**cfg_hand.network.coop_model).to(self.device)
        self.body_network = pretrain_actor_rel(relation_map=relation_map,**cfg.network.coop_model).to(self.device)

        cfg.best_model = os.path.join(cfg.work_dir, 'body_net.pt')
        cfg_contact.best_model = os.path.join(cfg_contact.work_dir, 'hand2contact_pt.pt')
        cfg_hand.best_model = os.path.join(cfg_hand.work_dir, 'contact2hand_pt.pt')

        self.contact_network.load_state_dict(torch.load(cfg_contact.best_model, map_location=self.device), strict=True)
        self.hand_network.load_state_dict(torch.load(cfg_hand.best_model, map_location=self.device), strict=True)
        self.body_network.load_state_dict(torch.load(cfg.best_model, map_location=self.device), strict=True)

        self.rh_ids_sampled = torch.from_numpy(np.load('./consts/valid_rh_idx_99.npy'))
        self.rh_verts_ids = to_tensor(np.load('./consts/MANO_SMPLX_vertex_ids.pkl',allow_pickle=True)['right_hand'], dtype=torch.long)

    def load_data(self,cfg, inference):

        ds_name = 'test'
        self.data_info[ds_name] = {}
        ds_test = LoadData(self.cfg.datasets, split_name=ds_name)
        self.test_dataset = ds_test
        self.data_info[ds_name]['frame_names'] = ds_test.frame_names
        self.data_info[ds_name]['frame_sbjs'] = ds_test.frame_sbjs
        self.data_info[ds_name]['frame_objs'] = ds_test.frame_objs
        # self.data_info['body_vtmp'] = ds_test.sbj_vtemp
        # self.data_info['body_betas'] = ds_test.sbj_betas
        self.data_info['obj_verts'] = ds_test.obj_verts
        self.data_info['obj_info'] = ds_test.obj_info
        self.data_info['sbj_info'] = ds_test.sbj_info
        self.ds_test = build_dataloader(ds_test, split='test', cfg=self.cfg.datasets)


    def prepare_rnet_rh(self, batch, pose, trans):

        d62rot = pose.shape[-1] == 96
        params = parms_6D2full_rh(pose, trans, d62rot=d62rot)
        

        B, _ = batch['transl_obj_RH'].shape
        v_template = batch['sbj_vtemp_rh'].to(self.device)

        cnet_output = {}
        self.rhand_model.v_template = v_template

        pose_aa = rotmat2aa(params['fullpose_rotmat']).view(B, -1)
        params['hand_pose'] = pose_aa[:,3:]
        params['global_orient'] = pose_aa[:,:3]

        # output = self.rhand_model(**params)
        # verts = output.vertices
        # cnet_output['verts_full'] = verts

        cnet_output['params'] = params

        return cnet_output



    def infer_contact(self, x):
        ##############################################
        bs = x['transl'].shape[0]

        input_ = {}

        input_['points'] = x['verts']
        input_['normal'] = x['normal']
        input_['obj_height'] = x['obj_height'].view(bs,1)

        net_output = self.contact_network.infer(**input_)
        results = {}
        results['contact_map'] = net_output['contact_map']

        return results


    def infer_hand(self, x):

        bs = x['transl'].shape[0]
        
        # if self.is_inference:
        #     return self.infer(x)

        #############################################

        input_ = {}

        input_['betas'] = x['betas_rh']
        input_['contact_map'] = x['contact_map_obj'].long()
        input_['points'] = x['verts']
        input_['normal'] = x['normal']
        input_['obj_height'] = x['obj_height'].view(bs,1)
            
        net_output = self.hand_network(**input_)

        pose, trans = net_output['pose'], net_output['trans']

        cnet_output = self.prepare_rnet_rh(x, pose, trans)

        results = {}
        cnet_output.update(net_output)
        results['cnet'] = cnet_output

        return results


    def prepare_rnet_body(self, batch, pose, trans):

        B, _ = batch['transl'].shape

        bparams = parms_6D2full_addrh_pretrain(pose, trans, batch['fullpose_rotmat'])

        bparams['betas'] = batch['betas']

        genders = batch['gender'].squeeze(1)
        males = genders == 0
        females = ~males

        FN = sum(females)
        MN = sum(males)

        cnet_output = {}
        refnet_in = {}

        if FN > 0:

            f_params = {k: v[females] for k, v in bparams.items()}
            f_output = self.female_model(**f_params)
            f_verts = f_output.vertices
            f_joints = f_output.joints

            cnet_output['f_verts_full'] = f_verts
            cnet_output['f_joints_full'] = f_joints
            cnet_output['f_params'] = f_params

        if MN > 0:

            m_params = {k: v[males] for k, v in bparams.items()}
            m_output = self.male_model(**m_params)
            m_verts = m_output.vertices
            m_joints = m_output.joints

            cnet_output['m_verts_full'] = m_verts
            cnet_output['m_joints_full'] = m_joints
            cnet_output['m_params'] = m_params

        return cnet_output


    def prepare_model_output(self,pose,trans,x,bs):

        trans_zero = torch.zeros(bs,1).to(self.device)

        transl = torch.cat([trans_zero,trans,trans_zero],dim=-1)

        pose = d62rotmat(pose).view(bs,-1,9)

        mask_ids = x['mask_ids'][:,:self.joints_num].unsqueeze(-1).bool()

        model_input = torch.cat([x['fullpose_rotmat'][:,:21],x['fullpose_rotmat'][:,25:40]],dim=1).view(bs,-1,9)

        final_pose_rotmat = model_input.where(mask_ids,pose).view(bs,-1,3,3)

        return final_pose_rotmat, transl


    def infer_body(self, x):

        bs = x['transl'].shape[0]

        mask_ids = torch.zeros(self.joints_num + 1)
        mask_ids[-1] = 1
        mask_ids = torch.repeat_interleave(mask_ids.unsqueeze(0),bs,0)
        x['mask_ids'] = mask_ids.to(self.device)
        x['gender'] = x['gender'].unsqueeze(1)
        x['betas'] = x['betas_body'].squeeze(1)

        input_ = {}

        input_['betas'] = x['betas']
        input_['wrist_transl'] = x['transl_obj']
        input_['gender'] = x['gender'].long()
        input_['body_pose'] = torch.cat([x['fullpose_rotmat'][:,:21,:2,:],x['fullpose_rotmat'][:,25:40,:2,:]],dim=1).view(bs,-1,6)
        input_['mask_ids'] = x['mask_ids']

        input_ = {k: v.to(self.device) for k,v in input_.items()}

        net_output = self.body_network(**input_)

        pose, transl = self.prepare_model_output(net_output['pose'], net_output['transl'], x, bs)

        cnet_output = self.prepare_rnet_body(x, pose, transl)

        results = {}

        cnet_output.update(net_output)
        results['cnet'] = cnet_output

        return results

    
    def inference_generate_difheight(self):

        self.hand_network.eval()
        self.contact_network.eval()
        self.body_network.eval()
        device = self.device

        ds_name = 'test'
        data = self.ds_test

        base_movie_path = self.cfg.save_file
        num_samples = self.cfg.n_inf_sample
        detail_result_save_path = os.path.join(base_movie_path,'detail_sts_result.json')
        result_save_path = os.path.join(base_movie_path,'sts_result.json')

        save_meshes = True
        save_file = True
        
        previous_movie_name = ''

        get_sbj_obj = []
        for batch_id, batch in enumerate(data):

            movie_name = 's' + self.data_info[ds_name]['frame_names'][batch['idx'].to(torch.long)].split('/s')[-1].replace('/', '_')
            movie_name = movie_name[:np.where([not i.isdigit() for i in movie_name])[0][-1]]

            sbj_name = movie_name.split('_')[0]

            if previous_movie_name == movie_name :
                continue

            previous_movie_name = movie_name

            obj_name = self.data_info[ds_name]['frame_names'][batch['idx'].to(torch.long)].split('/')[-1].split('_')[0]

            
            if self.cfg.object and  obj_name != self.cfg.object:
                continue
                
            if self.cfg.subject and sbj_name != self.cfg.subject:
                continue

            sbj_obj = sbj_name + '_' + obj_name

            if sbj_obj in get_sbj_obj:
                continue
            else:
                get_sbj_obj.append(sbj_obj)

            batch = {k:v.to(self.device) for k,v in batch.items()}

            obj_path = self.data_info['obj_info'][obj_name]['obj_mesh_file']
            obj_mesh = Mesh(filename=obj_path)
            obj_verts = torch.from_numpy(obj_mesh.v)
            obj_m = ObjectModel(v_template=obj_verts).to(device)
            
            # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

            rh_m = self.rhand_model

            rh_m.v_template = batch['sbj_vtemp_rh'].to(rh_m.v_template.device)

            # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

            gender = batch['gender'].data
            if gender == 1:
                sbj_m = self.female_model
            else:
                sbj_m = self.male_model
            
            sbj_m.v_template = batch['sbj_vtemp_body'].to(sbj_m.v_template.device)

            body_fit_smplx = CoopOptim(sbj_model=sbj_m,
                                        rh_model=rh_m,
                                        obj_model=obj_m,
                                        cfg=self.cfg,
                                        device=self.device,
                                        verbose=True)

             # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

            mov_count = 1 #
            movie_path = os.path.join(base_movie_path, str(mov_count), movie_name+'_final.html')
            grasp_meshes_path = os.path.join(base_movie_path, f'{obj_name}_grasp', sbj_name)

            while os.path.exists(movie_path):
                mov_count += 1
                movie_path = os.path.join(base_movie_path, str(mov_count), movie_name+'_final.html')
            
            if save_file:
                makepath(grasp_meshes_path)
            makepath(movie_path, isfile=True)

            grnd_mesh, cage, axis_l = get_ground()

             # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

            large_pic_sbj_meshes = []
            large_pic_obj_meshes = []
            detail_result = []

            def get_rotate_roz(r,m):
                R = r
                roata = torch.Tensor([[math.cos(R),-math.sin(R),0],
                                [math.sin(R),math.cos(R),0],
                                [0,0,1]]).to(torch.float32).to(self.device)
                return torch.matmul(m,roata)

                
            sp_anim = sp_animation()


            x = self.cfg.x
            z = self.cfg.z
            y = self.cfg.y

            key_ = str(x) + '_'+ str(z) + '_' + str(y)


            batch_ = batch.copy()

            print(f'{movie_name} -- {x},{y},{z} frames')

            batch_['transl_obj'][0][0] = y
            batch_['transl_obj'][0][1] = z
            batch_['transl_obj'][0][2] = x

            r = math.atan(y/x)

            reserved_r = - r 

            if x < 0:
                r = math.pi + r
                reserved_r = 2 * math.pi - r

            new_rotamat = get_rotate_roz(r,aa2rotmat(batch_['global_orient_obj_RH'])).reshape(-1,3,3)
            batch_['global_orient_obj_RH'] = rotmat2aa(new_rotamat)
            obj_params_gt = {'transl': batch_['transl_obj_RH'],'global_orient': new_rotamat,'pose2rot':False}
            obj_output = obj_m(**obj_params_gt)
            obj_verts = obj_output.vertices[0]

            def pc_normalize(pc):
                centroid = torch.mean(pc, dim=0) # bs , N
                pc_center = pc - centroid
                return pc_center, centroid

            pc_center, centroid = pc_normalize(obj_verts)

            all_points_num = obj_verts.shape[0]

            n_verts_sample = 2048

            if n_verts_sample <= all_points_num:
                simple_vertices_ids = np.random.choice(all_points_num, n_verts_sample, replace=False)
            else:
                simple_vertices_ids = np.random.choice(all_points_num, n_verts_sample, replace=True)

            simple_vertices_ids = torch.from_numpy(simple_vertices_ids)
            mesh = Meshes(verts=pc_center[None,:], faces=torch.from_numpy(obj_mesh.f[None,:].astype(np.int64)).to(self.device))
            normal = mesh.verts_normals_packed().view(-1, 3)
            batch_['normal'] = normal[simple_vertices_ids].unsqueeze(0).to(self.device)
            batch_['verts'] = pc_center[simple_vertices_ids].unsqueeze(0).to(self.device)
            batch_['obj_height'] = (torch.Tensor([z]).to(self.device) - centroid[2]) / 2

            obj_params_gt = {'transl': batch_['transl_obj'],
                             'global_orient': batch_['global_orient_obj']}
            obj_output = obj_m(**obj_params_gt)
            obj_verts = obj_output.vertices

            contact_output = self.infer_contact(batch_)
            pre_map = torch.softmax(contact_output['contact_map'],dim=-1)
            pre_map[:,0] -= 0.2
            pre_map = pre_map.argmax(-1).view(-1,n_verts_sample)
            batch_['contact_map_obj'] = pre_map

            R = torch.tensor(
            [[1., 0., 0.],
            [0., 0., -1.],
            [0., 1., 0.]]).reshape(1, 3, 3).transpose(1,2).to(self.device)

            hand_output = self.infer_hand(batch_)
            hand_output['cnet']['params']['transl'] += centroid

            body_net_output = self.infer_body(batch_)
            optim_output = body_fit_smplx.fitting(batch_, hand_output, body_net_output, obj_mesh, reserved_r)

            if save_meshes or save_file:
                sbj_opt = Mesh(v=to_cpu(optim_output['opt_verts'][0]), f=sbj_m.faces, vc=name_to_rgb['green'])
                obj_i = Mesh(to_cpu(obj_verts[0]), f = obj_mesh.f, vc=name_to_rgb['yellow'])

            if save_meshes:
                sp_anim.add_frame([sbj_opt, obj_i, grnd_mesh], ['refined_grasp' , 'object', 'ground_mesh'])
                sp_anim.save_animation(movie_path)

            if save_file:
                sbj_opt.write_ply(grasp_meshes_path + f'/{x}_{z}_{y}_sbj_refine.ply')
                obj_i.write_ply(grasp_meshes_path + f'/{x}_{z}_{y}_obj.ply')
                
            break


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def inference():
    
    import argparse

    parser = argparse.ArgumentParser(description='Coop-Inference')

    parser.add_argument('--rh-work-dir',
                        required=True,
                        type=str,
                        help='The path to the folder to save results')

    parser.add_argument('--body-work-dir',
                        required=True,
                        type=str,
                        help='The path to the folder to save results')

    parser.add_argument('--data-file', default='input data',
                        type=str,
                        help='proceed data file')

    parser.add_argument('--save-file', default='save file',
                        type=str,
                        help='proceed data file')
    
    parser.add_argument('--object', default='apple',
                        type=str,
                        help='unseen object')

    parser.add_argument('--subject', default="",
                        type=str,
                        help='body shape')

    parser.add_argument('--x', default=0.6,
                        type=float,
                        help='object x postition')
    
    parser.add_argument('--z', default=0.6,
                        type=float,
                        help='object z postition')
    
    parser.add_argument('--y', default=0.3,
                        type=float,
                        help='object y postition')

    cmd_args = parser.parse_args()

    save_file = cmd_args.save_file
    cfg_path = os.path.join(cmd_args.body_work_dir,'body_net.yaml')

    body_cfg = OmegaConf.load(cfg_path)
    body_cfg.datasets.dataset_dir = cmd_args.data_file
    body_cfg.datasets.grab_path = cmd_args.data_file
    body_cfg.save_file = save_file
    body_cfg.work_dir = cmd_args.body_work_dir
    body_cfg.num_gpus = 1
    body_cfg.batch_size = 1
    body_cfg.object = cmd_args.object
    body_cfg.subject = cmd_args.subject
    body_cfg.x = cmd_args.x
    body_cfg.z = cmd_args.z
    body_cfg.y = cmd_args.y
    
    hand_cfg_path = os.path.join(cmd_args.rh_work_dir,'contact2hand.yaml')
    hand_cfg = OmegaConf.load(hand_cfg_path)
    hand_cfg.datasets.dataset_dir = cmd_args.data_file
    hand_cfg.work_dir = cmd_args.rh_work_dir

    contact_cfg_path = os.path.join(cmd_args.rh_work_dir,'hand2contact.yaml')
    contact_cfg = OmegaConf.load(contact_cfg_path)
    contact_cfg.datasets.dataset_dir = cmd_args.data_file
    contact_cfg.work_dir = cmd_args.rh_work_dir

    tester = Tester(body_cfg, contact_cfg, hand_cfg, inference=True)
    tester.inference_generate_difheight()

if __name__ == '__main__':
    inference()