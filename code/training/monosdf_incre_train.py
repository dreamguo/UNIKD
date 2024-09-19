import os
import sys
import ipdb
import torch
import numpy as np
from tqdm import tqdm
from pyhocon import ConfigFactory

import utils.general as utils
import utils.plots as plt
from utils import rend_util
from torch.utils.tensorboard import SummaryWriter
from model.loss import compute_scale_and_shift
from utils.general import BackprojectDepth

from incre_utils import Process_Pose, save_xyz_beta, save_xyz_beta_3D


class MonoSDFIncreTrainRunner():
    def __init__(self, opt):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        self.conf = ConfigFactory.parse_file(opt.conf)
        self.batch_size = opt.batch_size
        self.nepochs = opt.nepochs
        self.total_nepochs = opt.total_nepochs
        self.exps_folder = opt.exps_folder

        self.plot_only_mesh = opt.plot_only_mesh
        self.step_i = opt.step_i
        self.incremental = opt.incremental
        self.disloss = opt.disloss
        self.ppose = opt.ppose
        self.uncertain_C_thresh = opt.uncertain_C_thresh
        self.gt_depth = opt.gt_depth
        self.training_type = opt.training_type
        self.teacher_freq = opt.teacher_freq

        self.expname = self.conf.get_string('train.expname') + opt.expname
        scan_id = opt.scan_id if opt.scan_id != -1 else self.conf.get_int('dataset.scan_id', default=-1)
        if scan_id != -1:
            self.expname = self.expname + '_{0}'.format(scan_id)
        if self.incremental != 0:
            self.expname = self.expname + '_incre{0}'.format(self.incremental)

        utils.mkdir_ifnotexists(os.path.join('../',self.exps_folder))
        self.timestamp = opt.incre_timestamp
        self.expdir = os.path.join('../', self.exps_folder, self.expname, self.timestamp)
        utils.mkdir_ifnotexists(self.expdir)

        self.plots_dir = os.path.join(self.expdir, 'plots')
        utils.mkdir_ifnotexists(self.plots_dir)

        # create checkpoints dirs
        self.checkpoints_path = os.path.join(self.expdir, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)
        self.model_params_subdir = "ModelParameters"
        self.optimizer_params_subdir = "OptimizerParameters"
        self.scheduler_params_subdir = "SchedulerParameters"

        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.scheduler_params_subdir))

        os.system("""cp -r {0} "{1}" """.format(opt.conf, os.path.join(self.expdir, 'runconf.conf')))

        print('shell command : {0}'.format(' '.join(sys.argv)))

        print('Loading data ...')

        dataset_conf = self.conf.get_config('dataset')
        dataset_conf['step_i'] = self.step_i
        dataset_conf['incremental'] = self.incremental
        dataset_conf['gt_depth'] = self.gt_depth
        dataset_conf['cross'] = opt.cross
        dataset_conf['block'] = opt.block
        dataset_conf['accumulate_data'] = opt.accumulate_data
        self.keyf_num = opt.keyf_num
        if self.keyf_num != 0 and self.step_i != 0:
            keyf_camera_idxs = np.load(os.path.join(self.expdir, 'keyf.npy'))
            assert len(keyf_camera_idxs) <= self.keyf_num
            dataset_conf['keyf_camera_idxs'] = keyf_camera_idxs
        if opt.varify_filter:
            dataset_conf['incremental'] = 0
        if opt.scan_id != -1:
            dataset_conf['scan_id'] = opt.scan_id

        self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(**dataset_conf)
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            collate_fn=self.train_dataset.collate_fn,
                                                            num_workers=8)
        self.plot_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                           batch_size=self.conf.get_int('plot.plot_nimgs'),
                                                           shuffle=True,
                                                           collate_fn=self.train_dataset.collate_fn
                                                           )

        self.total_pixels = self.train_dataset.total_pixels
        self.img_res = self.train_dataset.img_res
        self.n_batches = self.train_dataset.n_batches
        if opt.accumulate_data:
            self.n_batches = self.n_batches // (self.step_i + 1)

        if self.incremental and self.disloss:
            posedir = os.path.join(self.expdir, 'poses')
            utils.mkdir_ifnotexists(posedir)
            scene_scale = self.conf.get_int('model.scene_bounding_sphere')
            self.process_pose = Process_Pose(posedir, self.step_i, self.train_dataset.pose_all, scale=scene_scale*2, ppose=self.ppose)
            if opt.varify_filter:
                from vae import VAE
                from incre_utils import test_VAE
                from vae_utils import visualize_poses

                vae_model_path = os.path.join(posedir, 'VAE_' + str(self.step_i) + '.pth')
                model = VAE(latent_dim=64)
                model.load_state_dict(torch.load(vae_model_path))

                generate_poses_ = test_VAE(model, number=100, scale=scene_scale, device='cpu', trunc=0)
                generate_poses = test_VAE(model, number=100, scale=scene_scale, device='cpu', trunc=50)
                gt_poses = torch.stack(self.train_dataset.pose_all).numpy()
                gt_poses[:, :3, 3] *= 10
                generate_poses[:, :3, 3] *= 10
                generate_poses_[:, :3, 3] *= 10
                x_poses = np.eye(4)[None]
                x_poses[0, 0, 3] = 1
                y_poses = np.eye(4)[None].repeat(2, 0)
                y_poses[0, 1, 3] = 1
                y_poses[1, 1, 3] = 2
                ipdb.set_trace()
                visualize_poses([gt_poses, x_poses, y_poses])
                exit()

        conf_model = self.conf.get_config('model')
        self.model = utils.get_class(self.conf.get_string('train.model_class'))(conf=conf_model)
        if self.disloss:
            self.teacher_model = utils.get_class(self.conf.get_string('train.model_class'))(conf=conf_model)

        self.Grid_MLP = self.model.Grid_MLP
        if torch.cuda.is_available():
            self.model.cuda()
            if self.disloss:
                self.teacher_model.cuda()

        loss_conf = self.conf.get_config('loss')
        loss_conf['gt_depth'] = self.gt_depth
        loss_conf['use_uncertainty_loss'] = opt.use_uncertainty_loss
        loss_conf['uncertainty_weight'] = opt.uncertainty_weight
        self.loss = utils.get_class(self.conf.get_string('train.loss_class'))(**loss_conf)

        self.lr = self.conf.get_float('train.learning_rate')
        self.lr_factor_for_grid = self.conf.get_float('train.lr_factor_for_grid', default=1.0)
        self.uncertainty_min = self.conf.get_float('train.uncertainty_min', default=0.1)

        if self.Grid_MLP:
            self.optimizer = torch.optim.Adam([
                {'name': 'encoding', 'params': list(self.model.implicit_network.grid_parameters()), 
                    'lr': self.lr * self.lr_factor_for_grid},
                {'name': 'net', 'params': list(self.model.implicit_network.mlp_parameters()) +\
                    list(self.model.rendering_network.parameters()),
                    'lr': self.lr},
                {'name': 'density', 'params': list(self.model.density.parameters()),
                    'lr': self.lr},
            ], betas=(0.9, 0.99), eps=1e-15)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Exponential learning rate scheduler
        decay_rate = self.conf.get_float('train.sched_decay_rate', default=0.1)
        decay_steps = self.total_nepochs * self.n_batches
        if self.training_type == 'no_init':
            decay_steps = decay_steps // self.incremental
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, decay_rate ** (1./decay_steps))

        self.do_vis = not opt.cancel_vis

        self.start_epoch = 0

        # load checkpoints if there exists.
        old_checkpnts_dir = os.path.join(self.expdir, 'checkpoints')
        model_path = os.path.join(old_checkpnts_dir, 'ModelParameters', str(opt.checkpoint) + ".pth")
        if os.path.exists(model_path):
            if self.disloss:
                saved_teacher_model_state = torch.load(model_path)
                self.teacher_model.load_state_dict(saved_teacher_model_state["model_state_dict"])
                self.teacher_model.eval()
                print('load teacher model from: ', model_path)

            saved_model_state = torch.load(model_path)
            self.start_epoch = saved_model_state['epoch']
            print('load model from: ', model_path, ' epoch :', self.start_epoch)
            if self.training_type != 'no_init':
                self.model.load_state_dict(saved_model_state["model_state_dict"])

                optimizer_path = os.path.join(old_checkpnts_dir, 'OptimizerParameters', str(opt.checkpoint) + ".pth")
                data = torch.load(optimizer_path)
                self.optimizer.load_state_dict(data["optimizer_state_dict"])

                scheduler_path = os.path.join(old_checkpnts_dir, self.scheduler_params_subdir, str(opt.checkpoint) + ".pth")
                data = torch.load(scheduler_path)
                self.scheduler.load_state_dict(data["scheduler_state_dict"])

        self.num_pixels = self.conf.get_int('train.num_pixels')
        self.plot_freq = self.conf.get_int('train.plot_freq')
        self.plot_3d_freq = self.conf.get_int('train.plot_3d_freq')
        self.checkpoint_freq = self.conf.get_int('train.checkpoint_freq', default=100)
        self.print_freq = self.conf.get_int('train.print_freq', default=50)
        self.split_n_pixels = self.conf.get_int('train.split_n_pixels', default=10000)
        self.plot_conf = self.conf.get_config('plot')
        self.backproject = BackprojectDepth(1, self.img_res[0], self.img_res[1]).cuda()

        # MAS incremental learning method
        self.use_podnet = opt.use_podnet

        # MAS incremental learning method
        self.use_mas = opt.use_mas
        if self.use_mas and self.step_i != 0:
            self.mas_lambda = opt.mas_lambda
            mas_path = os.path.join(self.expdir, 'mas.mas')
            mas_ckpt = torch.load(mas_path)
            self.model_mas = mas_ckpt['model_mas']
            print('load MAS from', mas_path)

        # PackNet baseline
        self.packnet = opt.packnet
        if self.packnet:
            from prune import SparsePruner
            assert self.start_epoch % (self.total_nepochs // self.incremental) == 0
            self.perc = 1 - (1 / (self.incremental - self.step_i))
            if self.step_i != 0:
                previous_masks = torch.load(os.path.join(self.expdir, 'packnet'+str(self.step_i-1)+'.mask'))
            else:
                previous_masks = {}
            self.pruner = SparsePruner(self.model, self.perc, previous_masks)
            # init mask (i+1) for step i
            self.pruner.make_finetuning_mask()

    def save_checkpoints(self, epoch, timestep_save=0):
        print('save checkpoints at epoch ', epoch)
        if timestep_save:
            torch.save(
                {"epoch": epoch, "model_state_dict": self.model.state_dict()},
                os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
            torch.save(
                {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
                os.path.join(self.checkpoints_path, self.optimizer_params_subdir, str(epoch) + ".pth"))
            torch.save(
                {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
                os.path.join(self.checkpoints_path, self.scheduler_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, "latest.pth"))

    def infer(self):
        print("infering...")
        self.model.eval()

        indices, model_input, ground_truth = next(iter(self.plot_dataloader))
        model_input["intrinsics"] = model_input["intrinsics"].cuda()
        model_input["uv"] = model_input["uv"].cuda()
        model_input['pose'] = model_input['pose'].cuda()
        
        split = utils.split_input(model_input, self.total_pixels, n_pixels=self.split_n_pixels)
        if self.plot_only_mesh:
            plot_data = None
        else:
            res = []
            for s in tqdm(split):
                out = self.model(s, indices)
                d = {'rgb_values': out['rgb_values'].detach(),
                     'uncertainty_map': out['uncertain_C_values'].detach(),
                    'normal_map': out['normal_map'].detach(),
                    'depth_values': out['depth_values'].detach()}
                if 'rgb_un_values' in out:
                    d['rgb_un_values'] = out['rgb_un_values'].detach()
                del out
                res.append(d)

            batch_size = ground_truth['rgb'].shape[0]
            model_outputs = utils.merge_output(res, self.total_pixels, batch_size)
            plot_data = self.get_plot_data(model_input, model_outputs, model_input['pose'], ground_truth['rgb'], ground_truth['normal'], ground_truth['depth'])

        test_plots_dir = os.path.join(self.plots_dir + '_test', str(self.step_i))
        utils.mkdir_ifnotexists(test_plots_dir)
        plt.plot(self.model.implicit_network,
                indices,
                plot_data,
                test_plots_dir,
                self.start_epoch,
                1,
                self.img_res,
                **self.plot_conf
                )


    def run(self):
        print("training...")
        # self.iter_step = 0
        # self.writer = SummaryWriter(log_dir=os.path.join(self.plots_dir, 'logs'))

        for epoch in range(self.start_epoch + 1, self.nepochs + 1):
            if self.packnet and (epoch * 2) % (self.total_nepochs // self.incremental) == 0 and epoch % (self.total_nepochs // self.incremental) != 0 and self.perc != 0:
                self.pruner.current_masks = {}
                self.pruner.prune()
            self.model.train()
            self.train_dataset.change_sampling_idx(self.num_pixels)

            for data_index, (indices, model_input, ground_truth) in enumerate(self.train_dataloader):
                if data_index == self.n_batches:
                    break
                old_atts = None
                model_input["intrinsics"] = model_input["intrinsics"].cuda()
                model_input["uv"] = model_input["uv"].cuda()
                if epoch % self.teacher_freq != 0 and self.step_i != 0 and self.disloss:
                    indices = None
                    while 1:
                        model_input['pose'] = self.process_pose.get_pose()
                        teacher_model_outputs = self.teacher_model(model_input, indices)
                        if self.training_type == 'no_filter':
                            break
                        if teacher_model_outputs['uncertain_C_values'].mean() < self.uncertain_C_thresh:
                            break
                    ground_truth['rgb'] = teacher_model_outputs['rgb_values'][None].detach().cpu()
                    ground_truth['depth'] = teacher_model_outputs['depth_values'][None].detach().cpu()
                    # TODO: follow the hard-coded in get_depth_loss()
                    if not self.gt_depth:
                        ground_truth['depth'] = (ground_truth['depth'] - 0.5) / 50
                    ground_truth['normal'] = teacher_model_outputs['normal_map'][None].detach().cpu()
                    ground_truth['mask'] = torch.ones_like(ground_truth['depth'])
                    if self.use_podnet:
                        old_atts = teacher_model_outputs['pod_feature_c'][0]
                    del teacher_model_outputs
                    # dis_loss scheduler: 1/2*cos(pi*(1+r)) + 1/2
                    loss_weight = np.cos(np.pi*(1+epoch/self.total_nepochs)) / 2 + 0.5
                else:
                    model_input['pose'] = model_input['pose'].cuda()
                    loss_weight = 1

                self.optimizer.zero_grad()
                
                model_outputs = self.model(model_input, indices)
                
                loss_output = self.loss(model_outputs, ground_truth)
                loss = loss_output['loss'] * loss_weight
                
                # use podnet baseline
                if self.use_podnet and old_atts != None:
                    import my_podnet
                    loss_pod = my_podnet.loss_pod(model_outputs, old_atts, old_features)
                    loss += loss_pod

                if self.use_mas and self.step_i != 0:
                    model_mas_loss = self.model_mas.penalty(self.model)
                    mas_loss = self.mas_lambda / 2 * (model_mas_loss)
                    loss += mas_loss
                else:
                    mas_loss = torch.tensor(0.0).cuda().float()

                loss.backward()
                # Set fixed param grads to 0.
                if self.packnet:
                    self.pruner.make_grads_zero()
                self.optimizer.step()
                # Set pruned weights to 0.
                if self.packnet:
                    self.pruner.make_pruned_zero()
                self.train_dataset.change_sampling_idx(self.num_pixels)
                self.scheduler.step()

                psnr = rend_util.get_psnr(model_outputs['rgb_values'],
                                          ground_truth['rgb'].cuda().reshape(-1,3))

                if epoch % self.print_freq == 0:
                    print('{0}_{1} [{2}] ({3}/{4}): loss={5}, rgb_loss={6}, uncert_l={7}, psnr={8}'.format(
                        self.expname, self.timestamp, 
                        epoch, data_index, self.n_batches, 
                        loss.item(),
                        loss_output['rgb_loss'].item(),
                        loss_output['uncert_l'].item(),
                        psnr.item()))

                # self.iter_step += 1                     
                # if self.iter_step % self.tensorboard_freq == 0:
                #     self.writer.add_scalar('Loss/loss', loss.item(), self.iter_step)
                #     self.writer.add_scalar('Loss/color_loss', loss_output['rgb_loss'].item(), self.iter_step)
                #     self.writer.add_scalar('Loss/uncert_l', loss_output['uncert_l'].item(), self.iter_step)
                #     self.writer.add_scalar('Loss/eikonal_loss', loss_output['eikonal_loss'].item(), self.iter_step)
                #     self.writer.add_scalar('Loss/smooth_loss', loss_output['smooth_loss'].item(), self.iter_step)
                #     self.writer.add_scalar('Loss/depth_loss', loss_output['depth_loss'].item(), self.iter_step)
                #     self.writer.add_scalar('Loss/normal_l1_loss', loss_output['normal_l1'].item(), self.iter_step)
                #     self.writer.add_scalar('Loss/normal_cos_loss', loss_output['normal_cos'].item(), self.iter_step)
                    
                #     self.writer.add_scalar('Statistics/beta', self.model.density.get_beta().item(), self.iter_step)
                #     self.writer.add_scalar('Statistics/alpha', 1. / self.model.density.get_beta().item(), self.iter_step)
                #     self.writer.add_scalar('Statistics/psnr', psnr.item(), self.iter_step)
                    
                #     if self.Grid_MLP:
                #         self.writer.add_scalar('Statistics/lr0', self.optimizer.param_groups[0]['lr'], self.iter_step)
                #         self.writer.add_scalar('Statistics/lr1', self.optimizer.param_groups[1]['lr'], self.iter_step)
                #         self.writer.add_scalar('Statistics/lr2', self.optimizer.param_groups[2]['lr'], self.iter_step)
                #     else:
                #         self.writer.add_scalar('Statistics/lr', self.optimizer.param_groups[0]['lr'], self.iter_step)

            if epoch % self.checkpoint_freq == 0:
                self.save_checkpoints(epoch, epoch==self.nepochs)

            if self.do_vis and epoch % self.plot_freq == 0 and epoch != 0:
                self.model.eval()

                if self.incremental and self.disloss and self.ppose == 'random':
                    poses_all = []
                    uncertain_C_all = []
                    self.train_dataset.change_sampling_idx(self.num_pixels)
                    for data_index, (indices, model_input, ground_truth) in enumerate(self.train_dataloader):
                        poses_all.append(model_input['pose'])
                        model_input["intrinsics"] = model_input["intrinsics"].cuda()
                        model_input["uv"] = model_input["uv"].cuda()
                        model_input['pose'] = model_input['pose'].cuda()
                        out = self.model(model_input, indices)
                        uncertain_C_all.append(out['uncertain_C_values'].detach().cpu()[None, :, 0])
                    poses_all = torch.cat(poses_all)
                    uncertain_C_all = torch.cat(uncertain_C_all)
                    testsavedir_C = os.path.join(self.expdir, 'uncertainty', 'step{0}_{1}'.format(self.step_i, epoch))
                    os.makedirs(testsavedir_C, exist_ok=True)
                    save_xyz_beta(poses_all*10, testsavedir_C, uncertain_C_all.mean(1))

                self.train_dataset.change_sampling_idx(-1)

                indices, model_input, ground_truth = next(iter(self.plot_dataloader))
                model_input["intrinsics"] = model_input["intrinsics"].cuda()
                model_input["uv"] = model_input["uv"].cuda()
                model_input['pose'] = model_input['pose'].cuda()
                
                split = utils.split_input(model_input, self.total_pixels, n_pixels=self.split_n_pixels)
                if self.plot_only_mesh:
                    plot_data = None
                else:
                    res = []
                    for s in tqdm(split):
                        out = self.model(s, indices)
                        d = {'rgb_values': out['rgb_values'].detach(),
                             'uncertainty_map': out['uncertain_C_values'].detach(),
                            'normal_map': out['normal_map'].detach(),
                            'depth_values': out['depth_values'].detach()}
                        if 'rgb_un_values' in out:
                            d['rgb_un_values'] = out['rgb_un_values'].detach()
                        del out
                        res.append(d)

                    batch_size = ground_truth['rgb'].shape[0]
                    model_outputs = utils.merge_output(res, self.total_pixels, batch_size)
                    plot_data = self.get_plot_data(model_input, model_outputs, model_input['pose'], ground_truth['rgb'], ground_truth['normal'], ground_truth['depth'])

                plt.plot(self.model.implicit_network,
                        indices,
                        plot_data,
                        self.plots_dir,
                        epoch,
                        self.plot_3d_freq,
                        self.img_res,
                        **self.plot_conf
                        )

        if self.use_mas:
            import my_MAS
            import dill
            self.model.eval()
            self.model_mas = my_MAS.MAS(self.model)
            
            self.train_dataset.change_sampling_idx(self.num_pixels)
            for data_index, (indices, model_input, ground_truth) in enumerate(self.train_dataloader):
                model_input["intrinsics"] = model_input["intrinsics"].cuda()
                model_input["uv"] = model_input["uv"].cuda()
                model_input['pose'] = model_input['pose'].cuda()
                model_outputs = self.model(model_input, indices)
                self.model_mas.update_omega(model_outputs['rgb_values'])
                if data_index == 3:
                    break

            # save MAS
            mas_path = os.path.join(self.expdir, 'mas.mas')
            torch.save({'model_mas': self.model_mas}, mas_path, pickle_module=dill)
            print('Saved MAS at', mas_path)

        if self.keyf_num != 0:
            keyf_camera_idxs = self.train_dataset.camera_idxs
            np.random.shuffle(keyf_camera_idxs)
            np.save(os.path.join(self.expdir, 'keyf.npy'), keyf_camera_idxs[:self.keyf_num])

        if self.packnet:
            torch.save(self.pruner.current_masks, os.path.join(self.expdir, 'packnet'+str(self.step_i)+'.mask'))


    def get_plot_data(self, model_input, model_outputs, pose, rgb_gt, normal_gt, depth_gt):
        batch_size, num_samples, _ = rgb_gt.shape

        rgb_eval = model_outputs['rgb_values'].reshape(batch_size, num_samples, 3)
        uncertainty_map = model_outputs['uncertainty_map'].reshape(batch_size, num_samples, 1)
        normal_map = model_outputs['normal_map'].reshape(batch_size, num_samples, 3)
        normal_map = (normal_map + 1.) / 2.

        depth_map = model_outputs['depth_values'].reshape(batch_size, num_samples)
        depth_gt = depth_gt.to(depth_map.device)
        scale, shift = compute_scale_and_shift(depth_map[..., None], depth_gt, depth_gt > 0.)
        depth_map = depth_map * scale + shift
        
        # save point cloud
        depth = depth_map.reshape(1, 1, self.img_res[0], self.img_res[1])
        pred_points = self.get_point_cloud(depth, model_input, model_outputs)

        gt_depth = depth_gt.reshape(1, 1, self.img_res[0], self.img_res[1])
        gt_points = self.get_point_cloud(gt_depth, model_input, model_outputs)
        
        plot_data = {
            'rgb_gt': rgb_gt,
            'normal_gt': (normal_gt + 1.)/ 2.,
            'depth_gt': depth_gt,
            'pose': pose,
            'rgb_eval': rgb_eval,
            'uncertainty_map': uncertainty_map,
            'normal_map': normal_map,
            'depth_map': depth_map,
            "pred_points": pred_points,
            "gt_points": gt_points,
        }

        return plot_data
    
    def get_point_cloud(self, depth, model_input, model_outputs):
        color = model_outputs["rgb_values"].reshape(-1, 3)
        
        K_inv = torch.inverse(model_input["intrinsics"][0])[None]
        points = self.backproject(depth, K_inv)[0, :3, :].permute(1, 0)
        points = torch.cat([points, color], dim=-1)
        return points.detach().cpu().numpy()
