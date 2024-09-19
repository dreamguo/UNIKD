import os
import ipdb
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation, Slerp


def center2pose(center):
    '''
    INPUT
        center: camera center in world coordinate. (z > 0)
    RETURN
        pose: camera-to-world matrix [R | t].
    PROCESS
        t_wc = center
        R_z = norm(center)
        if R_z == [0, 0, 1/-1]:
            R_x = [1, 0, 0]  # aviod nan value.
        else:
            R_x = cross(Z_w, R_z)
        R_x = norm(R_x)
        R_y = cross(R_z, R_x)
        pose = [ R_x | R_y | R_z | t_wc ]
    '''
    t = center
    R_z = center / (center**2).sum().sqrt()
    if R_z[0] == 0 and R_z[1] == 0:
        R_x = torch.Tensor([[1], [0], [0]])
    else:
        R_x = torch.cross(torch.Tensor([[0],[0],[1]]), R_z)
        R_x = R_x / (R_x**2).sum().sqrt()
    R_y = torch.cross(R_z, R_x)
    pose = torch.cat((R_x, R_y, R_z, t), dim=1)
    return pose


def save_xyz_beta_3D(xyz, gt_xyz, testsavedir, betas):

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    gt_x = gt_xyz[:, 0]
    gt_y = gt_xyz[:, 1]
    gt_z = gt_xyz[:, 2]

    # plt.figure(figsize=(8,6))
    fig = plt.figure() 
    ax = fig.add_subplot(111,projection='3d')

    betas *= 1e7
    min_v = min(betas)
    max_v = max(betas)
    color = [plt.get_cmap("seismic", 100)(int(float(i-min_v)/(max_v-min_v)*100)) for i in betas]

    plt.set_cmap(plt.get_cmap("seismic", 100))
    im = ax.scatter(x, y, z, s=100, c=color,marker='.')
    ax.scatter(gt_x, gt_y, gt_z, s=10, c='g')
    fig.colorbar(im, format=matplotlib.ticker.FuncFormatter(lambda x,pos:int(x*(max_v-min_v)+min_v)))

    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    plt.savefig(os.path.join('./', 'test.png'))
    plt.show()


def save_xyz_beta(render_poses, testsavedir, betas):
    xs = render_poses[:, 0, 3].cpu().numpy()
    ys = render_poses[:, 1, 3].cpu().numpy()
    zs = render_poses[:, 2, 3].cpu().numpy()
    xyz_beta = np.stack((xs, ys, betas), axis=1)
    np.save(os.path.join(testsavedir, 'xyz_beta.npy'), xyz_beta)

    betas = np.maximum(betas * 1000, 0)
    plt.scatter(xs, betas, color='yellow', label='x')
    plt.scatter(ys, betas, color='red', label='y')
    plt.scatter(zs, betas, color='black', label='z')
    plt.legend(loc="upper left")
    plt.savefig(os.path.join(testsavedir, 'xyz_beta.png'))
    plt.cla()

    ax = plt.subplot(111, projection='3d')
    bottom = 0
    width = depth = 0.1
    top = betas
    ax.bar3d(xs, ys, bottom, width, depth, top, shade=True)
    ax.set_zlabel('beta')
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.savefig(os.path.join(testsavedir, 'xybeta.png'))
    plt.cla()

    plt.close()


def m2r(matrixs):
    tmp = Rotation.from_matrix(matrixs)
    angles = tmp.as_euler("xzy")
    return angles

def r2m(angles):
    tmp = Rotation.from_euler("xzy", angles)
    matrixs = tmp.as_matrix()
    return matrixs

def get_pose_range_eular(poses):
    t_min = poses[:, :, 3].min(axis=0)
    t_max = poses[:, :, 3].max(axis=0)
    angles = m2r(poses[:,:,:3])
    r_min = angles.min(axis=0)
    r_max = angles.max(axis=0)
    return np.stack((t_min, t_max, r_min, r_max))

def m2q(matrixs):
    tmp = Rotation.from_matrix(matrixs)
    quants = tmp.as_quat()
    return quants

def q2m(quants):
    tmp = Rotation.from_quat(quants)
    matrixs = tmp.as_matrix()
    return matrixs

def get_pose_range_quant(poses):
    t_min = poses[:, :, 3].min(axis=0)
    t_max = poses[:, :, 3].max(axis=0)
    start_q = m2q(poses[0,:,:3])
    end_q = m2q(poses[-1,:,:3])
    return np.concatenate((t_min, t_max, start_q, end_q))

def get_7d_pose(poses, scale):
    quant = (m2q(poses[:, :3, :3]) / np.pi + 1) / 2
    trans = (poses[:, :3, 3] + scale/2) / scale
    poses_7d = np.concatenate((quant, trans), axis=1)
    return poses_7d

def get_3x4_pose_from_7d(poses_7d, scale):
    rotation = q2m((poses_7d[:, :4] * 2 - 1) * np.pi)
    trans = poses_7d[:, 4:] * scale - scale/2
    poses_34 = np.concatenate((rotation, trans[:, :, np.newaxis]), axis=2)
    return poses_34

class Process_Pose():
    def __init__(self, posedir, step_i, poses, scale, ppose,
                    p_range_type='quant', dataset_type='room') -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ppose = ppose
        self.step_i = step_i
        poses = torch.stack(poses).numpy()
        self.pose_npy_path = os.path.join(posedir, 'pose.npy')

        if self.ppose == 'random':
            self.p_range_type = p_range_type
            self.dataset_type = dataset_type
            # if dataset_type != 'room':
            #     self.rad = np.sqrt((poses[0, :3, 3]**2).sum())
            # self.range_dir = posedir
            # self.save_pose_range(poses[:, :3, :])
            save_idxs = np.arange(len(poses))
            poses = poses[save_idxs]
            poses = np.insert(poses, 4, values=np.array(save_idxs)[None].T, axis=2)
            if step_i == 0:
                os.system('rm ' + self.pose_npy_path)
                np.save(self.pose_npy_path, poses)
            else:
                self.prev_poses = np.load(self.pose_npy_path)
                base_idx = self.prev_poses[-1, 0, -1]
                poses[:, :, -1] += base_idx + 1
                pose_npy = np.concatenate((self.prev_poses, poses))
                np.save(self.pose_npy_path, pose_npy)
        elif self.ppose == 'save':
            if step_i == 0:
                os.system('rm ' + self.pose_npy_path)
                np.save(self.pose_npy_path, poses)
            else:
                self.prev_poses = np.load(self.pose_npy_path)
                pose_npy = np.concatenate((self.prev_poses, poses))
                np.save(self.pose_npy_path, pose_npy)

    def save_pose_range(self, poses):
        os.makedirs(self.range_dir, exist_ok=True)
        if self.p_range_type == "quant":
            pose_range = get_pose_range_quant(poses)
        elif self.p_range_type == "eular":
            pose_range = get_pose_range_eular(poses)
        np.save(os.path.join(self.range_dir, '{}.npy'.format(self.step_i)), pose_range)

    def get_pose(self):
        if self.ppose == 'random':
            if self.dataset_type == "room":
                pose = self.random_gen_pose_new()
                # pose = self.random_gen_pose()
            else:
                # center = [x (-1, 1), y (-1, 1), z(0, 1)]
                center = torch.rand(3, 1)
                center[0] = center[0] * 2 - 1
                center[1] = center[1] * 2 - 1
                center = center / (center**2).sum().sqrt() * self.rad
                pose = center2pose(center)
        elif self.ppose == 'save' and self.step_i != 0:
            img_i = np.random.choice(self.prev_poses.shape[0])
            pose = self.prev_poses[img_i, :3, :4]

        return torch.Tensor(pose[None]).to(self.device)

    def random_gen_pose(self):
        prev_step = np.random.choice(np.arange(self.step_i))
        if self.p_range_type == "quant":
            pose_range = np.load(os.path.join(
                        self.range_dir, '{}.npy'.format(prev_step)))
            t_min, t_max, start_q, end_q = pose_range[:3], pose_range[3:6], pose_range[6:10], pose_range[10:]
            new_t = np.random.uniform(low=t_min, high=t_max)
            rotations = Rotation.concatenate([Rotation.from_quat(start_q), Rotation.from_quat(end_q)])
            slerp = Slerp([0,1], rotations)
            new_r = slerp(np.random.rand(1)).as_matrix()[0]
            pose = np.concatenate((new_r, new_t[:, np.newaxis]), axis=1)
        elif self.p_range_type == "eular":
            t_min, t_max, r_min, r_max = np.load(os.path.join(
                        self.range_dir, '{}.npy'.format(prev_step)))
            new_t = np.random.uniform(low=t_min, high=t_max)
            new_r = np.random.uniform(low=r_min, high=r_max)
            pose = np.concatenate((r2m(new_r), new_t[:, np.newaxis]), axis=1)
        return pose

    def random_gen_pose_new(self):
        if self.p_range_type == "quant":
            # normalize interval
            interval = self.prev_poses[:, 0, 4] / self.prev_poses[-1, 0, 4]
            rand_seed = np.random.rand(1)
            t_idx = (interval<rand_seed).sum()
            w_ = (rand_seed - interval[t_idx-1]) / (interval[t_idx] - interval[t_idx-1])

            # compute new ratation matrix
            quants = m2q(self.prev_poses[:, :3, :3])
            rotations = Rotation.from_quat(quants)
            slerp = Slerp(interval, rotations)
            new_r = slerp(rand_seed).as_matrix()[0]

            # compute new translation
            old_t = self.prev_poses[:, :3, 3]
            new_t = w_ * old_t[t_idx-1] + (1-w_) * old_t[t_idx]

            pose = np.concatenate((new_r, new_t[:, np.newaxis]), axis=1)
        elif self.p_range_type == "eular":
            pose = None
        return pose
