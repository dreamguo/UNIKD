# adapted from https://github.com/cvg/nice-slam
import os
import numpy as np
import argparse
import pickle
import os
import glob
import open3d as o3d
import matplotlib.pyplot as plt
import torch
import trimesh
import ipdb


def load_poses(path):
    poses = []
    with open(path, "r") as f:
        lines = f.readlines()
    idx = 0
    while idx < len(lines):
        c2w = np.eye(4)
        for i in range(3):
            c2w[i] = np.array(list(map(float, lines[idx].split()))).reshape(1, 4)
            idx += 1
        idx += 1
        c2w[0, 3] *= -1
        c2w[2, :3] *= -1
        c2w[1, :3] *= -1
        c2w = torch.from_numpy(c2w).float()
        poses.append(c2w)
    return poses


parser = argparse.ArgumentParser(
    description='Arguments to cull the mesh.'
)

parser.add_argument('--input_mesh', type=str,  help='path to the mesh to be culled')
parser.add_argument('--input_scalemat', type=str,  help='path to the scale mat')
parser.add_argument('--traj', type=str,  help='path to the trajectory')
parser.add_argument('--output_mesh', type=str,  help='path to the output mesh')
args = parser.parse_args()

H = 480
W = 480
fx = 481.2
fy = 480.0
cx = 239.5
cy = 239.5

poses = load_poses(args.traj)
n_imgs = len(poses)
mesh = trimesh.load(args.input_mesh, process=False)

# transform to original coordinate system with scale mat
scalemat = np.load(args.input_scalemat)['scale_mat_0']
# TODO: see plot 102
scalemat[0, 3] *= -1
scalemat[2, 3] *= -1
mesh.vertices = mesh.vertices @ scalemat[:3, :3].T + scalemat[:3, 3]

# delete additional mesh vertices
# [-2.4670404316178463, 0.06982126847378822, -1.9376205850489896]
# [2.530948203671869, 2.6656269135991693, 3.249339662808609]
transform = [[1,0,0, 0],
                [0,1,0, 2.7/2],
                [0,0,1, 0.6],
                [0,0,0, 1]]
box = trimesh.creation.box(extents=[5.2, 2.7, 5.1], transform=transform)
mesh = mesh.slice_plane(box.facets_origin, -box.facets_normal)
print(mesh.vertices.min(0).tolist())
print(mesh.vertices.max(0).tolist())

# pc = mesh.vertices
# faces = mesh.faces

# # delete mesh vertices that are not inside any camera's viewing frustum
# whole_mask = np.ones(pc.shape[0]).astype(bool)
# for i in range(0, n_imgs, 1):
#     c2w = poses[i]

#     # c2w[2, :] *= -1
#     # c2w[:, 1] *= -1
#     points = pc.copy()
#     points = torch.from_numpy(points).cuda()
#     w2c = np.linalg.inv(c2w)
#     w2c = torch.from_numpy(w2c).cuda().float()
#     K = torch.from_numpy(
#         np.array([[fx, .0, cx], [.0, fy, cy], [.0, .0, 1.0]]).reshape(3, 3)).cuda()
#     ones = torch.ones_like(points[:, 0]).reshape(-1, 1).cuda()
#     homo_points = torch.cat(
#         [points, ones], dim=1).reshape(-1, 4, 1).cuda().float()
#     cam_cord_homo = w2c@homo_points
#     cam_cord = cam_cord_homo[:, :3]

#     cam_cord[:, 0] *= -1
#     uv = K.float()@cam_cord.float()
#     z = uv[:, -1:]+1e-5
#     uv = uv[:, :2]/z
#     uv = uv.float().squeeze(-1).cpu().numpy()
#     edge = 0
#     mask = (0 <= -z[:, 0, 0].cpu().numpy()) & (uv[:, 0] < W -
#                                                edge) & (uv[:, 0] > edge) & (uv[:, 1] < H-edge) & (uv[:, 1] > edge)
#     whole_mask &= ~mask
# pc = mesh.vertices
# faces = mesh.faces
# face_mask = whole_mask[mesh.faces].all(axis=1)
# mesh.update_faces(~face_mask)
print(mesh.vertices.min(0).tolist())
print(mesh.vertices.max(0).tolist())

mesh.export(args.output_mesh)
