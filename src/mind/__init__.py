from __future__ import annotations
import open3d as o3d
import torch
import pymeshlab
import numpy as np
import networkx as nx
from skimage import measure
import trimesh
from scipy.spatial import KDTree
import polyscope as ps
import cc3d
import collections
from scipy.sparse import coo_matrix
from torch.optim import Optimizer
from typing import Optional

from ._core import __doc__, __version__, erode, dilate, grid_cut,grid_expansion, label_graph,label_graph_merge, inplace_label, m3c_py, is_bound
import math


__all__ = ["__doc__", "__version__", "erode", "dilate", "grid_cut", "grid_expansion", "label_graph", "label_graph_merge", "inplace_label", "m3c_py", "is_bound", "MIND"]





class AdamWithDirectionProjection(Optimizer):
    """
    一个支持将更新量投影到给定方向张量的 Adam 优化器。
    
    该实现可以处理投影方向 P_i = (0, 0, 0) 的情况，此时应用完整的 Adam 更新量。
    """

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, projection_direction: Optional[torch.Tensor] = None):
        
        # ... (参数校验与上一个实现相同，省略以节省空间) ...
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        
        super().__init__(params, defaults)
        
        self.projection_direction = None
        
        param_list = self.param_groups[0]['params']
        p_to_project = param_list[0]
        
        if projection_direction is not None:
            if p_to_project.shape != projection_direction.shape:
                raise ValueError(f"Parameter and projection direction must have the same shape. Got {p_to_project.shape} and {projection_direction.shape}")
            
            # 存储投影方向，并确保其在正确的设备上且不可求导
            self.projection_direction = projection_direction.detach().to(p_to_project.device)


    def step(self, closure=None):
        """执行单个优化步骤。"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # 获取优化方向（如果有）
            p_to_project = group['params'][0]
            P = self.projection_direction
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdamWithDirectionProjection does not support sparse gradients, please consider SparseAdam instead')
                
                # --- 标准 Adam 计算逻辑 (省略状态初始化和动量更新，与上一个实现相同) ---
                state = self.state[p]
                
                # State initialization (如果 state 为空，在这里进行初始化)
                if not state:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['amsgrad']:
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if group['amsgrad']:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Perform weight decay
                if group['weight_decay'] != 0:
                    grad = grad + group['weight_decay'] * p

                # 1. Update moments
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # 2. Compute denominator
                if group['amsgrad']:
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)

                    denom = (max_exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group['eps'])


                step_size = group['lr'] / bias_correction1
                
                # 3. Calculate the standard Adam update amount (Delta_d_adam)
                adam_update = -step_size * (exp_avg / denom)
                
                # 4. === 投影逻辑 (仅对需要投影的参数执行) ===
                if p is p_to_project and P is not None:
                    
                    # 计算 P 的平方模长 (V, 1)
                    P_sq_norm = (P * P).sum(dim=-1, keepdim=True)
                    
                    # 创建一个掩码来识别 P_i != 0 的位置
                    # 如果 P_sq_norm > 0，则需要投影；否则 P_i = 0，不需要投影
                    project_mask = P_sq_norm > 0
                    
                    # 4.1. 投影计算：只在 project_mask 为 True 的地方计算投影
                    
                    # 分子：(adam_update . P) -> shape (V, 1)
                    dot_product_num = (adam_update * P).sum(dim=-1, keepdim=True)
                    
                    # 分母：使用 P_sq_norm
                    dot_product_den = P_sq_norm.clamp(min=group['eps']) # clamp 确保不会除以零 (尽管有 mask，但为了数值稳定)
                    
                    # 投影系数 (V, 1)
                    projection_ratio = dot_product_num / dot_product_den
                    
                    # 投影后的更新量 Delta_d_proj = ratio * P
                    projected_update = projection_ratio * P
                    
                    # 4.2. 组合更新量
                    
                    # 初始化最终更新量为 Adam 更新量 (默认：无约束)
                    final_update = adam_update
                    
                    # 仅在需要投影的位置，将 Adam 更新量替换为投影后的更新量
                    final_update[project_mask.expand_as(final_update)] = \
                        projected_update[project_mask.expand_as(projected_update)]
                    
                    # 4.3. 应用最终更新量
                    p.data.add_(final_update)
                    
                else:
                    # 5. 应用标准 Adam 更新量 (如果 P is None 或 p 不是目标参数)
                    p.data.add_(adam_update)

        return loss


def edge_to_lap(activate_edges, max_index):
    neighbors = collections.defaultdict(set)

    [
        (neighbors[edge[0]].add(edge[1]), neighbors[edge[1]].add(edge[0]))
        for edge in activate_edges
    ]
    # [
    #     nei.add(i)
    #     for i, nei in enumerate(neighbors) if nei.shape > 0
    # ]

    # neighbors = [list(nei) for nei in neighbors]
    # [[i]+ for i in range(max_index) if len(neighbors[i]) > 0]
    neighbor_local = [[i] + list(neighbors[i]) if len(neighbors[i]) > 0 else [] for i in range(max_index)]
    print(len(neighbor_local))
    print("collect col for sparse matrix")
    col = np.concatenate(neighbor_local)
    print("collect row for sparse matrix")
    row = np.concatenate([[i] * len(n)
                          for i, n in enumerate(neighbor_local)])
    print("collect data for sparse matrix")
    data = np.concatenate([[1.0] + [-1.0 / (len(nei) - 1)] * (len(nei) - 1) if len(nei) > 0 else []
                           for nei in neighbor_local])
    print("generate scipy matrix")
    # create the sparse matrix
    matrix = coo_matrix((data, (row, col)),
                        shape=[max_index] * 2)
    values = matrix.data
    indices = np.vstack((matrix.row, matrix.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = matrix.shape
    print("generate torch matrix")
    return torch.sparse.FloatTensor(i, v, torch.Size(shape)).cuda()


def ml_laplacian_calculation(mesh, face_label):
    all_laplacian = []
    label = 1
    all_count = 0

    vertices = mesh.vertices.view(np.ndarray)
    max_index = vertices.shape[0]
    while all_count < 2 * face_label.shape[0]:
        print(label)
        face_mask = np.logical_or(face_label[:, 0] == label, face_label[:, 1] == label)
        face_idx = np.where(face_mask)[0]
        if len(face_idx) > 0:
            activate_edges = mesh.edges_unique[mesh.faces_unique_edges[face_idx]].reshape(-1, 2)
            all_laplacian.append(edge_to_lap(activate_edges, max_index))

        all_count += len(face_idx)
        label += 1
    return all_laplacian

def ml_laplacian_step(all_laplacian_op, samples):
    loss = torch.IntTensor(0).cuda()
    for i in range(len(all_laplacian_op)):
        laplacian_op = all_laplacian_op[i]
        laplacian_v = torch.sparse.mm(laplacian_op, samples[:, 0:3])
        laplacian_v = torch.mul(laplacian_v, laplacian_v)
        # lap_v = lap_v[head: min(head + self.max_batch, num_samples)]
        laplacian_loss = torch.sum(laplacian_v, dim=1)
        laplacian_loss = laplacian_loss.mean()
        if loss is None:
            loss = laplacian_loss
        else:
            loss += laplacian_loss
    return loss




class MIND:

    def __init__(self,query_func,resolution,r1=0.04, r2=0.01,
                 max_iter=200, sample_pc_iter=100,laplacian_weight=1000.0, bound_min=None,bound_max=None,
                 max_batch=100000, learning_rate=0.0005, warm_up_end=25,
                 report_freq=1):
        self.u = None
        self.mesh = None
        self.final_mesh = None
        self.face_label = None
        self.device = torch.device('cuda')


        # Evaluating parameters
        self.max_iter = max_iter
        self.sample_pc_iter = sample_pc_iter
        self.max_batch = max_batch
        self.report_freq = report_freq
        self.laplacian_weight = laplacian_weight
        self.warm_up_end = warm_up_end
        self.learning_rate = learning_rate
        self.resolution = resolution
        self.r1=r1
        self.r2=r2

        if bound_min is None:
            bound_min = torch.tensor([-1, -1, -1], dtype=torch.float32)
        if bound_max is None:
            bound_max = torch.tensor([1, 1, 1], dtype=torch.float32)
        if isinstance(bound_min, list):
            bound_min = torch.tensor(bound_min, dtype=torch.float32)
        if isinstance(bound_max, list):
            bound_max = torch.tensor(bound_max, dtype=torch.float32)
        if isinstance(bound_min, np.ndarray):
            bound_min = torch.from_numpy(bound_min).float()
        if isinstance(bound_max, np.ndarray):
            bound_max = torch.from_numpy(bound_max).float()
        self.bound_min = bound_min - self.r1
        self.bound_max = bound_max + self.r1

        self.optimizer = None

        self.query_func = query_func

    def extract_fields(self):

        N = 32
        X = torch.linspace(self.bound_min[0], self.bound_max[0], self.resolution).split(N)
        Y = torch.linspace(self.bound_min[1], self.bound_max[1], self.resolution).split(N)
        Z = torch.linspace(self.bound_min[2], self.bound_max[2], self.resolution).split(N)

        u = np.zeros([self.resolution, self.resolution, self.resolution], dtype=np.float32)
        # with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)

                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).cuda()
                    val = self.query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
        self.u = u
        return u

    def threshold_MC(self, ndf, threshold, resolution, bound_min=None, bound_max=None):
        try:
            vertices, triangles, _, _ = measure.marching_cubes(
                ndf, threshold, spacing=(2 / (resolution - 1), 2 / (resolution - 1), 2 / (resolution - 1)))
            vertices -= 1
            mesh = trimesh.Trimesh(vertices, triangles, process=False)
            if bound_min is not None:
                bound_min = bound_min.cpu().numpy()
                bound_max = bound_max.cpu().numpy()
                mesh.apply_scale((bound_max - bound_min) / 2)
                mesh.apply_translation((bound_min + bound_max) / 2)
        except ValueError:
            print("threshold too high")
            mesh = None
        

        return mesh

    def potential_3D(self, sources, sources_Normals, eps=1e-8, ai=0.00001, r=0.05, r2=0.01):
        '''
          input:
          calculate potential 2D
          sources: n * 2 points position, numpy
          sources_Normal: n * 2 normals, numpy
          QueryPs: m * 2 query points, numpy

          output:
          GWN values m*1, numpy
          '''
        kdtree = KDTree(sources, leafsize=20)
        result = np.zeros((self.resolution, self.resolution, self.resolution))
        mask = np.zeros((self.resolution, self.resolution, self.resolution), dtype=int)
        mask2 = np.zeros((self.resolution, self.resolution, self.resolution), dtype=int)

        N = 64
        X = torch.linspace(self.bound_min[0], self.bound_max[0], self.resolution).split(N)
        Y = torch.linspace(self.bound_min[1], self.bound_max[1], self.resolution).split(N)
        Z = torch.linspace(self.bound_min[2], self.bound_max[2], self.resolution).split(N)
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)

                    pts = torch.cat([xx.unsqueeze(-1), yy.unsqueeze(-1), zz.unsqueeze(-1)], dim=-1).cpu().numpy()
                    dd, ii = kdtree.query(pts, k=40, distance_upper_bound=1 * r, workers=16)
                    # dd = dd.reshape((N, N, N,40))
                    # ii = ii.reshape((N, N, N,40))
                    _, ii_2 = kdtree.query(pts, k=1, distance_upper_bound=r2, workers=16)
                    ii_2 = ii_2.reshape((N, N, N))
                    sub_mask = np.sum(ii == len(sources), axis=-1) > 0
                    sub_mask_2 = np.array(ii_2 == len(sources), dtype=int)
                    value = np.sum(
                        (pts[~sub_mask][:, np.newaxis] - sources[ii[~sub_mask]]) * sources_Normals[ii[~sub_mask]],
                        axis=-1) \
                            / (np.power(dd[~sub_mask], 3) + eps)
                    sub_result = np.zeros((N, N, N))
                    sub_result[~sub_mask] = np.sum(value, axis=-1)
                    result[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = sub_result
                    mask[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = sub_mask
                    mask2[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = sub_mask_2

        # dd,ii=kdtree.query(QueryPs, k=40,distance_upper_bound=1*r,workers=16)
        #
        # mask = np.sum(ii == len(sources),axis=-1)>0
        # value = np.sum((QueryPs[~mask][:,np.newaxis]-sources[ii[~mask]])*sources_Normals[ii[~mask]],axis=-1) \
        #         /(np.power(dd[~mask],3)+eps)
        # result[~mask] = np.sum(value,axis=-1)

        # dd, ii = kdtree.query(QueryPs, k=1, distance_upper_bound=r2, workers=16)
        # mask2 = np.array(ii == len(sources),dtype=int)
        # mask = mask.astype(int)
        return result, mask.astype(int), mask2.astype(int)

    def generate_pointcloud_mesh_op(self, sample_num=1000000, batch_size=20000):
        mesh = self.threshold_MC(self.extract_fields(),self.r2,self.resolution,self.bound_min,self.bound_max)
        if mesh is None:
            raise ValueError("mesh extraction failed")
        mesh.export("temp_mesh.ply")
        input_points, face_index = mesh.sample(sample_num, return_index=True)  # weighted by face area by default
        normals = torch.from_numpy(mesh.face_normals[face_index]).cuda().float()
        samples = torch.from_numpy(input_points).cuda().float()
        samples.requires_grad = True
        optimizer = AdamWithDirectionProjection([samples],projection_direction=normals, lr=0.0005)

        max_iter = self.sample_pc_iter
        for iter_step in range(max_iter):

            optimizer.zero_grad()
            head = 0
            while head < sample_num:
                sample_subset = samples[head: min(head + batch_size, samples.shape[0])]
                df_pred = self.query_func(sample_subset)
                df_pred.sum().backward()
                head += batch_size

            optimizer.step()
        head = 0
        filtered_samples = []
        while head < sample_num:
            sample_subset = samples[head: min(head + batch_size, samples.shape[0])].cuda()
            df_pred = self.query_func(sample_subset)
            sample_subset = sample_subset[df_pred.squeeze() < self.r2]
            filtered_samples.append(sample_subset.detach().cpu().numpy())
            head += batch_size
        return np.vstack(filtered_samples)

    def label_voxel(self, voxel, m1, m2):
        if np.sum(m2) == 0:
            return np.zeros_like(voxel)
        labels_out = cc3d.connected_components(voxel, connectivity=26).astype(int)
        labels_out = labels_out
        erode_label = erode(labels_out, m1, 1, 26)
        # view_data(erode_label)
        relabel = cc3d.connected_components(erode_label, connectivity=26).astype(int)
        if relabel.max() > 0:
            labels_out = grid_cut(labels_out, relabel,self.u, relabel.max(), -1)
            label_map = label_graph(labels_out, m2, labels_out.max(), 0.4)
            sub_label = inplace_label(labels_out, label_map)
            print(sub_label.max())
            # if sub_label.max()>=3:
            #     view_two(labels_out, erode_label)
        else:
            sub_label = np.zeros_like(voxel)
        # view_data(labels_out)
        # view_two(labels_out, sub_label)
        return sub_label

    def extract_mesh(self, point_cloud):
        pt = o3d.geometry.PointCloud()
        pt.points = o3d.utility.Vector3dVector(point_cloud)
        pt = pt.voxel_down_sample(voxel_size=0.0025)

        ms = pymeshlab.MeshSet()
        m = pymeshlab.Mesh(np.asarray(pt.points))
        ms.add_mesh(m, "cube_mesh",set_as_current=True)
        ms.compute_normal_for_point_clouds(k=40)
        pointcloud_normal = ms.current_mesh().vertex_normal_matrix()
        pt.normals = o3d.utility.Vector3dVector(pointcloud_normal)
        o3d.io.write_point_cloud("pc.ply", pt)
        #array = np.zeros(shape, dtype=bool)
        wn, mask, mask2 = self.potential_3D(np.asarray(pt.points),np.asarray(pt.normals),r=self.r1,r2=self.r2)
        mask2 = 1-mask2
        sign = np.zeros_like(wn,dtype=int)
        sign[wn>0] = 1
        sign[wn<0] = -1
        # view_data(wn)
        # view_data(wn_pad)
        block_size = 16
        labels = self.label_voxel(sign,mask, mask2)
        labels[mask==1]=0

        # mesh为0~1坐标系，同样需要缩放
        v, f, f_label = m3c_py(labels, np.abs(wn))
        v -= 0.5
        mesh = trimesh.Trimesh(v,f)
        self.mesh = mesh
        bound_min = self.bound_min.cpu().numpy()
        bound_max = self.bound_max.cpu().numpy()
        self.mesh.apply_scale((bound_max - bound_min) / 2)
        self.mesh.apply_translation((bound_min + bound_max) / 2)
        # self.mesh.export("result.ply")
        # view_with_mesh(expansion_labels,mesh)
        self.face_label = f_label
        return mesh, f_label
        # np.save(f"wn.npy", wn)
        # np.save(f"labels.npy", labels)

    def postprocess_mesh(self, mesh: trimesh.Trimesh, face_label: np.ndarray):
        faces = np.asarray(mesh.faces)
        vertices = np.asarray(mesh.vertices)
        inverse_mask = np.asarray(face_label[:, 0] > face_label[:, 1])
        faces[inverse_mask] = faces[inverse_mask][:, ::-1]

        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
        o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))

        # mesh.scale(2, center=(0.0, 0.0, 0.0))

        R = o3d_mesh.get_rotation_matrix_from_xyz((0, np.pi / 2, 0))
        o3d_mesh.rotate(R, center=(0, 0, 0))
        vv = np.asarray(o3d_mesh.vertices)
        vv[:, -1] = -vv[:, -1]
        trimesh_mesh = trimesh.Trimesh(vertices=vv, faces=faces, process=False)
        trimesh_mesh.merge_vertices()
        trimesh_mesh.vertices += 1 / self.resolution
        udf_value = self.query_func(torch.from_numpy(np.asarray(trimesh_mesh.vertices)))
        mask = udf_value < self.r2
        # print(mask.shape)
        face_mask = mask[trimesh_mesh.faces].any(axis=1)
        # 远离pc的面片集合
        face_idx = np.where(~face_mask)[0]
        edge_list = trimesh_mesh.edges_unique[trimesh_mesh.faces_unique_edges[face_idx]].reshape(-1, 2)
        G = nx.Graph()
        G.add_edges_from(edge_list)

        # 获取连通分量
        connected_components = list(nx.connected_components(G))
        edges_unique, edges_count = np.unique(trimesh_mesh.edges_unique_inverse, return_counts=True)
        boundary_idx = np.nonzero(edges_count < 2)[0]

        boundary_edge = trimesh_mesh.edges_unique[boundary_idx].flatten()
        boundary_ver = np.unique(boundary_edge)
        # 根据连通分量创建子图
        remove_v = []
        for component in connected_components:
            if len(component.intersection(boundary_ver)) > 0:
                remove_v += list(component)
        remove_v = np.asarray(remove_v)
        remove_v_mask = np.zeros(vertices.shape[0], dtype=bool)
        if remove_v.shape[0] > 0:
            remove_v_mask[remove_v] = True
        remain_face_mask = ~remove_v_mask[trimesh_mesh.faces].any(axis=1)
        face_label = face_label[remain_face_mask]
        trimesh_mesh.update_faces(remain_face_mask)
        # self.mesh.update_vertices(mask)
        trimesh_mesh.remove_unreferenced_vertices()
        self.mesh = trimesh_mesh
        return trimesh_mesh

    def update_learning_rate(self, iter_step):
        warn_up = self.warm_up_end
        max_iter = self.max_iter
        init_lr = self.learning_rate
        lr = (iter_step / warn_up) if iter_step < warn_up else 0.5 * (
                math.cos((iter_step - warn_up) / (max_iter - warn_up) * math.pi) + 1)
        lr = lr * init_lr
        for g in self.optimizer.param_groups:
            g['lr'] = lr

    def op_msdf_mesh(self,mesh, face_label):
        query_func = self.query_func
        xyz = torch.from_numpy(mesh.vertices.astype(np.float32)).cuda()
        xyz.requires_grad = True

        all_laplacian = ml_laplacian_calculation(mesh, face_label)
        print(f"We have {len(all_laplacian)} label for calculate laplacian")

        xyz = torch.from_numpy(mesh.vertices.astype(np.float32)).cuda()
        xyz.requires_grad = True
        num_samples = xyz.shape[0]

        # set optimizer to xyz
        self.optimizer = VectorAdam([xyz])

        for it in range(self.max_iter):

            self.update_learning_rate(it)


            self.optimizer.zero_grad()
            # xyz.grad=0
            num_samples = xyz.shape[0]
            head = 0

            grad_loss = []
            all_loss = 0

            # if it <= self.normal_step:
            while head < num_samples:
                sample_subset = xyz[head: min(head + self.max_batch, num_samples)]
                df = query_func(sample_subset)
                df_loss = df.mean()
                loss = df_loss
                all_loss +=loss.data
                loss.backward()
                head += self.max_batch

            non_manifold_lap_loss = ml_laplacian_step(all_laplacian, xyz)
            lap_loss = self.laplacian_weight * non_manifold_lap_loss
            loss = lap_loss

            loss.backward()
            # 累加到 xyz.grad
            self.optimizer.step()
            print(" {} iteration, udf loss ={},loss_non_manifold_lap={}".format(it,all_loss,non_manifold_lap_loss))
        self.final_mesh = trimesh.Trimesh(vertices=xyz.detach().cpu().numpy(), faces=mesh.faces, process=False)

        return self.final_mesh

    def run(self):
        pc = self.generate_pointcloud_mesh_op()
        mesh, face_label = self.extract_mesh(pc)
        mesh = self.postprocess_mesh(mesh, face_label)
        return self.op_msdf_mesh(mesh, face_label)



def view_data(labels_out, bound_low=None, bound_high=None):
    if bound_high is None:
        bound_high = [1, 1, 1]
    if bound_low is None:
        bound_low = [0, 0, 0]
    ps.init()

    # define the resolution and bounds of the grid

    # color = cm.get_cmap("Paired")(labels_out).astype(float)[...,:3].reshape((-1,3))
    # register the grid
    ps_grid = ps.register_volume_grid("sample grid", labels_out.shape, bound_low, bound_high)
    # add a scalar function on the grid
    ps_grid.add_scalar_quantity("node scalar1", labels_out,
                                defined_on='nodes', enabled=True)

    ps.show()


def view_two(labels_out,labels_out2, bound_low=None, bound_high=None):
    if bound_high is None:
        bound_high = [1, 1, 1]
    if bound_low is None:
        bound_low = [0, 0, 0]
    ps.init()

    # define the resolution and bounds of the grid

    # color = cm.get_cmap("Paired")(labels_out).astype(float)[...,:3].reshape((-1,3))
    # register the grid
    ps_grid = ps.register_volume_grid("sample grid", labels_out.shape, bound_low, bound_high)
    # add a scalar function on the grid
    ps_grid.add_scalar_quantity("node scalar1", labels_out,
                                defined_on='nodes', enabled=True)
    ps_grid = ps.register_volume_grid("sample grid 2", labels_out2.shape, bound_low, bound_high)
    # add a scalar function on the grid
    ps_grid.add_scalar_quantity("node scalar2", labels_out2,
                                defined_on='nodes', enabled=True)
    ps.show()