import os
import torch
# import torch.nn as nn
from scipy.io import loadmat
import numpy as np
import torch.nn.functional as F

# CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))


def perspective_projection(focal, center):
    # return p.T (N, 3) @ (3, 3)
    return np.array([
        focal, 0, center,
        0, focal, center,
        0, 0, 1
    ]).reshape([3, 3]).astype(np.float32).transpose()


class SH:
    def __init__(self):
        self.a = [np.pi, 2 * np.pi / np.sqrt(3.), 2 * np.pi / np.sqrt(8.)]
        self.c = [1/np.sqrt(4 * np.pi), np.sqrt(3.) / np.sqrt(4 * np.pi), 3 * np.sqrt(5.) / np.sqrt(12 * np.pi)]


class BFM(torch.nn.Module):
    # BFM 3D face model
    def __init__(self,
                 recenter=True,
                 camera_distance=10.,
                 init_lit=np.array([0.8, 0, 0, 0, 0, 0, 0, 0, 0]),
                 focal=1015.,
                 image_size=224,
                 bfm_model_path='pretrained/BFM_model_front.mat'
                 ):
        super().__init__()
        model = loadmat(bfm_model_path)
        # self.bfm_uv = loadmat(os.path.join(CURRENT_PATH, 'BFM/BFM_UV.mat'))
        # print(model.keys())
        # mean face shape. [3*N,1]
        # self.meanshape = torch.from_numpy(model['meanshape'])
        self.register_buffer('meanshape', torch.from_numpy(model['meanshape']).float())

        if recenter:
            meanshape = self.meanshape.view(-1, 3)
            meanshape = meanshape - torch.mean(meanshape, dim=0, keepdim=True)
            self.meanshape = meanshape.view(-1, 1)

        # identity basis. [3*N,80]
        # self.idBase = torch.from_numpy(model['idBase'])
        self.register_buffer('idBase', torch.from_numpy(model['idBase']).float())
        # self.idBase = nn.Parameter(torch.from_numpy(model['idBase']).float())
        # self.exBase = torch.from_numpy(model['exBase'].astype(
        #     np.float32))    # expression basis. [3*N,64]
        self.register_buffer('exBase', torch.from_numpy(model['exBase']).float())
        # self.exBase = nn.Parameter(torch.from_numpy(model['exBase']).float())
        # mean face texture. [3*N,1] (0-255)
        # self.meantex = torch.from_numpy(model['meantex'])
        self.register_buffer('meantex', torch.from_numpy(model['meantex']).float())
        # texture basis. [3*N,80]
        # self.texBase = torch.from_numpy(model['texBase'])
        self.register_buffer('texBase', torch.from_numpy(model['texBase']).float())
        # self.texBase = nn.Parameter(torch.from_numpy(model['texBase']).float())

        # triangle indices for each vertex that lies in. starts from 0. [N,8]
        self.register_buffer('point_buf', torch.from_numpy(model['point_buf']).long()-1)
        # self.point_buf = model['point_buf'].astype(np.int32)
        # vertex indices in each triangle. starts from 0. [F,3]
        self.register_buffer('face_buf', torch.from_numpy(model['tri']).long()-1)
        # self.tri = model['tri'].astype(np.int32)
        # vertex indices of 68 facial landmarks. starts from 0. [68]
        self.register_buffer('keypoints', torch.from_numpy(model['keypoints']).long().view(68)-1)
        # self.keypoints = model['keypoints'].astype(np.int32)[0]
        # print(self.keypoints)
        # print('keypoints', self.keypoints)

        # vertex indices for small face region to compute photometric error. starts from 0.
        # self.front_mask = np.squeeze(model['frontmask2_idx']).astype(np.int64) - 1
        self.register_buffer('front_mask', torch.from_numpy(np.squeeze(model['frontmask2_idx'])).long()-1)
        # vertex indices for each face from small face region. starts from 0. [f,3]
        # self.front_face_buf = model['tri_mask2'].astype(np.int64) - 1
        self.register_buffer('front_face_buf', torch.from_numpy(np.squeeze(model['tri_mask2'])).long() - 1)
        # vertex indices for pre-defined skin region to compute reflectance loss
        # self.skin_mask = np.squeeze(model['skinmask'])
        self.register_buffer('skin_mask', torch.from_numpy(np.squeeze(model['skinmask'])))


        # keypoints_222 = []
        # with open(os.path.join(CURRENT_PATH, 'BFM/D3DFR_222.txt'), 'r') as f:
        #     for line in f.readlines():
        #         idx = int(line.strip())
        #         keypoints_222.append(max(idx, 0))
        # self.register_buffer('keypoints_222', torch.from_numpy(np.array(keypoints_222)).long())

        # (1) right eye outer corner, (2) right eye inner corner, (3) left eye inner corner, (4) left eye outer corner,
        # (5) nose bottom, (6) right mouth corner, (7) left mouth corner
        self.register_buffer('keypoints_7', self.keypoints[[36, 39, 42, 45, 33, 48, 54]])

        # self.persc_proj = torch.from_numpy(perspective_projection(focal, center)).float()
        self.register_buffer('persc_proj', torch.from_numpy(perspective_projection(focal, image_size/2)))
        self.camera_distance = camera_distance
        self.image_size = image_size
        self.SH = SH()
        # self.init_lit = init_lit.reshape([1, 1, -1]).astype(np.float32)
        self.register_buffer('init_lit', torch.from_numpy(init_lit.reshape([1, 1, -1]).astype(np.float32)))

        # (1) right eye outer corner, (2) right eye inner corner, (3) left eye inner corner, (4) left eye outer corner,
        # (5) nose bottom, (6) right mouth corner, (7) left mouth corner
        # print(self.keypoints[[36, 39, 42, 45, 33, 48, 54]])

        # Lm3D = loadmat(os.path.join(CURRENT_PATH, 'BFM/similarity_Lm3D_all.mat'))
        # Lm3D = Lm3D['lm']
        # # print(Lm3D)
        #
        # # calculate 5 facial landmarks using 68 landmarks
        # lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
        # Lm3D = np.stack([Lm3D[lm_idx[0], :], np.mean(Lm3D[lm_idx[[1, 2]], :], 0), np.mean(
        #     Lm3D[lm_idx[[3, 4]], :], 0), Lm3D[lm_idx[5], :], Lm3D[lm_idx[6], :]], axis=0)
        # Lm3D = Lm3D[[1, 2, 0, 3, 4], :]
        # self.Lm3D = Lm3D
        # print(Lm3D.shape)

    def split_coeff(self, coeff):
        # input: coeff with shape [1,258]
        id_coeff = coeff[:, 0:80]        # identity(shape) coeff of dim 80
        ex_coeff = coeff[:, 80:144]      # expression coeff of dim 64
        tex_coeff = coeff[:, 144:224]    # texture(albedo) coeff of dim 80
        gamma = coeff[:, 227:254]        # lighting coeff for 3 channel SH function of dim 27
        angles = coeff[:, 224:227]       # ruler angles(x,y,z) for rotation of dim 3
        translation = coeff[:, 254:257]  # translation coeff of dim 3

        return id_coeff, ex_coeff, tex_coeff, gamma, angles, translation

    def compute_exp_deform(self, exp_coeff):
        exp_part = torch.einsum('ij,aj->ai', self.exBase, exp_coeff)
        return exp_part

    def compute_id_deform(self, id_coeff):
        id_part = torch.einsum('ij,aj->ai', self.idBase, id_coeff)
        return id_part

    def compute_shape_from_coeff(self, coeff):
        id_coeff = coeff[:, 0:80]
        ex_coeff = coeff[:, 80:144]
        batch_size = coeff.shape[0]
        id_part = torch.einsum('ij,aj->ai', self.idBase, id_coeff)  #B, n
        exp_part = torch.einsum('ij,aj->ai', self.exBase, ex_coeff) #B, n
        face_shape = id_part + exp_part + self.meanshape.view(1, -1)
        return face_shape.view(batch_size, -1, 3)

    def compute_shape(self, id_coeff, exp_coeff):
        """
        Return:
            face_shape       -- torch.tensor, size (B, N, 3)
        Parameters:
            id_coeff         -- torch.tensor, size (B, 80), identity coeffs
            id_relative_scale  -- torch.tensor, size (B, 1), identity coeffs
            exp_coeff        -- torch.tensor, size (B, 64), expression coeffs
        """
        batch_size = id_coeff.shape[0]
        id_part = torch.einsum('ij,aj->ai', self.idBase, id_coeff)  #B, n
        exp_part = torch.einsum('ij,aj->ai', self.exBase, exp_coeff) #B, n
        face_shape = id_part + exp_part + self.meanshape.view(1, -1)
        return face_shape.view(batch_size, -1, 3)

    def compute_texture(self, tex_coeff, normalize=True):
        """
        Return:
            face_texture     -- torch.tensor, size (B, N, 3), in RGB order, range (0, 1.)
        Parameters:
            tex_coeff        -- torch.tensor, size (B, 80)
        """
        batch_size = tex_coeff.shape[0]
        face_texture = torch.einsum('ij,aj->ai', self.texBase, tex_coeff) + self.meantex
        if normalize:
            face_texture = face_texture / 255.
        return face_texture.view(batch_size, -1, 3)

    def compute_norm(self, face_shape):
        """
        Return:
            vertex_norm      -- torch.tensor, size (B, N, 3)
        Parameters:
            face_shape       -- torch.tensor, size (B, N, 3)
        """

        v1 = face_shape[:, self.face_buf[:, 0]]
        v2 = face_shape[:, self.face_buf[:, 1]]
        v3 = face_shape[:, self.face_buf[:, 2]]
        e1 = v1 - v2
        e2 = v2 - v3
        face_norm = torch.cross(e1, e2, dim=-1)
        face_norm = F.normalize(face_norm, dim=-1, p=2)
        face_norm = torch.cat([face_norm, torch.zeros(face_norm.shape[0], 1, 3).to(self.meanshape)], dim=1)

        vertex_norm = torch.sum(face_norm[:, self.point_buf], dim=2)
        vertex_norm = F.normalize(vertex_norm, dim=-1, p=2)
        return vertex_norm

    def compute_color(self, face_texture, face_norm, gamma):
        """
        Return:
            face_color       -- torch.tensor, size (B, N, 3), range (0, 1.)
        Parameters:
            face_texture     -- torch.tensor, size (B, N, 3), from texture model, range (0, 1.)
            face_norm        -- torch.tensor, size (B, N, 3), rotated face normal
            gamma            -- torch.tensor, size (B, 27), SH coeffs
        """
        batch_size = gamma.shape[0]
        v_num = face_texture.shape[1]
        a, c = self.SH.a, self.SH.c
        gamma = gamma.reshape([batch_size, 3, 9])
        gamma = gamma + self.init_lit
        gamma = gamma.permute(0, 2, 1)
        Y = torch.cat([
            a[0] * c[0] * torch.ones_like(face_norm[..., :1]).to(self.meanshape),
            -a[1] * c[1] * face_norm[..., 1:2],
            a[1] * c[1] * face_norm[..., 2:],
            -a[1] * c[1] * face_norm[..., :1],
            a[2] * c[2] * face_norm[..., :1] * face_norm[..., 1:2],
            -a[2] * c[2] * face_norm[..., 1:2] * face_norm[..., 2:],
            0.5 * a[2] * c[2] / np.sqrt(3.) * (3 * face_norm[..., 2:] ** 2 - 1),
            -a[2] * c[2] * face_norm[..., :1] * face_norm[..., 2:],
            0.5 * a[2] * c[2] * (face_norm[..., :1] ** 2 - face_norm[..., 1:2] ** 2)
        ], dim=-1)
        r = Y @ gamma[..., :1]
        g = Y @ gamma[..., 1:2]
        b = Y @ gamma[..., 2:]
        face_color = torch.cat([r, g, b], dim=-1) * face_texture
        return face_color

    def compute_rotation(self, angles):
        """
        Return:
            rot              -- torch.tensor, size (B, 3, 3) pts @ trans_mat
        Parameters:
            angles           -- torch.tensor, size (B, 3), radian
        """

        batch_size = angles.shape[0]
        ones = torch.ones([batch_size, 1]).to(self.meanshape)
        zeros = torch.zeros([batch_size, 1]).to(self.meanshape)
        x, y, z = angles[:, :1], angles[:, 1:2], angles[:, 2:],

        rot_x = torch.cat([
            ones, zeros, zeros,
            zeros, torch.cos(x), -torch.sin(x),
            zeros, torch.sin(x), torch.cos(x)
        ], dim=1).reshape([batch_size, 3, 3])

        rot_y = torch.cat([
            torch.cos(y), zeros, torch.sin(y),
            zeros, ones, zeros,
            -torch.sin(y), zeros, torch.cos(y)
        ], dim=1).reshape([batch_size, 3, 3])

        rot_z = torch.cat([
            torch.cos(z), -torch.sin(z), zeros,
            torch.sin(z), torch.cos(z), zeros,
            zeros, zeros, ones
        ], dim=1).reshape([batch_size, 3, 3])

        rot = rot_z @ rot_y @ rot_x
        return rot.permute(0, 2, 1)

    def to_camera(self, face_shape):
        face_shape[..., -1] = self.camera_distance - face_shape[..., -1]
        return face_shape

    def to_image(self, face_shape):
        """
        Return:
            face_proj        -- torch.tensor, size (B, N, 2), y direction is opposite to v direction
        Parameters:
            face_shape       -- torch.tensor, size (B, N, 3)
        """
        # to image_plane
        face_proj = face_shape @ self.persc_proj
        # print(face_proj.shape)
        face_proj = face_proj[..., :2] / face_proj[..., 2:]

        return face_proj

    def rotate(self, face_shape, rot):
        """
        Return:
            face_shape       -- torch.tensor, size (B, N, 3) pts @ rot + trans
        Parameters:
            face_shape       -- torch.tensor, size (B, N, 3)
            rot              -- torch.tensor, size (B, 3, 3)

        """
        return face_shape @ rot

    def get_landmarks7(self, face_proj):
        """
        Return:
            face_lms         -- torch.tensor, size (B, 68, 2)
        Parameters:
            face_proj       -- torch.tensor, size (B, N, 2)
        """
        return face_proj[:, self.keypoints_7, :]

    def get_landmarks68(self, face_proj):
        """
        Return:
            face_lms         -- torch.tensor, size (B, 68, 2)
        Parameters:
            face_proj       -- torch.tensor, size (B, N, 2)
        """
        return face_proj[:, self.keypoints, :]

    def get_landmarks222(self, face_proj):
        """
        Return:
            face_lms         -- torch.tensor, size (B, 68, 2)
        Parameters:
            face_proj       -- torch.tensor, size (B, N, 2)
        """
        return face_proj[:, self.keypoints_222, :]

    def compute_for_render(self, coeffs):
        """
        Return:
            face_vertex     -- torch.tensor, size (B, N, 3), in camera coordinate
            face_color      -- torch.tensor, size (B, N, 3), in RGB order
            landmark        -- torch.tensor, size (B, 68, 2), y direction is opposite to v direction
        Parameters:
            coeffs          -- torch.tensor, size (B, 258)
        """
        id_coeff, ex_coeff, tex_coeff, gamma, angles, translation = self.split_coeff(coeffs)
        # id_relative_scale = id_relative_scale.clamp(0.9,1.1)
        face_shape = self.compute_shape(id_coeff, ex_coeff)
        # face_shape_noexp = self.compute_shape(id_coeff, torch.zeros_like(ex_coeff))
        # print(face_shape.size())
        rotation = self.compute_rotation(angles)
        # print('rotation')

        face_shape_rotated = self.rotate(face_shape, rotation)
        face_shape_transformed = face_shape_rotated + translation.unsqueeze(1)
        face_vertex = self.to_camera(face_shape_transformed)
        face_proj = self.to_image(face_vertex)

        # face_shape_transformed_noexp = self.transform(face_shape_noexp, rotation, translation, scale_xyz)
        # face_vertex_noexp = self.to_camera(face_shape_transformed_noexp)

        landmark68 = self.get_landmarks68(face_proj)
        # landmark_face = face_proj[:,self.front_mask[::32], :]
        landmark68[:, :, 1] = self.image_size - 1 - landmark68[:, :, 1]

        face_texture = self.compute_texture(tex_coeff)
        face_norm_roted = self.compute_norm(face_shape_rotated)
        # face_norm_roted = face_norm @ rotation
        face_color = self.compute_color(face_texture, face_norm_roted, gamma)

        # face_norm_noexp = self.compute_norm(face_shape_noexp)
        # face_norm_noexp_roted = face_norm_noexp @ rotation
        # face_color_noexp = self.compute_color(face_texture, face_norm_noexp_roted, gamma)

        return face_shape, face_vertex, face_color, face_texture, landmark68
    
    def get_lm68(self, coeffs):
        id_coeff, ex_coeff, tex_coeff, gamma, angles, translation = self.split_coeff(coeffs)
        # id_relative_scale = id_relative_scale.clamp(0.9,1.1)
        face_shape = self.compute_shape(id_coeff, ex_coeff)
        # face_shape_noexp = self.compute_shape(id_coeff, torch.zeros_like(ex_coeff))
        # print(face_shape.size())
        rotation = self.compute_rotation(angles)
        # print('rotation')

        face_shape_rotated = self.rotate(face_shape, rotation)
        face_shape_transformed = face_shape_rotated + translation.unsqueeze(1)
        face_vertex = self.to_camera(face_shape_transformed)
        face_proj = self.to_image(face_vertex)

        landmark68 = self.get_landmarks68(face_proj)
        # landmark_face = face_proj[:,self.front_mask[::32], :]
        landmark68[:, :, 1] = self.image_size - 1 - landmark68[:, :, 1]
        return landmark68

    def get_vertex(self, coeffs):
        id_coeff, ex_coeff, tex_coeff, gamma, angles, translation = self.split_coeff(coeffs)
        # id_relative_scale = id_relative_scale.clamp(0.9,1.1)
        face_shape = self.compute_shape(id_coeff, ex_coeff)
        # face_shape_noexp = self.compute_shape(id_coeff, torch.zeros_like(ex_coeff))
        # print(face_shape.size())
        rotation = self.compute_rotation(angles)
        # print('rotation')

        face_shape_rotated = self.rotate(face_shape, rotation)
        face_shape_transformed = face_shape_rotated + translation.unsqueeze(1)
        face_vertex = self.to_camera(face_shape_transformed)
        face_proj = self.to_image(face_vertex)

        return face_proj
        

    def forward(self, coeffs):
        face_shape, face_vertex, face_color, face_texture, landmark68 = self.compute_for_render(coeffs)
        return face_shape, face_vertex, face_color, face_texture, landmark68


    def save_obj(self, coeff, obj_name):
        # The image size is 224 * 224
        # face reconstruction with coeff and BFM model
        id_coeff, ex_coeff, tex_coeff, gamma, angles, translation = self.split_coeff(coeff)

        # compute face shape
        face_shape = self.compute_shape(id_coeff, ex_coeff).cpu().numpy()[0]
        face_tri = self.face_buf.cpu().numpy()

        with open(obj_name, 'w') as fobj:
            for i in range(face_shape.shape[0]):
                fobj.write(
                    'v ' + str(face_shape[i][0]) + ' ' + str(face_shape[i][1]) + ' ' + str(face_shape[i][2]) + '\n')

            # start from 1
            for i in range(face_tri.shape[0]):
                fobj.write('f ' + str(face_tri[i][0] + 1) + ' ' + str(face_tri[i][1] + 1) + ' ' + str(
                    face_tri[i][2] + 1) + '\n')

        # lm7 = face_shape[[2215,  5828, 10455, 14066,  8204,  5522, 10795], :]
        # with open(obj_name[:-3]+'txt', 'w') as f:
        #     for point in lm7:
        #         f.write('{} {} {}\n'.format(point[0], point[1], point[2]))

    def save_neutral_obj(self, coeff, obj_name):
        # The image size is 224 * 224
        # face reconstruction with coeff and BFM model
        id_coeff, ex_coeff, tex_coeff, gamma, angles, translation = self.split_coeff(coeff)

        # compute face shape
        face_shape = self.compute_shape(id_coeff, ex_coeff*0).cpu().numpy()[0]
        face_tri = self.face_buf.cpu().numpy()

        with open(obj_name, 'w') as fobj:
            for i in range(face_shape.shape[0]):
                fobj.write(
                    'v ' + str(face_shape[i][0]) + ' ' + str(face_shape[i][1]) + ' ' + str(face_shape[i][2]) + '\n')

            # start from 1
            for i in range(face_tri.shape[0]):
                fobj.write('f ' + str(face_tri[i][0] + 1) + ' ' + str(face_tri[i][1] + 1) + ' ' + str(
                    face_tri[i][2] + 1) + '\n')

        # lm7 = face_shape[[2215,  5828, 10455, 14066,  8204,  5522, 10795], :]
        # with open(obj_name[:-3]+'txt', 'w') as f:
        #     for point in lm7:
        #         f.write('{} {} {}\n'.format(point[0], point[1], point[2]))

    # def clip(self, g_ratio=0.1, t_ratio=0.1):
    #     self.idBase.data = torch.minimum(torch.maximum(self.idBase_org * (1 - g_ratio), self.idBase.data), self.idBase_org * (1 + g_ratio))
    #     self.exBase.data = self.exBase_org #torch.minimum(torch.maximum(self.exBase_org * (1 - 0.001), self.exBase.data), self.exBase_org * (1 + 0.001))
    #     self.texBase.data = torch.minimum(torch.maximum(self.texBase_org * (1 - t_ratio), self.texBase.data), self.texBase_org * (1 + t_ratio))


