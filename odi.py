import torch.nn as nn

import torchvision.transforms as transforms
import torch
import os
from PIL import Image
import math
import os
import sys
import numpy as np
import sys
import cv2
import scipy.stats as st
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode

from PIL import Image
from odi_config import *

## Pytorch3D ########################################
# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    look_at_rotation,
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex,
    blending
)

class Render3D(object):
    def __init__(self,config_idx=1,count=1):

        exp_settings=exp_configuration[config_idx] # Load experiment configuration

        self.config_idx=config_idx
        self.count=count
        self.eval_count=0


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = self.device
        raster_settings = RasterizationSettings(
            image_size=299, 
            blur_radius=0.0, 
            faces_per_pixel=1, 
        )

        # Just initialization. light position and brightness are randomly set for each inference 
        self.lights = PointLights(device=self.device, ambient_color=((0.3, 0.3, 0.3),), diffuse_color=((0.5, 0.5, 0.5), ), specular_color=((0.5, 0.5, 0.5), ), 
        location=[[0.0, 3.0,0.0]])

        R, T = look_at_view_transform(dist=1.0, elev=0, azim=0)
        self.cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)

        self.materials = Materials(
            device=self.device,
            specular_color=[[1.0, 1.0, 1.0]],
            shininess=exp_settings['shininess']
        )

        # Note: the background color of rendered images is set to -1 for proper blending
        blend_params = blending.BlendParams(background_color=[-1., -1., -1.])


        # Create a renderer by composing a mesh rasterizer and a shader. 
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras, 
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device, 
                cameras=self.cameras,
                lights=self.lights,
                blend_params=blend_params
            )
        )
        # 3D Model setting
        # {'3d model name', ['filename', x, y, w, h, initial distance, initial elevation, initial azimuth, initial translation]}
        self.model_settings={'pack':['pack.obj',255,255,510,510,1.2,0,0,[0,0.02,0.]],
        'cup':['cup.obj',693,108,260,260,1.7,0,0,[0.,-0.1,0.]],
        'pillow':['pillow.obj',10,10,470,470,1.7,0,0],
        't_shirt':['t_shirt_lowpoly.obj',180,194,240,240,1.2,0,0,[0.0,0.05,0]],
        'book':['book.obj',715,66,510,510,1.3,0,0,[0.3,0.,0]],
        '1ball':['1ball.obj',359,84,328,328,2.1,-40,-10],
        '2ball':['2ball.obj',359,84,328,328,1.9,-40,-10,[-0.1,0.,0]],
        '3ball':['3ball.obj',359,84,328,328,1.8,-25,-10,[-0.1,0.15,0]],
        '4ball':['4ball.obj',359,84,328,328,1.8,-25,-10,[0.,0.1,0]]
        }


        self.source_models=exp_settings['source_3d_models'] # Import source model list

        self.background_img=torch.zeros((1,3,299,299)).to(device)
        
        for src_model in self.source_models:
                self.model_settings[src_model][0]=load_object(self.model_settings[src_model][0])

        # The following code snippet is for 'blurred image' backgrounds.
        kernel_size=50
        kernel = gkern(kernel_size, 15).astype(np.float32)
        gaussian_kernel = np.stack([kernel, kernel, kernel])
        gaussian_kernel = np.expand_dims(gaussian_kernel, 1)
        self.gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda()

    def render(self, img):
        self.eval_count+=1
        device = self.device
        exp_settings=exp_configuration[self.config_idx]

        # Default experimental settings.
        if 'background_type' not in exp_settings:
            exp_settings['background_type']='none'
        if 'texture_type' not in exp_settings:
            exp_settings['texture_type']='none'
        if 'visualize' not in exp_settings:
            exp_settings['visualize']=False

        x_adv=img
        # Randomly select an object from the source object pool
        pick_idx=np.random.randint(low=0,high=len(self.source_models))

        # Load the 3D mesh
        mesh=self.model_settings[self.source_models[pick_idx]][0]

        # Load the texture map
        texture_image=mesh.textures.maps_padded()

        texture_type=exp_settings['texture_type']

        if texture_type=='random_pixel':
            texture_image.data=torch.rand_like(texture_image,device=device)
        elif texture_type=='random_solid': # Default setting
            texture_image.data=torch.ones_like(texture_image,device=device)*(torch.rand((1,1,1,3),device=device)*0.6+0.1)
        elif  texture_type=='custom':
            texture_image.data=torch.ones_like(texture_image,device=device)*torch.FloatTensor( [ 0/255.,0./255.,0./255.]).view((1,1,1,3)).to(device)
        
        (pattern_h,pattern_w)=(self.model_settings[self.source_models[pick_idx]][4],self.model_settings[self.source_models[pick_idx]][3])

        # Resize the input image
        resized_x_adv=F.interpolate(x_adv, size=(pattern_h, pattern_w), mode='bilinear').permute(0,2,3,1)
        # Insert the resized image into the canvas area of the texture map
        (x,y)=self.model_settings[self.source_models[pick_idx]][1],self.model_settings[self.source_models[pick_idx]][2]
        texture_image[:,y:y+pattern_h,x:x+pattern_w,:]=resized_x_adv

        # Adjust the light parameters
        self.lights.location = torch.tensor(exp_settings['light_location'], device=device)[None]+(torch.rand((3,), device=device)*exp_settings['rand_light_location']-exp_settings['rand_light_location']/2)
        self.lights.ambient_color=torch.tensor([exp_settings['ambient_color']]*3, device=device)[None]+(torch.rand((1,),device=self.device)*exp_settings['rand_ambient_color'])
        self.lights.diffuse_color=torch.tensor([exp_settings['diffuse_color']]*3, device=device)[None]+(torch.rand((1,),device=self.device)*exp_settings['rand_diffuse_color'])
        self.lights.specular_color=torch.tensor([exp_settings['specular_color']]*3, device=device)[None]

        
        # Adjust the camera parameters
        rand_elev=torch.randint(exp_settings['rand_elev'][0],exp_settings['rand_elev'][1]+1, (1,))
        rand_azim=torch.randint(exp_settings['rand_azim'][0],exp_settings['rand_azim'][1]+1, (1,))
        rand_dist=(torch.rand((1,))*exp_settings['rand_dist']+exp_settings['min_dist'])
        rand_angle=torch.randint(exp_settings['rand_angle'][0],exp_settings['rand_angle'][1]+1, (1,))



        R, T = look_at_view_transform(dist=(self.model_settings[self.source_models[pick_idx]][5])*rand_dist, elev=self.model_settings[self.source_models[pick_idx]][6]+rand_elev, 
        azim=self.model_settings[self.source_models[pick_idx]][7]+rand_azim,up=((0,1,0),))

        if len(self.model_settings[self.source_models[pick_idx]])>8: # Apply initial translation if it is given.
            TT=T+torch.FloatTensor(self.model_settings[self.source_models[pick_idx]][8])
        else:
            TT=T

        # Compute rotation matrix for tilt
        angles=torch.FloatTensor([[0,0,rand_angle*math.pi/180]]).to(device)
        rot=compute_rotation(angles).squeeze()
        R=R.to(device)

        R=torch.matmul(rot,R)

        self.cameras = FoVPerspectiveCameras(device=self.device, R=R, T=TT)

        # Render the mesh with the modified rendering environments.
        rendered_img = self.renderer(mesh, lights=self.lights, materials=self.materials, cameras=self.cameras)

        rendered_img=rendered_img[:, :, :,:3] # RGBA -> RGB

        rendered_img=rendered_img.permute(0,3,1,2) # B X H X W X C -> B X C X H X W

        background_type=exp_settings['background_type']
        
        # The following code snippet is for blending
        rendered_img_mask = 1.-(rendered_img.sum(dim=1,keepdim=True)==-3.).float()
        rendered_img = torch.clamp(rendered_img, 0., 1.)
        if background_type=='random_pixel':
            background_img=torch.rand_like(rendered_img,device=device)
            result_img = background_img * (1 - rendered_img_mask) + rendered_img * rendered_img_mask
        elif background_type=='random_solid':
            background_img=torch.ones_like(rendered_img,device=device)*torch.rand((1,3,1,1),device=device)
            result_img = background_img * (1 - rendered_img_mask) + rendered_img * rendered_img_mask
        elif background_type=='blurred_image':
            background_img=img.clone().detach()
            background_img = F.conv2d(background_img, self.gaussian_kernel, bias=None, stride=1, padding='same', groups=3)
            result_img = background_img * (1 - rendered_img_mask) + rendered_img * rendered_img_mask
        elif background_type=='custom':
            background_img=torch.ones_like(rendered_img,device=device)*torch.FloatTensor( [ 0/255.,0./255.,0./255.]).view((1,3,1,1)).to(device)
            result_img = background_img * (1 - rendered_img_mask) + rendered_img * rendered_img_mask
        else:
            result_img=rendered_img

        if exp_settings['visualize']==True:
            result_img_npy=result_img.permute(0,2,3,1)
            result_img_npy=result_img_npy.squeeze().cpu().detach().numpy()
            converted_img=cv2.cvtColor(result_img_npy, cv2.COLOR_BGR2RGB)
            cv2.imshow('Video', converted_img) #[0, ..., :3]
            key=cv2.waitKey(1) & 0xFF

        return result_img

def compute_rotation(angles):
    """
    Return:
        rot              -- torch.tensor, size (B, 3, 3) pts @ trans_mat
    Parameters:
        angles           -- torch.tensor, size (B, 3), radian
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = angles.shape[0]
    ones = torch.ones([batch_size, 1]).to(device)
    zeros = torch.zeros([batch_size, 1]).to(device)
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

def rigid_transform( vs, rot, trans):
    vs_r = torch.matmul(vs, rot)
    vs_t = vs_r + trans.view(-1, 1, 3)
    return vs_t

def load_object(obj_file_name):
    obj_filename = os.path.join("./data", obj_file_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the 3D model using load_obj
    verts, faces, aux = load_obj(obj_filename)
    
    faces_idx = faces.verts_idx.to(device)
    verts = verts.to(device)

    # We scale normalize and center the mesh. 
    center = verts.mean(0)
    verts = verts - center
    scale = max(verts.abs().max(0)[0])
    verts = verts / scale
    angles=torch.FloatTensor([[90*math.pi/180,0,0]]).to(device)

    rot=compute_rotation(angles).squeeze()

    verts=torch.matmul(verts,rot)

    # Get the scale normalized textured mesh
    mesh = load_objs_as_meshes([obj_filename], device=device)
    mesh = Meshes(verts=[verts], faces=[faces_idx],textures=mesh.textures)


    return mesh

def render_3d_aug_input(x_adv, renderer,prob=1.0):
    '''
    We set prob as 1. Due to config_idx=101
    '''
    c = np.random.rand(1)
    if c <= prob:
        x_ri=x_adv.clone()
        for i in range(x_adv.shape[0]):
            x_ri[i]=renderer.render(x_adv[i].unsqueeze(0))
        return  x_ri 
    else:
        return  x_adv

def gkern(kernlen=15, nsig=3):
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel