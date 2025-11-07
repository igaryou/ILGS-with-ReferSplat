import torch
import math
from scene.gaussian_model import GaussianModel
from diff_gaussian_rasterization import GaussianRasterizationSettings

def GaussianSetting(viewpoint_camera, pc : GaussianModel, pipe, bg_color, scaling_modifier = 1.0):

    tanfovx = math.tan(viewpoint_camera.FoVx*0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy*0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    return raster_settings

