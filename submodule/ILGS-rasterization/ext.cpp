/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <torch/extension.h>
#include "rasterize_points.h"

std::tuple<at::Tensor, at::Tensor, at::Tensor>
VisibilityPrepassCUDA(
    const at::Tensor& means3D,
    const at::Tensor& scales,
    const float scale_modifier,
    const at::Tensor& rotations,
    const at::Tensor& opacities,
    const at::Tensor& shs,
    const at::Tensor& sh_objs,
    at::Tensor& clamped,
    const at::Tensor& cov3D_precomp,
    const at::Tensor& colors_precomp,
    const at::Tensor& viewmatrix,
    const at::Tensor& projmatrix,
    const at::Tensor& cam_pos,
    const int W, const int H,
    const float focal_x, const float focal_y,
    const float tan_fovx, const float tan_fovy,
    const bool prefiltered
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("mark_visible", &markVisible);
  m.def("visibility_prepass", &VisibilityPrepassCUDA, "...投影のみ...");
}