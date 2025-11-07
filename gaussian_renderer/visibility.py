from diff_gaussian_rasterization import visibility_prepass
import torch

def visibility_prepass_from_model(gaussians, rs):
    dev = gaussians.get_xyz().device
    P = gaussians.get_xyz().shape[0]

    means3D   = gaussians.get_xyz().to(torch.float32).contiguous()
    scales    = gaussians.get_scaling().to(torch.float32).contiguous()
    rotations = gaussians.get_rotation().to(torch.float32).contiguous()
    opacities = gaussians.get_opacity().to(torch.float32).contiguous()

    # 無ければ空テンソルでOK（C++側で nullptr 許容実装にしていない場合は空Tensorで）
    get_feats = getattr(gaussians, "get_features", None)
    shs     = get_feats().to(torch.float32).contiguous() if callable(get_feats) else torch.empty(0, device=dev)
    sh_objs = getattr(gaussians, "_objects_dc", torch.empty(0, device=dev))
    if not isinstance(sh_objs, torch.Tensor):
        sh_objs = torch.empty(0, device=dev)

    clamped      = torch.zeros(P, dtype=torch.bool, device=dev)
    cov3D_pre    = torch.empty(0, device=dev)
    colors_pre   = torch.empty(0, device=dev)

    # 焦点距離（必要なら rs に持たせた値を使う）
    fx = float(getattr(rs, "focal_x", rs.image_width  / (2.0 * rs.tanfovx)))
    fy = float(getattr(rs, "focal_y", rs.image_height / (2.0 * rs.tanfovy)))

    return visibility_prepass(
        means3D, scales, float(rs.scale_modifier), rotations, opacities,
        shs, sh_objs, clamped, cov3D_pre, colors_pre,
        rs.viewmatrix, rs.projmatrix, rs.campos,
        int(rs.image_width), int(rs.image_height),
        fx, fy, float(rs.tanfovx), float(rs.tanfovy),
        bool(rs.prefiltered)
    )