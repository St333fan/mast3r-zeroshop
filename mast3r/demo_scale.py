#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# sparse gradio demo functions
# --------------------------------------------------------
import math
import gradio
import os
import numpy as np
import functools
import trimesh
import copy
from scipy.spatial.transform import Rotation
import tempfile
import shutil
import torch

from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.cloud_opt.tsdf_optimizer import TSDFPostProcess
from mast3r.image_pairs import make_pairs
from mast3r.retrieval.processor import Retriever

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from dust3r.demo import get_args_parser as dust3r_get_args_parser

import matplotlib.pyplot as pl


class SparseGAState:
    def __init__(self, sparse_ga, should_delete=False, cache_dir=None, outfile_name=None):
        self.sparse_ga = sparse_ga
        self.cache_dir = cache_dir
        self.outfile_name = outfile_name
        self.should_delete = should_delete

    def __del__(self):
        if not self.should_delete:
            return
        if self.cache_dir is not None and os.path.isdir(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        self.cache_dir = None
        if self.outfile_name is not None and os.path.isfile(self.outfile_name):
            os.remove(self.outfile_name)
        self.outfile_name = None


def get_args_parser():
    parser = dust3r_get_args_parser()
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--gradio_delete_cache', default=None, type=int,
                        help='age/frequency at which gradio removes the file. If >0, matching cache is purged')
    parser.add_argument('--retrieval_model', default=None, type=str, help="retrieval_model to be loaded")
    parser.add_argument('--input_path', default=None, type=str, help="input path for automated demo")
    parser.add_argument('--output_path', default=None, type=str, help="output path for automated demo")
    parser.add_argument('--input_mask_path', default=None, type=str, help="optional input mask path for automated demo")

    actions = parser._actions
    for action in actions:
        if action.dest == 'model_name':
            action.choices = ["MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"]
    # change defaults
    parser.prog = 'mast3r demo'
    return parser


def _convert_scene_output_to_glb(outfile, imgs, pts3d, mask, focals, cams2world, cam_size=0.05,
                                 cam_color=None, as_pointcloud=False,
                                 transparent_cams=False, silent=False):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    # full pointcloud
    if as_pointcloud:
        pts = np.concatenate([p[m.ravel()] for p, m in zip(pts3d, mask)]).reshape(-1, 3)
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)]).reshape(-1, 3)
        valid_msk = np.isfinite(pts.sum(axis=1))
        pct = trimesh.PointCloud(pts[valid_msk], colors=col[valid_msk])
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            pts3d_i = pts3d[i].reshape(imgs[i].shape)
            msk_i = mask[i] & np.isfinite(pts3d_i.sum(axis=-1))
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d_i, msk_i))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    # add each camera
    for i, pose_c2w in enumerate(cams2world):
        if isinstance(cam_color, list):
            camera_edge_color = cam_color[i]
        else:
            camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
        add_scene_cam(scene, pose_c2w, camera_edge_color,
                      None if transparent_cams else imgs[i], focals[i],
                      imsize=imgs[i].shape[1::-1], screen_width=cam_size)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    if not silent:
        print('(exporting 3D scene to', outfile, ')')
    scene.export(file_obj=outfile)
    return outfile


def get_3D_model_from_scene(silent, scene_state, min_conf_thr=2, as_pointcloud=False, mask_sky=False,
                            clean_depth=False, transparent_cams=False, cam_size=0.05, TSDF_thresh=0, mask_path=None, outdir=None):
    """
    extract 3D_model (glb file) from a reconstructed scene
    """
    if scene_state is None:
        return None
    outfile = scene_state.outfile_name
    if outfile is None:
        return None

    # get optimized values from scene
    scene = scene_state.sparse_ga
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()

    # 3D pointcloud from depthmap, poses and intrinsics
    if TSDF_thresh > 0:
        tsdf = TSDFPostProcess(scene, TSDF_thresh=TSDF_thresh)
        pts3d, _, confs = to_numpy(tsdf.get_dense_pts3d(clean_depth=clean_depth))
    else:
        pts3d, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=clean_depth))

    #msk = to_numpy([c > min_conf_thr for c in confs])
    msk = to_numpy([c > 0.8 for c in confs])
    pts3dn = to_numpy(pts3d)
    rgbimgn = to_numpy(rgbimg)
    focalsn = to_numpy(focals)
    cams2worldn = to_numpy(cams2world)

    # Apply mask filtering to keep only confident points
    pts3dn_filtered = []
    for i, (pts, mask) in enumerate(zip(pts3dn, msk)):
        # Reshape points to match mask dimensions if needed
        pts_reshaped = pts.reshape(mask.shape + (3,))
        # Apply mask to filter points
        filtered_pts = pts_reshaped[mask]
        pts3dn_filtered.append(filtered_pts)

    # Now use pts3dn_filtered instead of pts3dn for further processing
    pts3dn = pts3dn_filtered

    img_shape = rgbimgn[0].shape[:2]
    pts_seen, height = get_pts_seen_by_cam0_with_mask(
    pts3dn, cams2worldn, focalsn, img_shape, 
    mask_path=mask_path, 
    depth_path=outdir + "/depth_mm_16bit.png",  # Needs to be correct 64x36 or else depth image
    depth_tolerance_cm=20.0  # 25cm tolerance
    )
    print("Points seen by cam0 and inside mask:", pts_seen.shape[0])
    # Example usage after extracting pts3d:
    save_points_as_ply(pts_seen, outdir + "/scene_points.ply")

    return _convert_scene_output_to_glb(outfile, rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud,
                                        transparent_cams=transparent_cams, cam_size=cam_size, silent=silent), height


def get_reconstructed_scene(outdir, gradio_delete_cache, model, retrieval_model, device, silent, image_size,
                            current_scene_state, filelist, optim_level, lr1, niter1, lr2, niter2, min_conf_thr,
                            matching_conf_thr, as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size,
                            scenegraph_type, winsize, win_cyclic, refid, TSDF_thresh, shared_intrinsics, mask_path, **kw):
    """
    from a list of images, run mast3r inference, sparse global aligner.
    then run get_3D_model_from_scene
    """
    imgs = load_images(filelist, size=image_size, verbose=not silent)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
        filelist = [filelist[0], filelist[0] + '_2']

    scene_graph_params = [scenegraph_type]
    if scenegraph_type in ["swin", "logwin"]:
        scene_graph_params.append(str(winsize))
    elif scenegraph_type == "oneref":
        scene_graph_params.append(str(refid))
    elif scenegraph_type == "retrieval":
        scene_graph_params.append(str(winsize))  # Na
        scene_graph_params.append(str(refid))  # k

    if scenegraph_type in ["swin", "logwin"] and not win_cyclic:
        scene_graph_params.append('noncyclic')
    scene_graph = '-'.join(scene_graph_params)

    sim_matrix = None
    if 'retrieval' in scenegraph_type:
        assert retrieval_model is not None
        retriever = Retriever(retrieval_model, backbone=model, device=device)
        with torch.no_grad():
            sim_matrix = retriever(filelist)

        # Cleanup
        del retriever
        torch.cuda.empty_cache()

    pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True, sim_mat=sim_matrix)
    if optim_level == 'coarse':
        niter2 = 0
    # Sparse GA (forward mast3r -> matching -> 3D optim -> 2D refinement -> triangulation)
    if current_scene_state is not None and \
        not current_scene_state.should_delete and \
            current_scene_state.cache_dir is not None:
        cache_dir = current_scene_state.cache_dir
    elif gradio_delete_cache:
        cache_dir = tempfile.mkdtemp(suffix='_cache', dir=outdir)
    else:
        cache_dir = os.path.join(outdir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    scene = sparse_global_alignment(filelist, pairs, cache_dir,
                                    model, lr1=lr1, niter1=niter1, lr2=lr2, niter2=niter2, device=device,
                                    opt_depth='depth' in optim_level, shared_intrinsics=shared_intrinsics,
                                    matching_conf_thr=matching_conf_thr, **kw)
    if current_scene_state is not None and \
        not current_scene_state.should_delete and \
            current_scene_state.outfile_name is not None:
        outfile_name = current_scene_state.outfile_name
    else:
        outfile_name = tempfile.mktemp(suffix='_scene.glb', dir=outdir)

    scene_state = SparseGAState(scene, gradio_delete_cache, cache_dir, outfile_name)

    print(scene_state.sparse_ga.depthmaps[0].cpu().numpy())
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio 

    # Dynamically reshape depthmap to (N, 64) where N is inferred
    depthmap = scene_state.sparse_ga.depthmaps[0]
    width = 64
    total = depthmap.numel()
    height = total // width
    if total % width != 0:
        raise RuntimeError(f"Cannot reshape depthmap of size {total} to (?, {width})")
    depth_image = depthmap.reshape(height, width).cpu().numpy()
    # Example: for 16:9 use 36, for 4:3 use 48

    # Normalize depth to 0-65535 for 16-bit PNG
    depth_min = np.nanmin(depth_image)
    depth_max = np.nanmax(depth_image)
    depth_norm = (depth_image - depth_min) / (depth_max - depth_min + 1e-8)
    depth_uint16 = (depth_norm * 65535).astype(np.uint16)

    imageio.imwrite(outdir + '/depth_image_16bit.png', depth_uint16)
    # Save depth in millimeters as uint16 PNG
    depth_mm = (depth_image * 1000).astype(np.uint16)  # Convert meters to mm if needed
    print(depth_mm)
    imageio.imwrite(outdir + '/depth_mm_16bit.png', depth_mm)
    print("16-bit depth image saved as 'depth_image_16bit.png'")

    plt.figure(figsize=(8, 8))
    plt.imshow(depth_image, cmap='viridis')
    plt.colorbar(label='Depth')
    plt.title('Depth Image')
    plt.savefig(outdir + '/depth_image.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("Depth image saved as 'depth_image.png'")


    outfile, height = get_3D_model_from_scene(silent, scene_state, min_conf_thr, as_pointcloud, mask_sky,
                                      clean_depth, transparent_cams, cam_size, TSDF_thresh, mask_path=mask_path,outdir=outdir)
    return scene_state, outfile, height


def set_scenegraph_options(inputfiles, win_cyclic, refid, scenegraph_type):
    num_files = len(inputfiles) if inputfiles is not None else 1
    max_winsize, min_winsize = 1, 1

    winsize = gradio.Slider(visible=False)
    win_cyclic = gradio.Checkbox(visible=False)
    graph_opt = gradio.Column(visible=False)
    refid = gradio.Slider(visible=False)

    if scenegraph_type in ["swin", "logwin"]:
        if scenegraph_type == "swin":
            if win_cyclic:
                max_winsize = max(1, math.ceil((num_files - 1) / 2))
            else:
                max_winsize = num_files - 1
        else:
            if win_cyclic:
                half_size = math.ceil((num_files - 1) / 2)
                max_winsize = max(1, math.ceil(math.log(half_size, 2)))
            else:
                max_winsize = max(1, math.ceil(math.log(num_files, 2)))

        winsize = gradio.Slider(label="Scene Graph: Window Size", value=max_winsize,
                                minimum=min_winsize, maximum=max_winsize, step=1, visible=True)
        win_cyclic = gradio.Checkbox(value=win_cyclic, label="Cyclic sequence", visible=True)
        graph_opt = gradio.Column(visible=True)
        refid = gradio.Slider(visible=False)

    elif scenegraph_type == "retrieval":
        graph_opt = gradio.Column(visible=True)
        winsize = gradio.Slider(label="Retrieval: Num. key images", value=min(20, num_files),
                                minimum=0, maximum=num_files, step=1, visible=True)
        win_cyclic = gradio.Checkbox(visible=False)
        refid = gradio.Slider(label="Retrieval: Num neighbors", value=min(num_files - 1, 10), minimum=1,
                              maximum=num_files - 1, step=1, visible=True)

    elif scenegraph_type == "oneref":
        graph_opt = gradio.Column(visible=True)
        winsize = gradio.Slider(visible=False)
        win_cyclic = gradio.Checkbox(visible=False)
        refid = gradio.Slider(label="Scene Graph: Id", value=0, minimum=0,
                              maximum=num_files - 1, step=1, visible=True)

    return graph_opt, winsize, win_cyclic, refid


def main_demo_automated(input_folder, output_dir, mask_path, model, retrieval_model=None, device='cuda', 
                       image_size=512, silent=False, gradio_delete_cache=False,
                       # Optimization parameters
                       lr1=0.07, niter1=300, lr2=0.01, niter2=300, optim_level='refine+depth',
                       # Matching parameters
                       matching_conf_thr=0.0, shared_intrinsics=False,
                       # Scene graph parameters
                       scenegraph_type='complete', winsize=1, win_cyclic=False, refid=0,
                       # Output parameters
                       min_conf_thr=1.5, as_pointcloud=True, mask_sky=False, clean_depth=True,
                       transparent_cams=False, cam_size=0.2, TSDF_thresh=0.0):
    """
    Automated version of main_demo without Gradio interface.
    
    Args:
        input_folder: Path to folder containing input images
        output_dir: Directory to save output files
        model: Pre-loaded MASt3R model
        retrieval_model: Pre-loaded retrieval model (optional)
        device: Device to run on ('cuda' or 'cpu')
        image_size: Size to resize images to
        silent: Whether to suppress output
        gradio_delete_cache: Whether to delete cache after processing
        
        # Optimization parameters
        lr1: Learning rate for coarse alignment
        niter1: Number of iterations for coarse alignment
        lr2: Learning rate for refinement
        niter2: Number of iterations for refinement
        optim_level: Optimization level ('coarse', 'refine', 'refine+depth')
        
        # Matching parameters
        matching_conf_thr: Matching confidence threshold
        shared_intrinsics: Whether to use shared intrinsics
        
        # Scene graph parameters
        scenegraph_type: Type of scene graph ('complete', 'swin', 'logwin', 'oneref', 'retrieval')
        winsize: Window size for scene graph
        win_cyclic: Whether to use cyclic window
        refid: Reference ID for scene graph
        
        # Output parameters
        min_conf_thr: Minimum confidence threshold for output
        as_pointcloud: Whether to output as point cloud
        mask_sky: Whether to mask sky
        clean_depth: Whether to clean depth maps
        transparent_cams: Whether to use transparent cameras
        cam_size: Camera size in output
        TSDF_thresh: TSDF threshold
        
    Returns:
        tuple: (scene_state, output_file_path)
    """
    import os
    import glob
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files from input folder
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    filelist = []
    for ext in image_extensions:
        filelist.extend(glob.glob(os.path.join(input_folder, ext)))
        filelist.extend(glob.glob(os.path.join(input_folder, ext.upper())))
    
    filelist.sort()  # Sort for consistent ordering
    
    if not filelist:
        raise ValueError(f"No image files found in {input_folder}")
    
    if not silent:
        print(f'Found {len(filelist)} images in {input_folder}')
        print('Processing images:', [os.path.basename(f) for f in filelist])
    
    # Use the existing reconstruction function
    recon_fun = functools.partial(get_reconstructed_scene, output_dir, gradio_delete_cache, model,
                                  retrieval_model, device, silent, image_size)
    
    # Call the reconstruction function with all parameters
    scene_state, outfile, height = recon_fun(
        current_scene_state=None,
        filelist=filelist,
        optim_level=optim_level,
        lr1=lr1,
        niter1=niter1,
        lr2=lr2,
        niter2=niter2,
        min_conf_thr=min_conf_thr,
        matching_conf_thr=matching_conf_thr,
        as_pointcloud=as_pointcloud,
        mask_sky=mask_sky,
        clean_depth=clean_depth,
        transparent_cams=transparent_cams,
        cam_size=cam_size,
        scenegraph_type=scenegraph_type,
        winsize=winsize,
        win_cyclic=win_cyclic,
        refid=refid,
        TSDF_thresh=TSDF_thresh,
        shared_intrinsics=shared_intrinsics,
        mask_path=mask_path
    )
    
    if not silent:
        print(f'3D model saved to: {outfile}')

    return scene_state, outfile, height


def main_demo(tmpdirname, model, retrieval_model, device, image_size, server_name, server_port, silent=False,
              share=False, gradio_delete_cache=False):
    if not silent:
        print('Outputing stuff in', tmpdirname)

    recon_fun = functools.partial(get_reconstructed_scene, tmpdirname, gradio_delete_cache, model,
                                  retrieval_model, device, silent, image_size)
    model_from_scene_fun = functools.partial(get_3D_model_from_scene, silent)

    available_scenegraph_type = [("complete: all possible image pairs", "complete"),
                                 ("swin: sliding window", "swin"),
                                 ("logwin: sliding window with long range", "logwin"),
                                 ("oneref: match one image with all", "oneref")]
    if retrieval_model is not None:
        available_scenegraph_type.insert(1, ("retrieval: connect views based on similarity", "retrieval"))

    def get_context(delete_cache):
        css = """.gradio-container {margin: 0 !important; min-width: 100%};"""
        title = "MASt3R Demo"
        if delete_cache:
            return gradio.Blocks(css=css, title=title, delete_cache=(delete_cache, delete_cache))
        else:
            return gradio.Blocks(css=css, title="MASt3R Demo")  # for compatibility with older versions

    with get_context(gradio_delete_cache) as demo:
        # scene state is save so that you can change conf_thr, cam_size... without rerunning the inference
        scene = gradio.State(None)
        gradio.HTML('<h2 style="text-align: center;">MASt3R Demo</h2>')
        with gradio.Column():
            inputfiles = gradio.File(file_count="multiple")
            with gradio.Row():
                with gradio.Column():
                    with gradio.Row():
                        lr1 = gradio.Slider(label="Coarse LR", value=0.07, minimum=0.01, maximum=0.2, step=0.01)
                        niter1 = gradio.Slider(value=300, minimum=0, maximum=1000, step=1,
                                               label="Iterations", info="For coarse alignment")
                        lr2 = gradio.Slider(label="Fine LR", value=0.01, minimum=0.005, maximum=0.05, step=0.001)
                        niter2 = gradio.Slider(value=300, minimum=0, maximum=1000, step=1,
                                               label="Iterations", info="For refinement")
                        optim_level = gradio.Dropdown(["coarse", "refine", "refine+depth"],
                                                      value='refine+depth', label="OptLevel",
                                                      info="Optimization level")
                    with gradio.Row():
                        matching_conf_thr = gradio.Slider(label="Matching Confidence Thr", value=0.,
                                                          minimum=0., maximum=30., step=0.1,
                                                          info="Before Fallback to Regr3D!")
                        shared_intrinsics = gradio.Checkbox(value=False, label="Shared intrinsics",
                                                            info="Only optimize one set of intrinsics for all views")
                        scenegraph_type = gradio.Dropdown(available_scenegraph_type,
                                                          value='complete', label="Scenegraph",
                                                          info="Define how to make pairs",
                                                          interactive=True)
                        with gradio.Column(visible=False) as graph_opt:
                            winsize = gradio.Slider(label="Scene Graph: Window Size", value=1,
                                                    minimum=1, maximum=1, step=1)
                            win_cyclic = gradio.Checkbox(value=False, label="Cyclic sequence")
                            refid = gradio.Slider(label="Scene Graph: Id", value=0,
                                                  minimum=0, maximum=0, step=1, visible=False)

            run_btn = gradio.Button("Run")

            with gradio.Row():
                # adjust the confidence threshold
                min_conf_thr = gradio.Slider(label="min_conf_thr", value=1.5, minimum=0.0, maximum=10, step=0.1)
                # adjust the camera size in the output pointcloud
                cam_size = gradio.Slider(label="cam_size", value=0.2, minimum=0.001, maximum=1.0, step=0.001)
                TSDF_thresh = gradio.Slider(label="TSDF Threshold", value=0., minimum=0., maximum=1., step=0.01)
            with gradio.Row():
                as_pointcloud = gradio.Checkbox(value=True, label="As pointcloud")
                # two post process implemented
                mask_sky = gradio.Checkbox(value=False, label="Mask sky")
                clean_depth = gradio.Checkbox(value=True, label="Clean-up depthmaps")
                transparent_cams = gradio.Checkbox(value=False, label="Transparent cameras")

            outmodel = gradio.Model3D()

            # events
            scenegraph_type.change(set_scenegraph_options,
                                   inputs=[inputfiles, win_cyclic, refid, scenegraph_type],
                                   outputs=[graph_opt, winsize, win_cyclic, refid])
            inputfiles.change(set_scenegraph_options,
                              inputs=[inputfiles, win_cyclic, refid, scenegraph_type],
                              outputs=[graph_opt, winsize, win_cyclic, refid])
            win_cyclic.change(set_scenegraph_options,
                              inputs=[inputfiles, win_cyclic, refid, scenegraph_type],
                              outputs=[graph_opt, winsize, win_cyclic, refid])
            run_btn.click(fn=recon_fun,
                          inputs=[scene, inputfiles, optim_level, lr1, niter1, lr2, niter2, min_conf_thr, matching_conf_thr,
                                  as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size,
                                  scenegraph_type, winsize, win_cyclic, refid, TSDF_thresh, shared_intrinsics],
                          outputs=[scene, outmodel])
            min_conf_thr.release(fn=model_from_scene_fun,
                                 inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                         clean_depth, transparent_cams, cam_size, TSDF_thresh],
                                 outputs=outmodel)
            cam_size.change(fn=model_from_scene_fun,
                            inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                    clean_depth, transparent_cams, cam_size, TSDF_thresh],
                            outputs=outmodel)
            TSDF_thresh.change(fn=model_from_scene_fun,
                               inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                       clean_depth, transparent_cams, cam_size, TSDF_thresh],
                               outputs=outmodel)
            as_pointcloud.change(fn=model_from_scene_fun,
                                 inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                         clean_depth, transparent_cams, cam_size, TSDF_thresh],
                                 outputs=outmodel)
            mask_sky.change(fn=model_from_scene_fun,
                            inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                    clean_depth, transparent_cams, cam_size, TSDF_thresh],
                            outputs=outmodel)
            clean_depth.change(fn=model_from_scene_fun,
                               inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                       clean_depth, transparent_cams, cam_size, TSDF_thresh],
                               outputs=outmodel)
            transparent_cams.change(model_from_scene_fun,
                                    inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                            clean_depth, transparent_cams, cam_size, TSDF_thresh],
                                    outputs=outmodel)
    demo.launch(share=share, server_name=server_name, server_port=server_port)


import imageio.v2 as imageio
import numpy as np

def get_pts_seen_by_cam0_with_mask(
    pts3d,
    cams2world,
    focals,
    img_shape,
    mask_path,
    depth_path=None,
    depth_tolerance_cm=10.0
):
    import imageio
    import numpy as np
    from scipy.ndimage import zoom
    
    # Load and resize mask
    mask_img = imageio.imread(mask_path)
    if mask_img.ndim == 3:
        mask_img = mask_img[..., 0]
    scale_y = img_shape[0] / mask_img.shape[0]
    scale_x = img_shape[1] / mask_img.shape[1]
    mask_resized = zoom(mask_img, (scale_y, scale_x), order=0)
    mask = mask_resized > 0
    
    # Load and resize depth image if provided
    depth_image = None
    mean_depth_in_mask = None
    if depth_path is not None:
        depth_img = imageio.imread(depth_path)
        depth_scale_y = img_shape[0] / depth_img.shape[0]
        depth_scale_x = img_shape[1] / depth_img.shape[1]
        depth_resized = zoom(depth_img.astype(np.float32),
                             (depth_scale_y, depth_scale_x),
                             order=1)
        depth_image = depth_resized
        
        # Calculate mean depth within the mask
        masked_depth = depth_image[mask]
        valid_depth = masked_depth[masked_depth > 0]  # Remove zero/invalid depths
        if len(valid_depth) > 0:
            mean_depth_in_mask = np.mean(valid_depth)
            print(f"Mean depth within mask: {mean_depth_in_mask:.2f} mm")
            print(f"Min depth within mask: {np.min(valid_depth):.2f} mm")
            print(f"Max depth within mask: {np.max(valid_depth):.2f} mm")
            print(f"Valid depth pixels in mask: {len(valid_depth)}")
        else:
            print("No valid depth values found within the mask")
    
    # Convert camera matrices to numpy if needed
    if hasattr(cams2world, 'detach'):
        cams2world = cams2world.detach().cpu().numpy()
    if hasattr(focals, 'detach'):
        focals = focals.detach().cpu().numpy()
    
    cam0_pose = cams2world[0]
    world2cam0 = np.linalg.inv(cam0_pose)
    
    try:
        fx, fy = focals[0]
    except Exception:
        fx = fy = float(focals[0])
    
    cx, cy = img_shape[1] / 2, img_shape[0] / 2
    seen_points = []
    count = 0
    for pts in pts3d:
        count += 1
        if count == 2:
            break
        # Convert to numpy if tensor
        if hasattr(pts, 'detach'):
            pts_flat = pts.detach().cpu().numpy().reshape(-1, 3)
        else:
            pts_flat = pts.reshape(-1, 3)
        
        pts_h = np.concatenate([pts_flat, np.ones((pts_flat.shape[0], 1))], axis=1)
        
        # Transform to camera coordinates
        pts_cam = (world2cam0 @ pts_h.T).T[:, :3]
        
        # Keep only points in front of camera
        in_front = pts_cam[:, 2] > 0
        if not np.any(in_front):
            continue
            
        pts_cam_front = pts_cam[in_front]
        pts_flat_front = pts_flat[in_front]
        
        # Project to image plane
        u = fx * (pts_cam_front[:, 0] / pts_cam_front[:, 2]) + cx
        v = fy * (pts_cam_front[:, 1] / pts_cam_front[:, 2]) + cy
        
        # Pixel coordinates
        u_int = np.round(u).astype(int)
        v_int = np.round(v).astype(int)
        
        # Check image bounds
        inside = ((u_int >= 0) & (u_int < img_shape[1]) &
                  (v_int >= 0) & (v_int < img_shape[0]))
        
        if not np.any(inside):
            continue
        
        # Create boolean arrays for all filters - initialize all as False
        final_filter = np.zeros(len(pts_cam_front), dtype=bool)
        
        # Only process points that are inside image bounds
        inside_indices = np.where(inside)[0]
        u_inside = u_int[inside_indices]
        v_inside = v_int[inside_indices]
        
        # Apply mask filter to inside points
        mask_valid = mask[v_inside, u_inside]

        if depth_image is not None and mean_depth_in_mask is not None:
            # Use mean depth within mask + tolerance for filtering
            cam_depths_mm = pts_cam_front[inside_indices, 2] * 1000.0
            tol_mm = depth_tolerance_cm * 10.0  # Convert cm to mm
            
            # Filter points based on mean depth + tolerance
            depth_diff = np.abs(cam_depths_mm - mean_depth_in_mask) - (mean_depth_in_mask - np.min(valid_depth))/2
            depth_valid = depth_diff <= tol_mm

            # Combine mask and depth filters
            combined_valid = mask_valid & depth_valid

            print(f"Points inside image: {len(inside_indices)}")
            print(f"Points passing mask: {np.sum(mask_valid)}")
            print(f"Mean depth used for filtering: {mean_depth_in_mask:.2f} mm")
            print(f"Tolerance: {tol_mm:.2f} mm")
            print(f"Points passing depth filter (mean±tolerance): {np.sum(depth_valid)}")
            print(f"Points passing both filters: {np.sum(combined_valid)}")
            
        else:
            combined_valid = mask_valid
            print(f"Points inside image: {len(inside_indices)}")
            print(f"Points passing mask: {np.sum(mask_valid)}")
        
        # Set the final filter: only inside points that pass combined filter
        if np.any(combined_valid):
            valid_inside_indices = inside_indices[combined_valid]
            final_filter[valid_inside_indices] = True
            
            # Get the final points (keep in camera coordinates for this view)
            seen_points.append(pts_cam_front[final_filter]) #world coordinates
    all_points = np.concatenate(seen_points, axis=0) if seen_points else np.zeros((0, 3))
    if all_points.shape[0] > 0:
        mean_z = np.mean(all_points[:, 2])
        print(f"Mean z value of all points: {mean_z:.4f}")
        # Keep only points within mean_z ± 3cm (0.03m)
        z_min = mean_z - 0.03
        z_max = mean_z + 0.03
        in_range = (all_points[:, 2] >= z_min) & (all_points[:, 2] <= z_max)
        filtered_points = all_points[in_range]
        print(f"Points within ±3cm of mean z: {filtered_points.shape[0]}")
    else:
        filtered_points = np.zeros((0, 3))
        print("No points found to calculate mean z value.")

    if filtered_points.shape[0] > 0:
        min_y = np.min(filtered_points[:, 1])
        max_y = np.max(filtered_points[:, 1])
        height = max_y - min_y
        print(f"Bounding box in y: min={min_y}, max={max_y}, height={height}")
    else:
        height = None
        print("No points to compute bounding box.")

    return filtered_points, height


import trimesh

def save_points_as_ply(pts3d, filename="points.ply"):
    """
    Save a list of 3D points (numpy array or list of arrays) as a PLY file.
    """
    if isinstance(pts3d, list):
        pts = np.concatenate([p.reshape(-1, 3) for p in pts3d], axis=0)
    else:
        pts = pts3d.reshape(-1, 3)
    # Remove NaNs or infs
    valid = np.isfinite(pts).all(axis=1)
    pts = pts[valid]
    cloud = trimesh.PointCloud(pts)
    cloud.export(filename)
    print(f"Saved {pts.shape[0]} points to {filename}")
