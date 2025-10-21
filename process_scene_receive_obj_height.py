#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# gradio demo executable
# --------------------------------------------------------
import os
from mast3r.demo_scale import main_demo_automated
import torch
import tempfile
from contextlib import nullcontext

from mast3r.demo_scale import get_args_parser, main_demo

from mast3r.model import AsymmetricMASt3R
from mast3r.utils.misc import hash_md5

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.demo import set_print_with_timestamp
import json

import matplotlib.pyplot as pl
pl.ion()

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

if __name__ == '__main__':
    parser = get_args_parser()
    parser.add_argument(
        "--object_path",
        type=str,
        default="/home/st3fan/Projects/Grounded-SAM-2/dataset/ycbv_real_subset/obj_000008",
        help="Path to the object directory"
    )
    args = parser.parse_args()
    set_print_with_timestamp()

    object_path = args.object_path

    if args.server_name is not None:
        server_name = args.server_name
    else:
        server_name = '0.0.0.0' if args.local_network else '127.0.0.1'

    if args.weights is not None:
        weights_path = args.weights
    else:
        weights_path = "naver/" + args.model_name

    model = AsymmetricMASt3R.from_pretrained(weights_path).to(args.device)
    chkpt_tag = hash_md5(weights_path)

    def get_context(tmp_dir):
        return tempfile.TemporaryDirectory(suffix='_mast3r_gradio_demo') if tmp_dir is None \
            else nullcontext(tmp_dir)
    with get_context(args.tmp_dir) as tmpdirname:
        cache_path = os.path.join(tmpdirname, chkpt_tag)
        os.makedirs(cache_path, exist_ok=True)
        #main_demo(cache_path, model, args.retrieval_model, args.device, args.image_size, server_name, args.server_port,
        #          silent=args.silent, share=args.share, gradio_delete_cache=args.gradio_delete_cache)
        # automated demo call
        scene_state, output_file, height = main_demo_automated(
            input_folder=object_path + "/scene/images",
            output_dir=object_path + "/scene/output",
            mask_path=object_path + "/train_pbr/000000/mask/000000_000000.png",
            model=model,
            retrieval_model=None,  # optional
            device=args.device,
            image_size=512,
            lr1=0.07,
            niter1=300,
            lr2=0.01,
            niter2=500,
            optim_level='refine+depth',
            scenegraph_type='complete',
            as_pointcloud=True,
            clean_depth=False,
            cam_size=0.2,
            min_conf_thr=1.0,
            shared_intrinsics=False # if scene images matches object images this improves accuracy
        )

        print(f"Estimated Object height: {height}")
        object_info = {"estimated_height": height}
        output_json_path = os.path.join(object_path, "scene", "output", "object_info.json")
        with open(output_json_path, "w") as f:
            json.dump(object_info, f, indent=4)
        print(f"Saved estimated height to {output_json_path}")
