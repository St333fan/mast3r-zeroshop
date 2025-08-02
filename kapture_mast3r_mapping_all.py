#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# mast3r exec for running standard sfm
# --------------------------------------------------------
import pycolmap
import os
import os.path as path
import argparse

from mast3r.model import AsymmetricMASt3R
from mast3r.colmap.mapping import (kapture_import_image_folder_or_list, run_mast3r_matching, pycolmap_run_triangulator,
                                   pycolmap_run_mapper, glomap_run_mapper)
from kapture.io.csv import kapture_from_dir

from kapture.converter.colmap.database_extra import kapture_to_colmap, generate_priors_for_reconstruction
from kapture_localization.utils.pairsfile import get_pairs_from_file
from kapture.io.records import get_image_fullpath
from kapture.converter.colmap.database import COLMAPDatabase


def get_argparser():
    parser = argparse.ArgumentParser(description='point triangulator with mast3r from kapture data')
    parser_weights = parser.add_mutually_exclusive_group(required=True)
    parser_weights.add_argument("--weights", type=str, help="path to the model weights", default=None)
    parser_weights.add_argument("--model_name", type=str, help="name of the model weights",
                                choices=["MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"])

    parser.add_argument('--path', required=True, help='base path containing train_phr/mast3r-sfm subdirectories')
    parser.add_argument('--use_single_camera', action='store_true', default=False, help='use shared intrinsics for all images')

    parser.add_argument('--glomap_bin', default='glomap', type=str, help='glomap bin')

    parser_mapper = parser.add_mutually_exclusive_group()
    parser_mapper.add_argument('--ignore_pose', action='store_true', default=False)
    parser_mapper.add_argument('--use_glomap_mapper', action='store_true', default=False)

    parser_matching = parser.add_mutually_exclusive_group()
    parser_matching.add_argument('--dense_matching', action='store_true', default=False)
    parser_matching.add_argument('--pixel_tol', default=0, type=int)
    parser.add_argument('--device', default='cuda')

    parser.add_argument('--conf_thr', default=1.001, type=float)
    parser.add_argument('--skip_geometric_verification', action='store_true', default=False)
    parser.add_argument('--min_len_track', default=5, type=int)

    return parser


def process_single_directory(subdir, base_path, model, maxdim, patch_size, args):
    """Process a single directory (surface or segmented)"""
    images_dir = path.join(base_path, 'train_pbr', 'mast3r-sfm', subdir, 'images')
    output_dir = path.join(base_path, 'train_phr', 'mast3r-sfm', subdir)
    pairsfile_path = path.join(images_dir, 'pairs.txt')
    
    # Set different conf_thr values for each directory
    if subdir == 'surface':
        conf_thr = 1.0
    elif subdir == 'segmented':
        conf_thr = 0.01
    else:
        conf_thr = args.conf_thr  # fallback to default
    
    if not path.exists(images_dir):
        print(f"Warning: Directory {images_dir} does not exist, skipping...")
        return False
        
    if not path.exists(pairsfile_path):
        print(f"Warning: Pairs file {pairsfile_path} does not exist, skipping...")
        return False
    
    print(f"Processing {subdir} directory: {images_dir} with conf_thr: {conf_thr}")
    
    # Import image folder
    kdata = kapture_import_image_folder_or_list(images_dir, args.use_single_camera)
    has_pose = kdata.trajectories is not None
    image_pairs = get_pairs_from_file(pairsfile_path, kdata.records_camera, kdata.records_camera)

    colmap_db_path = path.join(output_dir, 'colmap.db')
    reconstruction_path = path.join(output_dir, "reconstruction")
    priors_txt_path = path.join(output_dir, "priors_for_reconstruction")
    
    for path_i in [reconstruction_path, priors_txt_path]:
        os.makedirs(path_i, exist_ok=True)
    
    # Remove existing database if it exists
    if os.path.isfile(colmap_db_path):
        os.remove(colmap_db_path)

    colmap_db = COLMAPDatabase.connect(colmap_db_path)
    try:
        kapture_to_colmap(kdata, None, tar_handler=None, database=colmap_db,
                          keypoints_type=None, descriptors_type=None, export_two_view_geometry=False)
        if has_pose:
            generate_priors_for_reconstruction(kdata, colmap_db, priors_txt_path)

        colmap_image_pairs = run_mast3r_matching(model, maxdim, patch_size, args.device,
                                                 kdata, images_dir, image_pairs, colmap_db,
                                                 args.dense_matching, args.pixel_tol, conf_thr,
                                                 args.skip_geometric_verification, args.min_len_track)
        colmap_db.close()
    except Exception as e:
        print(f'Error processing {subdir}: {e}')
        colmap_db.close()
        return False

    if len(colmap_image_pairs) == 0:
        print(f"Warning: no matches were kept for {subdir}")
        return False

    # colmap db is now full, run colmap
    if not args.skip_geometric_verification:
        print(f"verify_matches for {subdir}")
        pairs_txt_path = path.join(output_dir, 'pairs.txt')
        with open(pairs_txt_path, "w") as f:
            for image_path1, image_path2 in colmap_image_pairs:
                f.write("{} {}\n".format(image_path1, image_path2))
        pycolmap.verify_matches(colmap_db_path, pairs_txt_path)

    print(f"running mapping for {subdir}")
    if has_pose and not args.ignore_pose and not args.use_glomap_mapper:
        pycolmap_run_triangulator(colmap_db_path, priors_txt_path, reconstruction_path, images_dir)
    elif not args.use_glomap_mapper:
        pycolmap_run_mapper(colmap_db_path, reconstruction_path, images_dir)
    else:
        glomap_run_mapper(args.glomap_bin, colmap_db_path, reconstruction_path, images_dir)
    
    print(f"Completed processing {subdir}: output in {output_dir}")
    return True


def process_directories(base_path, model, maxdim, patch_size, args):
    """Process both surface and segmented directories"""
    subdirs = ['surface', 'segmented']
    success_count = 0
    
    for subdir in subdirs:
        if process_single_directory(subdir, base_path, model, maxdim, patch_size, args):
            success_count += 1
    
    print(f"Successfully processed {success_count}/{len(subdirs)} directories")


if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()
    if args.weights is not None:
        weights_path = args.weights
    else:
        weights_path = "naver/" + args.model_name
    model = AsymmetricMASt3R.from_pretrained(weights_path).to(args.device)
    maxdim = max(model.patch_embed.img_size)
    patch_size = model.patch_embed.patch_size

    process_directories(args.path, model, maxdim, patch_size, args)
