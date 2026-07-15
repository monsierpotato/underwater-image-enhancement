import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import argparse
from multiprocessing import Pool
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

def process_image(args):
    img_path, prior_method, img_size = args
    parent_dir = os.path.dirname(os.path.dirname(img_path))
    cache_dir = os.path.join(parent_dir, f"physics_cache_{prior_method}_{img_size}")
    
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except OSError:
        import hashlib
        path_hash = hashlib.md5(parent_dir.encode('utf-8')).hexdigest()
        cache_dir = os.path.join(os.getcwd(), "physics_cache", f"{prior_method}_{img_size}", path_hash)
        os.makedirs(cache_dir, exist_ok=True)
        
    stem = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(cache_dir, f"{stem}.npz")
    
    if os.path.exists(out_path):
        return
        
    try:
        from uwir.physics import (
            compute_physics_maps,
            compute_physics_maps_gdcp,
            compute_physics_maps_gupdm,
        )
        img = Image.open(img_path).convert("RGB")
        img = TF.resize(img, (img_size, img_size))
        img_t = TF.to_tensor(img)
        img_np = img_t.permute(1, 2, 0).numpy().astype(np.float32)
        
        if prior_method == "gupdm":
            t, b = compute_physics_maps_gupdm(img_np)
        elif prior_method == "udcp":
            t, b = compute_physics_maps(img_np)
        elif prior_method == "gdcp":
            t, b = compute_physics_maps_gdcp(img_np)
        else:
            raise ValueError(f"Unknown prior: {prior_method}")
            
        np.savez(out_path, t=t.astype(np.float32), b=b.astype(np.float32))
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute physics maps for datasets.")
    parser.add_argument("--data_euvp", type=str, default="", help="Path to EUVP root")
    parser.add_argument("--data_uieb", type=str, default="", help="Path to UIEB root")
    parser.add_argument("--prior_method", type=str, default="gupdm", choices=["gupdm", "udcp", "gdcp"])
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()
    
    files_to_process = []
    
    if args.data_euvp and os.path.exists(args.data_euvp):
        for s in ["underwater_imagenet", "underwater_dark", "underwater_scenes"]:
            input_dir = os.path.join(args.data_euvp, "Paired", s, "trainA")
            if os.path.isdir(input_dir):
                for f in os.listdir(input_dir):
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        files_to_process.append(os.path.join(input_dir, f))
                        
    if args.data_uieb and os.path.exists(args.data_uieb):
        input_dir = os.path.join(args.data_uieb, "raw-890")
        if os.path.isdir(input_dir):
            for f in os.listdir(input_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    files_to_process.append(os.path.join(input_dir, f))
                    
    print(f"Found {len(files_to_process)} images to process.")
    if len(files_to_process) == 0:
        print("Please provide --data_euvp or --data_uieb.")
        exit(0)
        
    tasks = [(f, args.prior_method, args.img_size) for f in files_to_process]
    
    print(f"Using {args.workers} workers to precompute {args.prior_method} maps...")
    with Pool(args.workers) as p:
        list(tqdm(p.imap_unordered(process_image, tasks), total=len(tasks)))
        
    print("Done!")
