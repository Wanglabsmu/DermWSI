import openslide
import h5py
import numpy as np
import glob
import os
import cv2
import time
import argparse
import multiprocessing as mp


def artifacts_removal(coord, slide_name, patch_size):
    """
    Remove the patch if the white area is larger than 90 percent
    Input:
        coord (np.array): The coordinate of patche in the slide
        slide_name (str): The slide to process
        patch_size (int): The patch size used to patch the slide
    Output:
        (bool): 1: The patch is white, otherwise, 0
    """
    wsi = openslide.open_slide(slide_name)
    region = wsi.read_region(coord, 0, (patch_size, patch_size)).convert("L").resize((256, 256))
    _, white_region = cv2.threshold(np.array(region), 235, 255, cv2.THRESH_BINARY)
    if np.sum(white_region == 255) / (256 * 256) > 0.9:
        return 1
    else:
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--slide", required=True)
    parser.add_argument('--resolution', type=str, default='40x')
    parser.add_argument("--mosaic_path", required=True)
    args = parser.parse_args()
    pool = mp.Pool(4)
    # total = len(glob.glob(os.path.join(args.mosaic_path, "*", "*",
    #             "coord", "*")))
    total = len(glob.glob(os.path.join(args.mosaic_path, "coord", "*")))

    progress = 0
##./DATA/MOSAICS/xiangya_20250412/Pso/40x/coord/66531.h5
    # for mosaic in glob.glob(os.path.join(args.mosaic_path,
    #                         "*", "*", "coord", "*")):
    for slide in [args.slide]:
        slide_id = os.path.basename(slide).replace(".ndpi", "").replace("svs","")
        mosaic = os.path.join(args.mosaic_path, "coord", slide_id+".h5")
        # print(mosaic)
    
        with h5py.File(mosaic, 'r') as hf:
            coords = hf['coords'][:]
        if args.resolution == '20x':
            patch_size = 1024
        elif args.resolution == '40x':
            patch_size = 2048
        # print("Original mosaic size:", len(coords))

        # Remove the white artifacts
        t_start = time.time()
        iterable = [(coord, args.slide, patch_size) for coord in coords]
        artifacts_indicator = pool.starmap(artifacts_removal, iterable)
        coord_clean = coords[np.array(artifacts_indicator) == 0]
        coord_artifacts = coords[np.array(artifacts_indicator) == 1]
        # print("Clean mosaic size:", len(coord_clean))
        # print("Removal takes: ", time.time() - t_start)

        save_path_clean = os.path.join(args.mosaic_path, "coord_clean")
        save_path_artifacts = os.path.join(args.mosaic_path, "coord_artifacts")
        # Save the results
        if not os.path.exists(save_path_clean):
            os.makedirs(save_path_clean)
        if not os.path.exists(save_path_artifacts):
            os.makedirs(save_path_artifacts)

        with h5py.File(os.path.join(save_path_clean, slide_id + ".h5"), 'w') as hf:
            hf.create_dataset("coords", data=coord_clean)
        if len(coord_clean) == len(coords):
            # print("")
            progress += 1
            continue
        else:
            with h5py.File(os.path.join(save_path_artifacts, slide_id + ".h5"), 'w') as hf:
                hf.create_dataset("coords", data=coord_artifacts)
            progress += 1
            # print("")
