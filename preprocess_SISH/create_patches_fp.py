# internal imports
from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.wsi_utils import StitchCoords
# other imports
import os
import  sys
os.chdir(sys.path[0])
import numpy as np
import time
import argparse
import pandas as pd


def stitching(file_path, wsi_object, downscale=64):
    start = time.time()
    heatmap = StitchCoords(file_path, wsi_object, downscale=downscale, bg_color=(0,0,0), alpha=-1, draw_grid=False)
    total_time = time.time() - start

    return heatmap, total_time


def segment(WSI_object, seg_params, filter_params):
    # Start Seg Timer
    start_time = time.time()

    # Segment
    WSI_object.segmentTissue(**seg_params, filter_params=filter_params)

    # Stop Seg Timers
    seg_time_elapsed = time.time() - start_time
    return WSI_object, seg_time_elapsed


def patching(WSI_object, **kwargs):
    # Start Patch Timer
    start_time = time.time()

    # Patch
    file_path = WSI_object.process_contours(**kwargs)

    # Stop Patch Timer
    patch_time_elapsed = time.time() - start_time
    return file_path, patch_time_elapsed


def initialize_df(slides, seg_params, filter_params, vis_params, patch_params):
    total = len(slides)
    df = pd.DataFrame({
        'slide_id': slides, 'process': np.full((total), 1, dtype=np.int8),
        'status': np.full((total), 'tbp'),

        # seg params
        'seg_level': np.full((total), int(seg_params['seg_level']), dtype=np.int8),
        'sthresh': np.full((total), int(seg_params['sthresh']), dtype=np.uint8),
        'mthresh': np.full((total), int(seg_params['mthresh']), dtype=np.uint8),
        'close': np.full((total), int(seg_params['close']), dtype=np.uint32),
        'use_otsu': np.full((total), bool(seg_params['use_otsu']), dtype=bool),

        # filter params
        'a_t': np.full((total), int(filter_params['a_t']), dtype=np.uint32),
        'a_h': np.full((total), int(filter_params['a_h']), dtype=np.uint32),
        'max_n_holes': np.full((total), int(filter_params['max_n_holes']), dtype=np.uint32),

        # vis params
        'vis_level': np.full((total), int(vis_params['vis_level']), dtype=np.int8),
        'line_thickness': np.full((total), int(vis_params['line_thickness']), dtype=np.uint32),

        # patching params
        'use_padding': np.full((total), bool(patch_params['use_padding']), dtype=bool),
        'contour_fn': np.full((total), patch_params['contour_fn'])
        })
    return df


def seg_and_patch(slide, save_dir, patch_save_dir, mask_save_dir, stitch_save_dir,
                  patch_size=256, step_size=256,
                  seg_params={'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False},
                  filter_params={'a_t': 100, 'a_h': 16, 'max_n_holes': 10},
                  vis_params={'vis_level': -1, 'line_thickness': 500},
                  patch_params={'use_padding': True, 'contour_fn': 'four_pt'},
                  patch_level=0,
                  use_default_params=False,
                  seg=False, save_mask=True,
                  stitch=False,
                  patch=False, auto_skip=False, process_list=None):

    # slides = sorted(os.listdir(slide))
    # slides = [slide for slide in slides if os.path.isfile(os.path.join(slide, slide))]

    slides= [slide]
    if process_list is None:
        df = initialize_df(slides, seg_params, filter_params, vis_params, patch_params)

    else:
        df = pd.read_csv(process_list)

    mask = df['process'] == 1
    process_stack = df[mask]

    total = len(process_stack)

    legacy_support = 'a' in df.keys()
    if legacy_support:
        print('detected legacy segmentation csv file, legacy support enabled')
        df = df.assign(**{'a_t': np.full((len(df)), int(filter_params['a_t']), dtype=np.uint32),
        'a_h': np.full((len(df)), int(filter_params['a_h']), dtype=np.uint32),
        'max_n_holes': np.full((len(df)), int(filter_params['max_n_holes']), dtype=np.uint32),
        'line_thickness': np.full((len(df)), int(vis_params['line_thickness']), dtype=np.uint32),
        'contour_fn': np.full((len(df)), patch_params['contour_fn'])})

    seg_times = 0.
    patch_times = 0.
    stitch_times = 0.

    for i in range(total):
        df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
        idx = process_stack.index[i]
        slide = process_stack.loc[idx, 'slide_id']
        

        print('\nprocessing {}'.format(slide))

        df.loc[idx, 'process'] = 0
        slide_id = os.path.basename(slide).replace(".ndpi", "").replace(".svs", "")


        if auto_skip and os.path.isfile(os.path.join(patch_save_dir, slide_id + '.h5')):
            print('{} already exist in destination location, skipped'.format(slide_id))
            df.loc[idx, 'status'] = 'already_exist'
            continue

        # Inialize WSI
        full_path = slide#os.path.join(slide, slide)
        WSI_object = WholeSlideImage(full_path, hdf5_file=None)

        if use_default_params:
            current_vis_params = vis_params.copy()
            current_filter_params = filter_params.copy()
            current_seg_params = seg_params.copy()
            current_patch_params = patch_params.copy()

        else:
            current_vis_params = {}
            current_filter_params = {}
            current_seg_params = {}
            current_patch_params = {}

            for key in vis_params.keys():
                if legacy_support and key == 'vis_level':
                    df.loc[idx, key] = -1
                current_vis_params.update({key: df.loc[idx, key]})

            for key in filter_params.keys():
                if legacy_support and key == 'a_t':
                    old_area = df.loc[idx, 'a']
                    seg_level = df.loc[idx, 'seg_level']
                    scale = WSI_object.level_downsamples[seg_level]
                    adjusted_area = int(old_area * (scale[0] * scale[1]) / (512 * 512))
                    current_filter_params.update({key: adjusted_area})
                    df.loc[idx, key] = adjusted_area
                current_filter_params.update({key: df.loc[idx, key]})

            for key in seg_params.keys():
                if legacy_support and key == 'seg_level':
                    df.loc[idx, key] = -1
                current_seg_params.update({key: df.loc[idx, key]})

            for key in patch_params.keys():
                current_patch_params.update({key: df.loc[idx, key]})

        if current_vis_params['vis_level'] < 0:
            if len(WSI_object.level_dim) == 1:
                current_vis_params['vis_level'] = 0

            else:
                wsi = WSI_object.getOpenSlide()
                best_level = wsi.get_best_level_for_downsample(64)
                current_vis_params['vis_level'] = best_level

        if current_seg_params['seg_level'] < 0:
            if len(WSI_object.level_dim) == 1:
                current_seg_params['seg_level'] = 0

            else:
                wsi = WSI_object.getOpenSlide()
                best_level = wsi.get_best_level_for_downsample(64)
                current_seg_params['seg_level'] = best_level

        w, h = WSI_object.level_dim[current_seg_params['seg_level']] 
        if w * h > 5e8:  # 1e8:
            print('level_dim {} x {} is likely too large for successful segmentation, aborting'.format(w, h))
            df.loc[idx, 'status'] = 'failed_seg'
            continue

        df.loc[idx, 'vis_level'] = current_vis_params['vis_level']
        df.loc[idx, 'seg_level'] = current_seg_params['seg_level']

        seg_time_elapsed = -1
        if seg:
            WSI_object, seg_time_elapsed = segment(WSI_object, current_seg_params, current_filter_params) 

        if save_mask:
            mask = WSI_object.visWSI(**current_vis_params)
            mask_path = os.path.join(mask_save_dir, slide_id+'.jpg')
            mask.save(mask_path)

        patch_time_elapsed = -1  # Default time
        if patch:
            current_patch_params.update({'patch_level': patch_level, 'patch_size': patch_size, 'step_size': step_size, 
                                            'save_path': patch_save_dir})
            file_path, patch_time_elapsed = patching(WSI_object = WSI_object,  **current_patch_params,)

        stitch_time_elapsed = -1
        if stitch:
            if os.path.exists(os.path.join(patch_save_dir, slide_id + ".h5")):
                file_path = os.path.join(patch_save_dir, slide_id+'.h5')
                heatmap, stitch_time_elapsed = stitching(file_path, WSI_object, downscale=64)
                stitch_path = os.path.join(stitch_save_dir, slide_id+'.jpg')
                heatmap.save(stitch_path)
            else:
                print("No contour detect")
                print("Ignore ", slide_id)
                with open(os.path.join(save_dir, "ignore.txt"), 'a') as fw:
                    fw.write(slide_id + "\n")

        # print("segmentation took {} seconds".format(seg_time_elapsed))
        # print("patching took {} seconds".format(patch_time_elapsed))
        # print("stitching took {} seconds".format(stitch_time_elapsed))
        df.loc[idx, 'status'] = 'processed'

        seg_times += seg_time_elapsed
        patch_times += patch_time_elapsed
        stitch_times += stitch_time_elapsed

    seg_times /= total
    patch_times /= total
    stitch_times /= total

    df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
    # print("average segmentation time in s per slide: {}".format(seg_times))
    # print("average patching time in s per slide: {}".format(patch_times))
    # print("average stiching time in s per slide: {}".format(stitch_times))
    return seg_times, patch_times


parser = argparse.ArgumentParser(description='seg and patch')
parser.add_argument('--slide', type=str, default="./DATA/WSI",
                    help='path of raw wsi image file')
parser.add_argument('--resolution', type=str, default='40x')
# parser.add_argument('--step_size', type=int, default=2048,
#                     help='step_size, 1024x for 20x, 2048x for 40x')
# parser.add_argument('--patch_size', type=int, default=2048,
#                     help='patch_size')
parser.add_argument('--patch', default=True)
parser.add_argument('--seg', default=True)
parser.add_argument('--stitch', default=False, action='store_true')
parser.add_argument('--auto_skip', default=False, action='store_false')
parser.add_argument('--save_dir', type=str,default="./DATA/PATCHES",
                    help='directory to save processed data')
parser.add_argument('--preset', default=None, type=str,
                    help='predefined profile of default segmentation and filter parameters (.csv)')
parser.add_argument('--patch_level', type=int, default=0,
                    help='downsample level at which to patch')
parser.add_argument('--process_list',  type=str, default=None,
                    help='name of list of images to process with parameters (.csv)')

if __name__ == '__main__':
    args = parser.parse_args()

    patch_save_dir = os.path.join(args.save_dir, 'patches')
    mask_save_dir = os.path.join(args.save_dir, 'masks')
    stitch_save_dir = os.path.join(args.save_dir, 'stitches')

    if args.process_list:
        process_list = os.path.join(args.save_dir, args.process_list)

    else:
        process_list = None


    directories = {'slide': args.slide,
                   'save_dir': args.save_dir,
                   'patch_save_dir': patch_save_dir,
                   'mask_save_dir': mask_save_dir,
                   'stitch_save_dir': stitch_save_dir}

    for key, val in directories.items():
        # print("{} : {}".format(key, val))
        if 'dir' in key:
            os.makedirs(val, exist_ok=True)

    seg_params = {'seg_level': -1, 'sthresh': 15, 'mthresh': 7, 'close': 4, 'use_otsu': False}

    filter_params = {'a_t': 100, 'a_h': 16, 'max_n_holes': 8}
    vis_params = {'vis_level': -1, 'line_thickness': 250}
    patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

    if args.preset:
        preset_df = pd.read_csv(os.path.join('presets', args.preset))
        for key in seg_params.keys():
            seg_params[key] = preset_df.loc[0, key]

        for key in filter_params.keys():
            filter_params[key] = preset_df.loc[0, key]

        for key in vis_params.keys():
            vis_params[key] = preset_df.loc[0, key]

        for key in patch_params.keys():
            patch_params[key] = preset_df.loc[0, key]

    parameters = {'seg_params': seg_params,
                  'filter_params': filter_params,
                  'patch_params': patch_params,
                  'vis_params': vis_params}

    # print(parameters)

    if args.resolution=="40x":
        patch_size = 2048
        step_size = 2028
    elif args.resolution=="20x":
        patch_size = 1024
        step_size = 1024

    seg_times, patch_times = seg_and_patch(**directories, **parameters,
                                           patch_size=patch_size, step_size=step_size,
                                           seg=args.seg,  use_default_params=False, save_mask=True,
                                           stitch=args.stitch,
                                           patch_level=args.patch_level, patch=args.patch,
                                           process_list=process_list, auto_skip=args.auto_skip)
