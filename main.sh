slide_path=$(readlink -f $1)

tmp_dir=`dirname $slide_path`
save_path=`echo ${tmp_dir%/*}`

python -W ignore ./preprocess_SISH/create_patches_fp.py --slide $slide_path --resolution 40x --seg True --patch True --save_dir $save_path/PATCHES
python -W ignore ./preprocess_SISH/extract_mosaic.py --slide $slide_path --slide_patch_path $save_path/PATCHES/patches/ --save_path $save_path//MOSAICS
python -W ignore ./preprocess_SISH/artifacts_removal.py --slide $slide_path --resolution 40x --mosaic_path $save_path/MOSAICS 
python -W ignore ./CTransPath/ctranspath/build_index_self_for_CTransPath_get_mosaic_features.py --slide $slide_path --mosaic_path $save_path/MOSAICS --save_path $save_path/LATENT --resolution 40x

id=`basename $slide_path`
feature_path=`echo $save_path/LATENT/$id|sed "s#.ndpi#.h5#g"|sed "s#.svs#.h5#g"`
python ./AttMIL/predict.py --model_weights model_checkpoint_best.pth --wsi_path $slide_path --feature_path $feature_path --out $save_path/OUTPUT

n=`echo $id|sed "s#.ndpi##g"|sed "s#.svs##g"`
echo -e "Key patches are stored in $save_path/OUTPUT/$n \n"