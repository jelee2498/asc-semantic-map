
"""
00. Prepare data for parameter search using release 11 with phenotypic information from LORIS
---
0. Extract eye mask from fMRI and infer the gaze position using the pretrained model (including rule-out and by-history subjects)
"""

import os
from os.path import join, exists, basename
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from pprint import pprint
import argparse
import pickle
import shutil
# DeepMReye imports
from deepmreye import architecture, train, analyse, preprocess
from deepmreye.util import util, data_generator, model_opts
# tensorflow imports
import tensorflow as tf

# Paths come from config/paths.yml via lib/project_config.py.  The original
# working tree hard-coded storage roots here ('S:/jelee' and 'Q:/' on Windows,
# '/workspace' and '/MIPL/store9' on Linux); those are machine-specific and are
# not distributed.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'lib'))
from project_config import PROJECT, TEMPLATES, HBN_PHENOTYPE, require  # noqa: E402

proj = '02_asd_semantic_map'  # project name
pipe = '17_param_search'  # pipeline name
task = '00_prepare_data_release11'  # task in pipeline

# NOTE: this script predates the 99_main consolidation and writes into the legacy
# `17_param_search` tree.  Its two outputs are copied to 99_main/05_prepare_reg/out
# for the downstream pipelines; the copies are byte-identical.
proj_path = str(PROJECT)
pipe_path = join(proj_path, '2_pipeline', pipe)
task_path = join(pipe_path, task)
out_path, save_path, tmp_path = join(task_path, 'out'), join(task_path, 'save'), join(task_path, 'tmp')
tpl_path = str(TEMPLATES)
fig_path = join(proj_path, '3_output/figures')
raw_path = join(proj_path, '0_data/raw')
code_path = join(proj_path, '1_code', pipe)

# HBN imaging release, access-controlled.  Fails with an explicit message rather
# than silently globbing an empty directory.
store9 = str(require(HBN_PHENOTYPE, 'hbn_phenotype',
                     'the HBN preprocessed volumes read by this script'))

# Parameter settings #
prep_option = 'smooth-2'
movie_list = ['task-movieTP', 'task-movieDM']

# Path settings #
prep_volume_path_original = join(store9, 'HBN/prep/mri')
prep_path = join(proj_path, '2_pipeline/00_preprocess_fmri/24_release11_with_loris/out/mri')
functional_data_path = join(out_path, 'functional_data')
processed_data_path = join(out_path, 'processed_data')
##################

print("* Load participants list")

demo_df = pd.read_excel(join(proj_path, '2_pipeline/00_preprocess_fmri/24_release11_with_loris/out/participants_df_inc_byhx.xlsx'), index_col=0, engine='openpyxl')

# id_list for successuful preprocessing
id_list = demo_df[demo_df['prep_ok (task-movieDM_Atlas_s2_10k.dtseries.nii)']==1].index.tolist()
id_list_td = demo_df[demo_df['DX']=='TD'].index.tolist()
id_list_asd = demo_df[demo_df['DX']=='ASD'].index.tolist()

# print number of participants for each group
print(f"Number of participants: {len(id_list)}")
print(f"Number of TD participants: {len(id_list_td)}")
print(f"Number of ASD participants: {len(id_list_asd)}")


print(" Copy the preprocessed volume fMRI data to the functional_data folder")
"""
- volume fMRI source example: "/store7/jelee/02_asd_semantic_map/2_pipeline/00_preprocess_fmri/24_release11_with_loris/out/mri/sub-NDARZH672BAM/fmriprep/sub-NDARZH672BAM/func/sub-NDARZH672BAM_task-movieDM_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
- volume fMRI target example: "/store7/jelee/02_asd_semantic_map/2_pipeline/17_param_search/00_prepare_data_release11/out/functional_data/sub-NDARZH672BAM/task-movieDM/task-movieDM_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
"""
os.makedirs(functional_data_path, exist_ok=True)

for sub_id in tqdm(id_list, desc='Copying functional data'):
    if not exists(join(out_path, 'functional_data', sub_id)):
        for movie in movie_list:
            src = join(prep_path, sub_id, 'fmriprep', sub_id, 'func', f'{sub_id}_{movie}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')
            trg = join(functional_data_path, sub_id, movie, f'{movie}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')
            os.makedirs(join(functional_data_path, sub_id, movie), exist_ok=True)
            if not exists(trg):
                try:
                    shutil.copy(src, trg)
                except FileNotFoundError:
                    try:
                        src = join(prep_volume_path_original, sub_id, 'fmriprep', sub_id, 'func', f'{sub_id}_{movie}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')
                        shutil.copy(src, trg)
                    except FileNotFoundError:
                        print(f'File not found: {src}')


print("* Extract eye mask from fMRI")

# preload masks to save time within participant loop
(eyemask_small, eyemask_big, dme_template, mask, x_edges, y_edges, z_edges) = preprocess.get_masks()

# extract eye mask
for sub_id in tqdm(id_list, desc='Extracting eye mask'):  
    print('Running participant {}...'.format(sub_id))
    for movie in movie_list:
        movie_folder = join(functional_data_path, sub_id, movie)
        if glob(join(movie_folder, '*.p')):  # if mask already exists, skip
            continue
        for f in os.listdir(movie_folder):
            if f.endswith('.nii.gz'):
                func_path = join(movie_folder, f)  # filepath to functional
                preprocess.run_participant(func_path, dme_template, eyemask_big, eyemask_small, x_edges, y_edges, z_edges)


print("* Quality control of eye mask extraction")

# The QC table stores clickable links to each subject's eye-mask report.  The
# original hard-coded the lab's own mount points here ('/workspace' rewritten to
# 'file://S:/jelee'); build a file:// URL from the configured project root instead.
qc_df = {}
# movieDM
htmlp_list = []
for sub_id in id_list:
    htmlp = glob(join(out_path, 'functional_data', sub_id, 'task-movieDM', 'report*html'))[0]
    htmlp = Path(htmlp).resolve().as_uri()
    htmlp_list.append(htmlp)
qc_df['movieDM'] = htmlp_list
# movieTP 
htmlp_list = [htmlp.replace('movieDM','movieTP') for htmlp in htmlp_list]
qc_df['movieTP'] = htmlp_list
qc_df = pd.DataFrame(qc_df, index=id_list)

# add columns for rating
qc_df['rating-movieDM (0=failed, 1=succesful)'] = 'nan'
qc_df['rating-movieTP (0=failed, 1=succesful)'] = 'nan'
qc_df['rating-movieDM after aggressive registration (0=failed, 1=succesful)'] = None
qc_df['rating-movieTP after aggressive registration (0=failed, 1=succesful)'] = None
qc_df['final rating-movieDM (0=failed, 1=succesful)'] = 'nan'
qc_df['final rating-movieTP (0=failed, 1=succesful)'] = 'nan'

# get rating from qc_deepmreye.xlsx
try:
    qc_deepmreye_df = pd.read_excel(join(out_path, 'qc_deepmreye_inc_ruleout.xlsx'), index_col=0, engine='openpyxl')
    for sub_id in qc_df.index:
        for movie in movie_list:
            try:
                qc_df.loc[sub_id, f'rating-{movie} (0=failed, 1=succesful)'.replace('task-','')] = qc_deepmreye_df.loc[sub_id, f'rating-{movie} (0=failed, 1=succesful)'.replace('task-','')]
                qc_df.loc[sub_id, f'rating-{movie} after aggressive registration (0=failed, 1=succesful)'.replace('task-','')] = qc_deepmreye_df.loc[sub_id, f'rating-{movie} after aggressive registration (0=failed, 1=succesful)'.replace('task-','')]
                qc_df.loc[sub_id, f'final rating-{movie} (0=failed, 1=succesful)'.replace('task-','')] = qc_deepmreye_df.loc[sub_id, f'final rating-{movie} (0=failed, 1=succesful)'.replace('task-','')]
            except KeyError:
                pass
except FileNotFoundError:
    print('qc_deepmreye_inc_ruleout.xlsx not found...')

qc_df.to_excel(join(out_path, 'qc_deepmreye_inc_byhx.xlsx'), engine='openpyxl')

# convert into hyperlink 
# - https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=kirin_grimm&logNo=220682199418


print("* Load 1st QC failed participants")

from copy import deepcopy

qc_df = pd.read_excel(join(out_path, 'qc_deepmreye_inc_byhx.xlsx'), index_col=0, engine='openpyxl')

failed_id_list = {movie: [] for movie in movie_list}
for movie in movie_list:
    failed_ids = (qc_df[f'rating-{movie.replace("task-","")} (0=failed, 1=succesful)']==0).values.nonzero()[0]
    failed_id_list[movie] = qc_df.index[failed_ids].tolist()

# get only newly preprocessed subjects
# - check if the 'rating-movieDM after aggressive registration (0=failed, 1=succesful)' is 1 or 0
for movie in movie_list:
    print(f'Checking {movie}...')
    temp_id_list = deepcopy(failed_id_list[movie])  # to avoid changing the list during iteration
    for sub_id in temp_id_list:
        if qc_df.loc[sub_id, f'rating-{movie} after aggressive registration (0=failed, 1=succesful)'.replace('task-','')] == 1 or qc_df.loc[sub_id, f'rating-{movie} after aggressive registration (0=failed, 1=succesful)'.replace('task-','')] == 0:
            print(f'{sub_id} is excluded from the list...')
            failed_id_list[movie].remove(sub_id)


print("* Extract eye mask with different transforms for 1st QC failed participants")

for movie in movie_list:
    for sub_id in tqdm(failed_id_list[movie], desc=f'Extracting eye mask for {movie}'):
        movie_folder = join(functional_data_path, sub_id, movie)
        for f in os.listdir(movie_folder):
            if f.endswith('.nii.gz'):
                func_path = join(movie_folder, f)  # filepath to functional
                preprocess.run_participant(func_path, dme_template, eyemask_big, eyemask_small, x_edges, y_edges, z_edges, transforms=['Affine', 'Affine', 'SyNAggro'])


print("* 2nd QC of eye mask extraction")
"""
Done in excel
"""


print("* Load 2nd QC failed participants and exclude them from further analysis")

qc_df = pd.read_excel(join(out_path, 'qc_deepmreye_inc_byhx.xlsx'), index_col=0, engine='openpyxl')

tp_failed_ids = (qc_df['final rating-movieTP (0=failed, 1=succesful)'] == 0).values.nonzero()[0]
dm_failed_ids = (qc_df['final rating-movieDM (0=failed, 1=succesful)'] == 0).values.nonzero()[0]

id_list_failed_tp = qc_df.index[tp_failed_ids].tolist()
id_list_failed_dm = qc_df.index[dm_failed_ids].tolist()

id_list_tp = [sub_id for sub_id in id_list if sub_id not in id_list_failed_tp]
id_list_dm = [sub_id for sub_id in id_list if sub_id not in id_list_failed_dm]

# print number of participants for each movie
print(f"Number of participants for movieTP: {len(id_list_tp)}")
print(f"Number of participants for movieDM: {len(id_list_dm)}")


print("* Combine processed masks with labels")

os.makedirs(processed_data_path, exist_ok=True)

for movie, id_list in zip(movie_list, [id_list_tp, id_list_dm]):
    for sub_id in tqdm(id_list, desc=f'Combining processed masks with labels-{movie}'):
        if basename(join(processed_data_path, f'{sub_id}_{movie}_no_label.npz')) in os.listdir(processed_data_path):  # if already exists, skip
            continue
        movie_folder = join(functional_data_path, sub_id, movie)
        participant_data, participant_labels, participant_ids = [], [], []
        for file_idx, file in enumerate(os.listdir(movie_folder)):
            if not file.endswith(".p"):
                continue
            # load mask and normalize it
            this_mask = join(movie_folder, file)
            this_mask = pickle.load(open(this_mask, 'rb'))
            this_mask = preprocess.normalize_img(this_mask)

            # if experiment has no labels use dummy labels
            this_label = np.zeros((this_mask.shape[3], 10, 2))

            # check if each functional image has a corresponding label (note that mask has time as third dimension)
            if not this_mask.shape[3] == this_label.shape[0]:
                print('WARNING --- Skipping Subject {} Movie {} --- Wrong alignment (Mask {} - Label {}).'.format(sub_id, movie, this_mask.shape, this_label.shape))
                continue

            # store across runs 
            participant_data.append(this_mask)
            participant_labels.append(this_label)
            participant_ids.append(([sub_id]*this_label.shape[0], [file_idx]*this_label.shape[0]))

        # save participant file
        preprocess.save_data(f'{sub_id}_{movie}_no_label', participant_data, participant_labels, participant_ids, processed_data_path, center_labels=False)


print("* Check GPU availability")

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled for GPUs")
    except RuntimeError as e:
        print(e)


print("* Set GPU to use")
"""
Check GPU usage using nvidia-smi and set the GPU to use
"""
tf.config.experimental.set_visible_devices([gpus[0]], 'GPU')


print("* Infer gaze position using the pretrained model")

opts = model_opts.get_opts()
dataset = [join(processed_data_path, p) for p in os.listdir(processed_data_path) if 'no_label' in p]
generators = data_generator.create_generators(dataset, dataset)
generators = (*generators, dataset, dataset)  # add participant list

model_weights = join(out_path, 'model_weights', 'datasets_1to5.h5')

# get untrained model and load with trained weights
(model, model_inference) = train.train_model(dataset="hbn_td_asd_inc_byhx", generators=generators, opts=opts, return_untrained=True)
model_inference.load_weights(model_weights)

(evaluation, scores) = train.evaluate_model(dataset='hbn_td_asd_inc_byhx', model=model_inference, generators=generators,
                                            save=True, model_path=out_path+os.path.sep, model_description='', verbose=2)


print("* Edit the gaze results")

import pickle

gaze_results = np.load(join(out_path, 'results_hbn_td_asd_inc_byhx.npy'), allow_pickle=True)
gaze_results = gaze_results.item()

new_gaze_reulsts = {}
for key in tqdm(gaze_results.keys(), desc='Editing gaze results'):
    new_gaze_reulsts[basename(key).replace('_no_label.npz', '')] = {}
    sub_results = gaze_results[key]
    len_tr = len(sub_results['euc_pred'])
    if len_tr==250 or len_tr==750:
        for data in sub_results.keys():
            new_gaze_reulsts[basename(key).replace('_no_label.npz', '')][data.replace('_y','_xy').replace('euc_pred','uncertainty')] = sub_results[data]
    else:
        print(f'{key} is being edited..')
        for data in sub_results.keys():
            new_gaze_reulsts[basename(key).replace('_no_label.npz', '')][data.replace('_y','_xy').replace('euc_pred','uncertainty')] = sub_results[data][:250] if 'TP' in key else sub_results[data][:750]

with open(join(out_path, 'results_hbn_td_asd_inc_byhx_edited.pickle'),'wb') as fp:
    pickle.dump(new_gaze_reulsts, fp)


print("* Add 'Rating_deepmreye_{movie}' column to the participants dataframe")

participants_df = pd.read_excel(join(proj_path, '2_pipeline/00_preprocess_fmri/24_release11_with_loris/out/participants_df_inc_byhx.xlsx'), index_col=0, engine='openpyxl')
deepmreye_qc_df = pd.read_excel(join(out_path, 'qc_deepmreye_inc_byhx.xlsx'), index_col=0, engine='openpyxl')

# add columns for rating
participants_df['Rating_deepmreye_movieDM'] = 'nan'
participants_df['Rating_deepmreye_movieTP'] = 'nan'

# update rating
for sub_id in participants_df.index:
    for movie in movie_list:
        try:
            rating = deepmreye_qc_df.loc[sub_id, f'final rating-{movie} (0=failed, 1=succesful)'.replace('task-','')]
        except KeyError:
            rating = 'nan'
        participants_df.loc[sub_id, f'Rating_deepmreye_{movie}'.replace('task-','')] = rating

participants_df.to_excel(join(out_path, 'participants_df_deepmreye_inc_byhx.xlsx'), engine='openpyxl')
