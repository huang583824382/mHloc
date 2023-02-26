import argparse
import pycolmap
import os
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster
from typing import Dict, List, Union
from pathlib import Path
import numpy as np
from hloc import logger, extractors, matchers
from hloc import extract_features, match_features, reconstruction, localize_inloc, visualization, pairs_from_retrieval, localize_sfm
from tqdm import tqdm
from hloc.utils.parsers import parse_image_lists, parse_retrieval
from hloc.utils.base_model import dynamic_load
import torch
import time
from hloc.utils import viz_3d
import dash
import dash_core_components as dcc
from dash import html




class Singleton(object):
    def __init__(self, MainWork):
        self._MainWork = MainWork
        self._instance = {}

    def __call__(self):
        if self._MainWork not in self._instance:
            self._instance[self._MainWork] = self._MainWork()
        return self._instance[self._MainWork]


@Singleton
class MainWork(object):
    sfm_model: pycolmap.Reconstruction
    global_feature_path: Path
    local_feature_path: Path
    global_feature_conf = extract_features.confs['netvlad']
    local_feature_conf = extract_features.confs['superpoint_aachen']
    matcher_conf = match_features.confs['superglue-fast']
    netvlad_model: any
    superpoint_model: any
    superglue_model: any
    
    
    def __init__(self):
        print('MainWork Loading...')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        Model = dynamic_load(extractors, self.global_feature_conf['model']['name'])
        self.netvlad_model = Model(self.global_feature_conf['model']).eval().to(device)

        Model = dynamic_load(extractors, self.local_feature_conf['model']['name'])
        self.superpoint_model = Model(self.local_feature_conf['model']).eval().to(device)

        Model = dynamic_load(matchers, self.matcher_conf['model']['name'])
        self.superglue_model = Model(self.matcher_conf['model']).eval().to(device)
        print('MainWork Load Success!')




def reconstruction_from_dataset(dataset: Union[Path, str], pair_num: int)->pycolmap.Reconstruction:
    # path definition
    if not isinstance(dataset, Path):
        dataset = Path(dataset)
    
    outputs = Path('outputs/'+dataset.name)
    logger.debug('outputs path: ', outputs)
    db_pairs = outputs/'pair'/'db_pairs.txt'
    matches = outputs/'match'/'db_matches.h5'
    sfm_dir = outputs/'sfm'

    # config definition
    global_feature_conf = extract_features.confs['netvlad']
    local_feature_conf = extract_features.confs['superpoint_aachen']
    matcher_conf = match_features.confs['superglue']

    # extract global features
    global_features = extract_features.main(global_feature_conf, dataset, outputs)
    # retrival pairs
    pairs_from_retrieval.main(global_features, db_pairs, pair_num)
    # extract local features
    local_features = extract_features.main(local_feature_conf, dataset, outputs)
    # match features
    match_features.main(matcher_conf, db_pairs, local_features, matches=matches)
    # reconstruct sfm model
    model = reconstruction.main(sfm_dir, dataset, db_pairs, local_features, matches)
    return model


def localize_image(dataset: Union[Path, str],
        image_query_dir: Union[Path, str],
        image_query_name: str = None,
        #  results: Path,
        ransac_thresh: int = 12,
        covisibility_clustering: bool = False,
        prepend_camera_name: bool = False,
        config: Dict = None, 
        overwrite: bool = False):

    mainWork = MainWork()
    time1 = time.time()
    # config definition
    # global_feature_conf = extract_features.confs['netvlad']
    # local_feature_conf = extract_features.confs['superpoint_aachen']
    # matcher_conf = match_features.confs['superglue']

    # path definition
    if not isinstance(dataset, Path):
        dataset = Path(dataset)
    if not isinstance(image_query_dir, Path):
        image_query_dir = Path(image_query_dir)
    
    outputs = Path('outputs/'+dataset.name)
    # logger.debug('outputs path: ', outputs)
    db_pairs = outputs/'pair'/'db_pairs.txt'
    q_pairs = outputs/'pair'/'q_pairs.txt'
    q_matches = outputs/'match'/'q_matches.h5'
    sfm_dir = outputs/'sfm'
    global_features = outputs/(mainWork.global_feature_conf['output']+'.h5')
    local_features = outputs/(mainWork.local_feature_conf['output']+'.h5')

    assert image_query_dir.exists(), image_query_dir
    assert sfm_dir.exists(), sfm_dir
    assert global_features.exists(), global_features
    assert local_features.exists(), local_features
    time2 = time.time()
    print('configurate for ', time2-time1, 's')
    # logger.info('Reading the 3D model...')
    reference_sfm = pycolmap.Reconstruction(sfm_dir)
    # 获取sfm模型中的图片
    sfm_images = []
    for item in reference_sfm.images:
        sfm_images.append(reference_sfm.images[item].name)
    # 查询的图片
    query_images = [p. relative_to(image_query_dir).as_posix() for p in (image_query_dir).iterdir()]
    if image_query_name in query_images:
        query_images = [image_query_name]
    time3 = time.time()
    # print('loading 3D model for ', time3-time2, 's')

    # 提取查询图片的全局、局部特征，进行全局查询和特征匹配
    # logger.info('Extracting global features...')
    extract_features.main(mainWork.global_feature_conf, image_query_dir, outputs, image_list=query_images, model=mainWork.netvlad_model, overwrite=overwrite)
    time4 = time.time()
    # print('global feature for ', time4-time3, 's')
    # logger.info('Extracting local features...')
    extract_features.main(mainWork.local_feature_conf, image_query_dir, outputs, image_list=query_images, model=mainWork.superpoint_model, overwrite=overwrite)
    time5 = time.time()
    # print('local feature for ', time5-time4, 's')
    # logger.info('Retrieving images...')
    pairs_from_retrieval.main(global_features, q_pairs, 3, query_list=query_images, db_model=sfm_dir)
    time6 = time.time()
    # print('pairs for ', time6-time5, 's')
    # logger.info('Matching features...')
    match_features.main(mainWork.matcher_conf, q_pairs, local_features, matches = q_matches, model=mainWork.superglue_model, overwrite=overwrite)
    time7 = time.time()
    # print('matching for ', time7-time6, 's')
    
    i = 0

    camera = pycolmap.infer_camera_from_image(image_query_dir/query_images[i])
    
    retrieval_dict = parse_retrieval(q_pairs)
    # # print(retrieval_dict)
    conf = {
        'estimation': {'ransac': {'max_error': ransac_thresh}},
        'refinement': {'refine_focal_length': True, 'refine_extra_params': True},
    }
    localizer = QueryLocalizer(reference_sfm, conf)
    query_list = retrieval_dict[query_images[i]]

    ref_list = []
    for r in query_list:
        if reference_sfm.find_image_with_name(r) is not None:
            ref_list.append(reference_sfm.find_image_with_name(r).image_id)
    # print(query_images[i])
    
    ret, log = pose_from_cluster(localizer, query_images[i], camera, ref_list, local_features, q_matches)
    time8 = time.time()
    # print('localize for ', time8-time7, 's')
    print('total: ', time8 - time1, 's')
    
    fig = None

    # if ret['success'] is True:
    #     fig = viz_3d.init_figure()
    #     viz_3d.plot_reconstruction(fig, reference_sfm, color='rgba(255,0,0,0.5)', name="mapping")
    #     pose = pycolmap.Image(tvec=ret['tvec'], qvec=ret['qvec'])
    #     viz_3d.plot_camera_colmap(fig, pose, camera, color='rgba(0,255,0,0.5)', name='query')

    return ret, reference_sfm, log, fig
