#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np
import cv2
from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
import sortednp as snp
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose, TriangulationParameters, eye3x4, triangulate_correspondences, build_correspondences,
    rodrigues_and_translation_to_view_mat3x4
)


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None
                          ) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )
    shape = rgb_sequence[0].shape
    triang_param_enrich = TriangulationParameters(max(5, min(shape[:2]) / 100), 5, 0)
    triang_param = TriangulationParameters(max(5, min(shape[:2]) / 100), 0, 0)
    view_mats = [eye3x4() for _ in range(len(corner_storage))]
    ratio_trash = 0.7
    all_pts_tresh = 300
    ransac_threshold = 3

    init_points = None
    init_points_ids = None
    init_pose = None
    init_frame = None
    best_init_params = (-1, -1)
    for frame in range(1, len(corner_storage)):
        b_corresp = build_correspondences(corner_storage[0], corner_storage[frame])
        E, mask = cv2.findEssentialMat(b_corresp.points_1, b_corresp.points_2, intrinsic_mat,
                                       threshold=ransac_threshold, prob=0.99999999)
        outliers = b_corresp.ids[mask.flatten() == 0]

        b_corresp = build_correspondences(corner_storage[0], corner_storage[frame], outliers)

        _, mask_H = cv2.findHomography(b_corresp.points_1, b_corresp.points_2,
                                       method=cv2.RANSAC)
        hom_ess_ratio = mask_H.mean()

        R1, R2, t = cv2.decomposeEssentialMat(E)
        r_mat = [R1.T, R1.T, R2.T, R2.T]
        vec = [np.dot(R1.T, t), np.dot(R1.T, -t), np.dot(R2.T, t), np.dot(R2.T, -t)]
        poses = []
        for i in range(4):
            poses.append(Pose(r_mat[i], vec[i]))

        triang_res = []
        for pose in poses:
            triang_corr = triangulate_correspondences(b_corresp,
                                                      eye3x4(),
                                                      pose_to_view_mat3x4(pose),
                                                      intrinsic_mat,
                                                      triang_param)
            triang_res.append((triang_corr, pose))

        (cur_points, cur_points_ids, median_cos), cur_pose = max(triang_res, key=lambda x: x[0][1].size)

        if cur_points_ids.size >= all_pts_tresh and hom_ess_ratio <= ratio_trash:
            if init_frame is None or best_init_params < (-median_cos, cur_points_ids.size):
                best_init_params = (-median_cos, cur_points_ids.size)
                init_points = cur_points
                init_points_ids = cur_points_ids
                init_pose = cur_pose
                init_frame = frame

    point_cloud_builder = PointCloudBuilder()
    if init_frame:
        point_cloud_builder.add_points(init_points_ids, init_points)
        prev_outliers = set()
        for frame in range(1, len(corner_storage)):
            ids2d = corner_storage[frame].ids.flatten()
            _, (points2d_idx, points3d_idx) = snp.intersect(ids2d, point_cloud_builder.ids.flatten(), indices=True)
            to_remove_from_ids = np.searchsorted(ids2d, list(prev_outliers))
            to_remove_from_idx = np.searchsorted(points2d_idx, to_remove_from_ids)
            points2d_idx = np.delete(points2d_idx, to_remove_from_idx, 0)
            points3d_idx = np.delete(points3d_idx, to_remove_from_idx, 0)
            points2d = corner_storage[frame].points[points2d_idx]
            points3d = point_cloud_builder.points[points3d_idx]
            if points2d_idx.size >= 6:
                retval, r_vec, t_vec, inliers = cv2.solvePnPRansac(points3d, points2d, intrinsic_mat, None,
                                                                   rvec=np.zeros((3, 1)), tvec=np.zeros((3, 1)),
                                                                   useExtrinsicGuess=True,
                                                                   reprojectionError=triang_param_enrich.max_reprojection_error)

                view_mat = rodrigues_and_translation_to_view_mat3x4(r_vec, t_vec)
                view_mats[frame] = view_mat
                outliers = np.delete(np.arange(points2d_idx.size, dtype=np.int),
                                     inliers.astype(np.int))

                prev_outliers.update(ids2d[points2d_idx[outliers]])
            else:
                view_mats[frame] = view_mats[frame - 1]
                continue
            #
            for f in range(frame):
                b_corresp = build_correspondences(corner_storage[f],
                                                  corner_storage[frame],
                                                  point_cloud_builder.ids.flatten())
                if b_corresp.ids.size > 0:
                    new_points, new_ids, cos_thres = triangulate_correspondences(
                        b_corresp,
                        view_mats[f],
                        view_mats[frame],
                        intrinsic_mat,
                        triang_param_enrich
                    )
                    point_cloud_builder.add_points(new_ids, new_points)
            print(f'number frame {frame} '
                  f', point cloud size: {point_cloud_builder.ids.size}, '
                  f'inliers size: {inliers.size}'
                  f'points size: {points2d_idx.size}, '
                  f'inliers_ratio: {inliers.size / points2d_idx.size}')

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
