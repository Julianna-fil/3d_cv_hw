#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'calc_track_interval_mappings',
    'calc_track_len_array_mapping',
    'without_short_tracks'
]

from collections import namedtuple

import click
import cv2
import numpy as np
import pims

from _corners import (
    FrameCorners,
    CornerStorage,
    StorageImpl,
    dump,
    load,
    draw,
    calc_track_interval_mappings,
    calc_track_len_array_mapping,
    without_short_tracks,
    create_cli
)


DETECTION_PYRAMID_COUNT = 2
MAX_CORNER_COUNT_PER_PYR_LEVEL = 10**5
MIN_DISTANCE_BETWEEN_CORNERS = 7
CORNER_BLOCK_SIZE = 6

GOOD_FEATURES_TO_TRACK_SETTINGS = {
    'maxCorners': MAX_CORNER_COUNT_PER_PYR_LEVEL,
    'qualityLevel': 0.01,
    'minDistance': MIN_DISTANCE_BETWEEN_CORNERS,
    'blockSize': CORNER_BLOCK_SIZE,
    'useHarrisDetector': False
}

MAX_L1_ERROR = 0.25
MAX_BF_ERROR = 15.0

LUCAS_KANADE_SETTINGS = {
    'winSize': (31, 31),
    'maxLevel': 4,
    'criteria': (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        100,
        0.001
    ),
    'minEigThreshold': 0.001
}


def _upscale_points(points, levels):
    points = points * 2**levels
    increment = np.zeros(levels.shape)
    mask = levels > 0
    increment[mask] = 2**(levels[mask] - 1)
    points += increment
    return points.astype(np.float32)


def _detect_good_features_to_track(image, mask=None):
    return cv2.goodFeaturesToTrack(
        image,
        mask=mask,
        **GOOD_FEATURES_TO_TRACK_SETTINGS
    )


def _calc_min_eigen_values_in_points(image, points):
    eigens = cv2.cornerMinEigenVal(image, CORNER_BLOCK_SIZE)
    points = np.round(points).astype(np.int)
    rows = points[:, 1].flatten()
    cols = points[:, 0].flatten()
    return eigens[rows, cols].reshape(-1, 1)


_Corners = namedtuple('_Corners', (
    'ids',
    'points',
    'min_eigens',
    'l1_errors',
    'bf_errors',
    'levels'
))


def _detect_corners(image):
    points = []
    levels = []
    eigens = []
    for level in range(DETECTION_PYRAMID_COUNT):
        curr_points = _detect_good_features_to_track(image)
        points.append(curr_points.reshape(-1, 2))
        levels.append(np.full((points[-1].shape[0], 1), level))
        eigens.append(_calc_min_eigen_values_in_points(image, points[-1]))
        image = cv2.pyrDown(image)
    levels = np.vstack(levels)
    points = _upscale_points(np.vstack(points), levels)
    eigens = np.vstack(eigens)
    ids = np.arange(0, points.shape[0], 1, np.int64).reshape(-1, 1)
    sorting_idx = np.argsort(eigens.flatten())
    return _Corners(
        ids[sorting_idx],
        points[sorting_idx],
        eigens[sorting_idx],
        np.zeros((sorting_idx.shape[0], 1)),
        np.zeros((sorting_idx.shape[0], 1)),
        levels[sorting_idx]
    )


def _to_uint8(image_f32):
    return np.round(255 * image_f32).astype(np.uint8)


def _run_optical_flow(image_0, image_1, points_0):
    points_1, status, errors = cv2.calcOpticalFlowPyrLK(
        _to_uint8(image_0),
        _to_uint8(image_1),
        points_0,
        None,
        **LUCAS_KANADE_SETTINGS
    )
    return points_1, status.reshape(-1, 1) != 0, (errors / 255.).reshape(-1, 1)


def _filter_points(points, status):
    return points[np.hstack((status, status))].reshape(-1, 2)


def _filter_corners(corners, status):
    return _Corners(
        ids=corners.ids[status].reshape(-1, 1),
        points=_filter_points(corners.points, status),
        min_eigens=corners.min_eigens[status].reshape(-1, 1),
        l1_errors=corners.l1_errors[status].reshape(-1, 1),
        bf_errors=corners.bf_errors[status].reshape(-1, 1),
        levels=corners.levels[status].reshape(-1, 1)
    )


def _track_corners(image_0, image_1, corners_0):
    points_1, status_1, l1_errors = _run_optical_flow(
        image_0,
        image_1,
        corners_0.points
    )
    points_1 = np.nan_to_num(points_1)
    points_2, status_2, _ = _run_optical_flow(image_1, image_0, points_1)

    bf_errors = np.linalg.norm(
        points_2 - corners_0.points,
        axis=1
    ).reshape(-1, 1)

    status = \
        status_1 & \
        status_2 & \
        (l1_errors <= MAX_L1_ERROR) & \
        (bf_errors <= MAX_BF_ERROR) & \
        (points_1[:, 0] >= 0).reshape(-1, 1) & \
        (points_1[:, 1] >= 0).reshape(-1, 1) & \
        (points_1[:, 0] < image_0.shape[1] - 0.5).reshape(-1, 1) & \
        (points_1[:, 1] < image_0.shape[0] - 0.5).reshape(-1, 1)

    corners_1 = corners_0._replace(
        points=points_1,
        l1_errors=l1_errors,
        bf_errors=bf_errors
    )
    corners_1 = _filter_corners(corners_1, status)
    corners_1 = corners_1._replace(
        min_eigens=_calc_min_eigen_values_in_points(image_1, corners_1.points)
    )

    return corners_1


def _to_int_tuple(point):
    return tuple(map(int, np.round(point)))


def _create_mask(image_shape):
    rows, cols = image_shape
    mask = [np.zeros((rows // 2**lvl, cols // 2**lvl), dtype=np.uint8)
            for lvl in range(DETECTION_PYRAMID_COUNT)]
    return mask


def _mark_point(mask, point, level):
    point = _to_int_tuple(point / 2**level)
    cv2.circle(mask[level], point, MIN_DISTANCE_BETWEEN_CORNERS, (255,), -1)


def _is_point_free(mask, point, level):
    mask = mask[level]
    point = point / 2**level
    mask_size = np.array([mask.shape[1], mask.shape[0]])
    if np.any(point < 0) or np.any(point > mask_size):
        return False
    col, row = _to_int_tuple(point)
    return mask[row, col] == 0


def _mark_all_points(points, levels, mask):
    for point, level in zip(points, levels.flatten()):
        _mark_point(mask, point, level)


def _count_points_on_pyramid_levels(levels):
    return [len(levels[levels == level])
            for level in range(DETECTION_PYRAMID_COUNT)]


def _detect_new_corners(image_1, tracked_corners, free_id):
    mask = _create_mask(image_1.shape)
    _mark_all_points(tracked_corners.points, tracked_corners.levels, mask)

    point_cnts = _count_points_on_pyramid_levels(tracked_corners.levels)

    detected_corners = _detect_corners(image_1)
    new_corners = []
    for point, quality, level in zip(detected_corners.points,
                                     detected_corners.min_eigens,
                                     detected_corners.levels):
        level = level[0]
        if not _is_point_free(mask, point, level):
            continue
        if point_cnts[level] >= MAX_CORNER_COUNT_PER_PYR_LEVEL:
            continue
        _mark_point(mask, point, level)
        point_cnts[level] += 1
        new_corners.append((free_id, point, quality, level))
        free_id += 1

    if not new_corners:
        return tracked_corners

    new_corners_len = len(new_corners)

    def get_arr(idx):
        return np.array([c[idx] for c in new_corners])

    return _Corners(
        np.vstack((tracked_corners.ids, get_arr(0).reshape(-1, 1))),
        np.vstack((tracked_corners.points, get_arr(1))),
        np.vstack((tracked_corners.min_eigens, get_arr(2).reshape(-1, 1))),
        np.vstack((tracked_corners.l1_errors, np.zeros((new_corners_len, 1)))),
        np.vstack((tracked_corners.bf_errors, np.zeros((new_corners_len, 1)))),
        np.vstack((tracked_corners.levels, get_arr(3).reshape(-1, 1)))
    )


def _track_and_detect_corners(image_0, image_1, corners_0, free_id):
    corners_1 = _track_corners(image_0, image_1, corners_0)
    quan = 0.95
    eps = 1e-9
    min_eigens_quan = np.quantile(corners_1.min_eigens, quan)
    l1_errors_quan = np.quantile(corners_1.l1_errors, quan)
    bf_errors_quan = np.quantile(corners_1.bf_errors, quan)
    filtering_mask = \
        (corners_1.min_eigens < min_eigens_quan + eps) & \
        (corners_1.l1_errors < l1_errors_quan + eps) & \
        (corners_1.bf_errors < bf_errors_quan + eps)
    corners_1 = _filter_corners(corners_1, filtering_mask)
    corners_1 = _detect_new_corners(image_1, corners_1, free_id)
    return corners_1


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()
        self._free_id = 0

    def get_free_id(self):
        return self._free_id

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        self._free_id = corners.ids.max() + 1
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def _convert_corners(corners):
    return FrameCorners(
        corners.ids,
        corners.points,
        CORNER_BLOCK_SIZE * 2**corners.levels,
        )


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param builder: corner storage builder.
    """
    image_0 = frame_sequence[0]
    corners_0 = _detect_corners(image_0)
    builder.set_corners_at_frame(0, _convert_corners(corners_0))
    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        corners_1 = _track_and_detect_corners(image_0, image_1, corners_0,
                                              builder.get_free_id())
        builder.set_corners_at_frame(frame, _convert_corners(corners_1))
        image_0 = image_1
        corners_0 = corners_1


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return without_short_tracks(builder.build_corner_storage(), 2)


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter