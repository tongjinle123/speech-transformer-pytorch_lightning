import numpy as np


def concat_and_subsample(features, left_frames=3, right_frames=0, skip_frames=2):

    time_steps, feature_dim = features.shape
    concated_features = np.zeros(
        shape=[time_steps, (1+left_frames+right_frames) * feature_dim], dtype=np.float32)

    concated_features[:, left_frames * feature_dim: (left_frames+1)*feature_dim] = features

    for i in range(left_frames):
        concated_features[i+1: time_steps, (left_frames-i-1)*feature_dim: (
            left_frames-i) * feature_dim] = features[0:time_steps-i-1, :]

    for i in range(right_frames):
        concated_features[0:time_steps-i-1, (right_frames+i+1)*feature_dim: (
            right_frames+i+2)*feature_dim] = features[i+1: time_steps, :]

    return concated_features[::skip_frames+1, :]
