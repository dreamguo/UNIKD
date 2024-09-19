import os
import sys
import ipdb
import torch
import torch.nn as nn

sys.path.append("../../incremental_learning.pytorch")
from inclearn.lib import losses


def loss_pod(pred, old_atts, old_features):
    atts = pred['pod_feature_c'][0]
    features = pred['pod_feature_c'][1]

    # reshape
    atts = atts.view(1, atts.shape[0], 1, atts.shape[1], 1)
    old_atts = old_atts.view(1, old_atts.shape[0], 1, old_atts.shape[1], 1)

    pod_flat_loss = losses.embeddings_similarity(old_features, features)
    pod_spatial_loss = losses.pod(old_atts, atts)

    return pod_flat_loss + pod_spatial_loss
