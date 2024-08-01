from nltk.tree import *
from nltk.corpus import wordnet as wn
import nltk
nltk.download('wordnet')

import torch
import numpy as np
import scipy

from torchvision.models import (googlenet as pt_model1, GoogLeNet_Weights as pt_model1_weights,
                                alexnet as pt_model2, AlexNet_Weights as pt_model2_weights)

import torchvision
import torch.distributions as dist

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import defaultdict
from tqdm import tqdm

# Step 1: Initialize model with the best available weights
model1_weights = pt_model1_weights.DEFAULT
model1 = pt_model1(weights=model1_weights)
model1.eval()

# Step 2: Initialize the inference transforms
model1_preprocess = model1_weights.transforms()
img_classes = model1_weights.meta["categories"]

def get_min_levels(class1, class2, class_name=False):
    try:
        if class_name:
            syn1 = wn.synsets(class1.replace(' ', '_'), 'n')[0]
            syn2 = wn.synsets(class2.replace(' ', '_'), 'n')[0]
        else:
            syn1 = wn.synsets(img_classes[class1].replace(' ', '_'), 'n')[0]
            syn2 = wn.synsets(img_classes[class2].replace(' ', '_'), 'n')[0]
        
        levels = (syn1.shortest_path_distance(syn2) - 1)/2
    except:
        levels = 1
        syn1, syn2 = None, None
    
    return levels, syn1, syn2

def get_set_min_levels(set1, set2):
    diff = 0
    for i in set1:
        for j in set2:
            level_diff, _, _ = get_min_levels(i, j)
            
            diff += level_diff
            
    set_diff = diff/(len(set1)*len(set2))
    
    return set_diff

def squared_cos_sim(v, w, eps=1e-6):
    """PyTorch operation to compute the elementwise squared cosine
    similarity between two sets of vectors."""
    num = torch.sum(v * w, dim=1)**2
    den = torch.sum(v**2, dim=1) * torch.sum(w**2, dim=1)
    return num / (den + eps)

def onehot(y, num_classes=None):
    y = y.long()
    if num_classes is None:
        num_classes = y.max().item() + 1

    y_onehot = torch.zeros(y.shape[0], num_classes)
    y_onehot.scatter_(1, y.unsqueeze(1), 1)

    return y_onehot

import os
from torch.utils.data import Dataset
from PIL import Image
import json

class ImageNetKaggle(Dataset):
    def __init__(self, root, split, transform=None):
        self.samples = []
        self.targets = []
        self.transform = transform
        self.syn_to_class = {}
        with open(os.path.join(root, "imagenet_class_index.json"), "rb") as f:
                    json_file = json.load(f)
                    for class_id, v in json_file.items():
                        self.syn_to_class[v[0]] = int(class_id)
        with open(os.path.join(root, "ILSVRC2012_val_labels.json"), "rb") as f:
                    self.val_to_syn = json.load(f)
        samples_dir = os.path.join(root, "ILSVRC/Data/CLS-LOC", split)
        for entry in os.listdir(samples_dir):
            if entry != ".DS_Store":
                if split == "train":
                    syn_id = entry
                    target = self.syn_to_class[syn_id]
                    syn_folder = os.path.join(samples_dir, syn_id)
                    for sample in os.listdir(syn_folder):
                        sample_path = os.path.join(syn_folder, sample)
                        self.samples.append(sample_path)
                        self.targets.append(target)
                elif split == "val":
                    syn_id = self.val_to_syn[entry]
                    target = self.syn_to_class[syn_id]
                    sample_path = os.path.join(samples_dir, entry)
                    self.samples.append(sample_path)
                    self.targets.append(target)
    def __len__(self):
            return len(self.samples)
    def __getitem__(self, idx):
            x = Image.open(self.samples[idx]).convert("RGB")
            if self.transform:
                x = self.transform(x)
            return x, self.targets[idx]

def calculate_fid_score(x_recon, x):
    # Load InceptionV3
    inception_model = torchvision.models.inception_v3(pretrained=True)
    inception_model = inception_model.eval()

    # Function to preprocess images and extract features
    def get_features(images):
        # Preprocess images
        images = torch.nn.functional.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        images = images * 2 - 1  # Scale from [0, 1] to [-1, 1]

        # Extract features
        with torch.no_grad():
            features = inception_model(images)

        return features.numpy()

    # Calculate features
    x_recon_features = get_features(x_recon)
    x_features = get_features(x)

    # Calculate mean and covariance
    mu1 = np.mean(x_recon_features, axis=0)
    sigma1 = np.cov(x_recon_features, rowvar=False)
    mu2 = np.mean(x_features, axis=0)
    sigma2 = np.cov(x_features, rowvar=False)

    # Calculate sum of squared differences
    ssd = np.sum((mu1 - mu2)**2.0)

    # Calculate sqrt of product between cov
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))

    # Check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    # Calculate score
    fid_score = ssd + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean)
    return fid_score

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

def multiclass_roc_auc_score(y_true, y_pred, c, average="macro"):
    y_true = label_binarize(y_true, classes=[*range(0, c)])
    y_pred = label_binarize(y_pred, classes=[*range(0, c)])

    # Compute ROC AUC for each class
    roc_auc = [roc_auc_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]

    # Compute average ROC AUC
    if average == "macro":
        return sum(roc_auc) / len(roc_auc)
    elif average == "weighted":
        class_counts = np.bincount(y_true.flatten())[1:]  # Count of each class in y_true
        return sum(auc * count for auc, count in zip(roc_auc, class_counts)) / sum(class_counts)
    else:
        raise ValueError(f"Unknown average type: {average}. Use 'macro' or 'weighted'")

def JS_divergence(prob_dists):
    num_models = len(prob_dists)
    assert num_models > 1
    
    avg_dist = sum([p.probs for p in prob_dists])/num_models

    kl_divs = [dist.kl_divergence(p, dist.Categorical(probs=avg_dist)) for p in prob_dists]

    js_div = sum(kl_divs)/num_models

    return js_div

from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform
import numpy as np

def compute_cluster(data, eps, min_samples):
    """
    Perform DBSCAN clustering and compute the Cluster Spread Index.
    
    Args:
    data: Numpy array of shape (N, d), representing N points in d dimensions.
    eps: The maximum distance between two samples for one to be considered
         as in the neighborhood of the other in DBSCAN algorithm.
    min_samples: The number of samples in a neighborhood for a point
                 to be considered as a core point in DBSCAN algorithm.

    Returns:
    db: DBSCAN object after fitting
    CSI: Cluster Spread Index of the clustering
    """
    # Perform DBSCAN clustering
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)

    # Compute pairwise distances between all points
    distances = squareform(pdist(data))

    # Initialize list to hold average intra-cluster distances for each cluster
    intra_cluster_distances = []

    # Iterate over each cluster label (excluding noise)
    for label in set(db.labels_):
        if label == -1:
            continue

        # Get the points in this cluster
        cluster_points = data[db.labels_ == label]

        # Compute the average intra-cluster distance for this cluster
        intra_cluster_distances.append(np.mean(distances[label, db.labels_ == label]))

    # Compute the average intra-cluster distance across all clusters
    mean_intra_cluster_distance = np.mean(intra_cluster_distances)

    # Compute the average inter-cluster distance
    inter_cluster_distances = []
    for i in np.unique(db.labels_):
        for j in np.unique(db.labels_):
            if i != j and i != -1 and j != -1:
                inter_cluster_distances.append(np.mean(distances[db.labels_ == i][:, db.labels_ == j]))
    if len(inter_cluster_distances) == 0:
        mean_inter_cluster_distance = -1
    else:
        mean_inter_cluster_distance = np.mean(inter_cluster_distances)

    print(mean_intra_cluster_distance, mean_inter_cluster_distance)

    # Compute the CSI
    CSI = mean_intra_cluster_distance / mean_inter_cluster_distance

    return db, CSI

# Template code to train CVAE, generic classifier, evaluate one classifier, evaluate a set of classifiers. Do not use without customising to use-case!
def train_cvae(
    vae, 
    train_dataset, 
    max_iters=10000, 
    lr=1e-3, 
    batch_size=128, 
    es=True, 
    min_es_iters=5000, 
    ref_es_iters=1000, 
    es_thresh=1e-3):
    
    # Data augmentation.
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ])

    train_dataset.transform = train_transform

    # Data Loader.
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    train_loader_iterator = iter(train_loader)

    # Optimizer.
    opt = torch.optim.Adam(vae.parameters(), lr=lr)

    # Learning rate scheduler.
    scheduler = ReduceLROnPlateau(opt, patience=10, factor=0.1, verbose=True)

    c_dim = vae.c_dim[0]  # number of classes

    tqdm_iter = tqdm(range(max_iters), desc="Iterations")
    tracker = defaultdict(list)
    for iter_idx in tqdm_iter:
        try:
            (x_batch, y_batch) = next(train_loader_iterator)
        except StopIteration:
            train_loader_iterator = iter(train_loader)
            (x_batch, y_batch) = next(train_loader_iterator)

        # Create one-hot encoding of labels for use as conditional input.
        c_batch = torch.zeros((y_batch.size(0), c_dim))
        c_batch[range(y_batch.size(0)), y_batch] = 1

        opt.zero_grad()
        elbo, exp_ll, kl = vae.elbo(x_batch.round(), c=c_batch, n=len(train_loader))  # Include condition in ELBO calculation
        (-elbo / len(train_dataset)).backward()

        # Gradient clipping.
        torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1)

        opt.step()

        # Update learning rate scheduler.
        scheduler.step(elbo)

        metrics = {"elbo": elbo.item(), "exp_ll": exp_ll.item(), "kl": kl.item()}

        for key, val in metrics.items():
            tracker[key].append(val)

        tqdm_iter.set_postfix(metrics)

        # Early stopping.
        if iter_idx > min_es_iters:
            curr_elbo = sum(tracker["elbo"][-100:]) / 50
            ref_elbo = sum(tracker["elbo"][-ref_es_iters - 100 : -ref_es_iters]) / 100
            if es and (curr_elbo - ref_elbo) < abs(es_thresh * ref_elbo):
                break
                
    return vae

def train_classifier(
    classifier,
    train_dataset,
    max_iters=10_000,
    lr=1e-3,
    batch_size=128,
    es=True,
    min_es_iters=5_000,
    ref_es_iters=1_000,
    es_thresh=1e-3,
):

    # Data Loader.
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    train_loader_iterator = iter(train_loader)

    # Optimiser.
    opt = torch.optim.Adam(classifier.parameters(), lr=lr)

    tqdm_iter = tqdm(range(max_iters), desc="Iterations")
    tracker = defaultdict(list)
    for iter_idx in tqdm_iter:
        try:
            (x_batch, y_batch) = next(train_loader_iterator)
        except StopIteration:
            train_loader_iterator = iter(train_loader)
            (x_batch, y_batch) = next(train_loader_iterator)

        opt.zero_grad()
        py_x = classifier(x_batch)
        exp_ll = py_x.log_prob(y_batch.long().unsqueeze(1)).sum()
        (-exp_ll / len(train_dataset)).backward()
        opt.step()

        metrics = {"exp_ll": exp_ll.item()}

        for key, val in metrics.items():
            tracker[key].append(val)

        tqdm_iter.set_postfix(metrics)

        # Early stopping.
        if iter_idx > min_es_iters:
            curr_elbo = sum(tracker["exp_ll"][-100:]) / 100
            ref_elbo = sum(tracker["exp_ll"][-ref_es_iters - 100 : -ref_es_iters]) / 100
            if es and (curr_elbo - ref_elbo) < abs(es_thresh * ref_elbo):
                break

def evaluate_classifier(
    classifier, 
    max_iters=10000, 
    lr=1e-3, 
    batch_size=128, 
    es=True, 
    min_es_iters=5000, 
    ref_es_iters=1000, 
    es_thresh=1e-3
    ):
    # Data Loader.
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader_iterator = iter(test_loader)

    # Optimiser.
    opt = torch.optim.Adam(classifier.parameters(), lr=lr)

    tqdm_iter = tqdm(range(len(test_loader)), desc="Iterations")
    tracker = defaultdict(list)
    with torch.no_grad():
        for iter_idx in tqdm_iter:
            try:
                (x_batch, y_batch) = next(test_loader_iterator)
            except StopIteration:
                break

            opt.zero_grad()
            py_x = classifier(x_batch)

            mll = py_x.log_prob(y_batch).mean()
            probs = py_x.probs
            acc = (y_batch == py_x.probs.argmax(-1)).float().mean()

            tracker["mll"].append(mll)
            tracker["acc"].append(acc)

    metrics = {
        "mll": torch.tensor(tracker["mll"]).mean(),
        "acc": torch.tensor(tracker["acc"]).mean(),
    }

    for name, val in metrics.items():
        print(f"{name}: {val:.3f}.")

    return metrics

def evaluate_classifiers(
    classifiers, 
    max_iters=10000, 
    lr=1e-3, 
    batch_size=128, 
    es=True, 
    min_es_iters=5000, 
    ref_es_iters=1000, 
    es_thresh=1e-3):
    # Optimizer.
    opt = torch.optim.Adam(vae.parameters(), lr=lr)

    # Data Loader.
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader_iterator = iter(test_loader)
    
    # Optimiser.
    opt = torch.optim.Adam(classifier.parameters(), lr=lr)

    tqdm_iter = tqdm(range(len(test_loader)), desc="Iterations")
    tracker = defaultdict(list)
    with torch.no_grad():
        for iter_idx in tqdm_iter:
            try:
                (x_batch, y_batch) = next(test_loader_iterator)
            except StopIteration:
                break

            opt.zero_grad()
            py_x = classifiers[0](x_batch).probs
            
            for classifier in classifiers[1:]:
                py_x += classifier(x_batch).probs

            py_x = py_x/len(classifiers)
            py_x = torch.distributions.categorical.Categorical(probs=py_x)
            
            mll = py_x.log_prob(y_batch).mean()
            probs = py_x.probs
            acc = (y_batch == py_x.probs.argmax(-1)).float().mean()

            tracker["mll"].append(mll)
            tracker["acc"].append(acc)

    metrics = {
        "mll": torch.tensor(tracker["mll"]).mean(),
        "acc": torch.tensor(tracker["acc"]).mean(),
    }

    for name, val in metrics.items():
        print(f"{name}: {val:.3f}.")

    return metrics



