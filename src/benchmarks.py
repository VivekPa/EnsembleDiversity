import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np

from .utils import *

import torch
from collections import defaultdict
from tqdm import tqdm

def train_NCL(classifiers, n, train_dataset,
              classification=True,
              lambda_corr=0.01,
              max_iters=10_000,
              lr=1e-3,
              batch_size=128,
              es=True,
              min_es_iters=5_000,
              ref_es_iters=1_000,
              es_thresh=1e-3):

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    train_loader_iterator = iter(train_loader)

    regular_loss = 0

    opt = torch.optim.Adam([p for classifier in classifiers for p in classifier.parameters()], lr=lr)

    tqdm_iter = tqdm(range(max_iters), desc="Iterations")
    tracker = defaultdict(list)
    for iter_idx in tqdm_iter:
        try:
            (x_batch, y_batch) = next(train_loader_iterator)
        except StopIteration:
            train_loader_iterator = iter(train_loader)
            (x_batch, y_batch) = next(train_loader_iterator)

        opt.zero_grad()

        corr_loss = 0
        outputs = []

        exp_ll_list = []
        for classifier in classifiers:
            py_x = classifier(x_batch)
            exp_ll = py_x.log_prob(y_batch).sum()
            exp_ll_list.append(exp_ll)

            # py_x_max = torch.argmax(py_x.probs, dim=1)
            # outputs.append(py_x_max.float())
            outputs.append(py_x.probs)

        for i in range(n):
            for j in range(i+1, n):
                corr = (1 / (n * (n - 1))) * torch.sum((outputs[i] - torch.mean(outputs[i])) * (outputs[j] - torch.mean(outputs[j])))
                corr_loss += corr

        regular_loss = -torch.mean(torch.stack(exp_ll_list)) / len(train_loader)
        loss = regular_loss + lambda_corr*corr_loss
        loss.backward()
        opt.step()

        metrics = {"regular_loss": regular_loss.item(), "corr_loss": corr_loss.item(), "total_loss": loss.item()}

        for key, val in metrics.items():
            tracker[key].append(val)

        tqdm_iter.set_postfix(metrics)

        if iter_idx > min_es_iters:
            curr_elbo = sum(tracker["total_loss"][-100:]) / 100
            ref_elbo = sum(tracker["total_loss"][-ref_es_iters - 100 : -ref_es_iters]) / 100
            if es and (curr_elbo - ref_elbo) < abs(es_thresh * ref_elbo):
                break

    return classifiers, tracker

def train_bagging(classifiers, n, train_dataset,
                 classification=True,
                 max_iters=10_000,
                 lr=1e-3,
                 batch_size=128,
                 es=True,
                 min_es_iters=5_000,
                 ref_es_iters=1_000,
                 es_thresh=1e-3):

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    tracker = defaultdict(list)

    for i, classifier in enumerate(classifiers):
        print(f"Training classifier {i+1}/{n}")
        
        # Create a bootstrap sample for the current classifier
        bootstrap_indices = torch.randint(0, len(train_dataset), (len(train_dataset),))
        bootstrap_dataset = torch.utils.data.Subset(train_dataset, bootstrap_indices)
        bootstrap_loader = torch.utils.data.DataLoader(bootstrap_dataset, batch_size=batch_size, shuffle=True)
        bootstrap_loader_iterator = iter(bootstrap_loader)

        opt = torch.optim.Adam(classifier.parameters(), lr=lr)
        
        tqdm_iter = tqdm(range(max_iters), desc="Iterations")
        for iter_idx in tqdm_iter:
            try:
                (x_batch, y_batch) = next(bootstrap_loader_iterator)
            except StopIteration:
                bootstrap_loader_iterator = iter(bootstrap_loader)
                (x_batch, y_batch) = next(bootstrap_loader_iterator)

            opt.zero_grad()

            py_x = classifier(x_batch)
            loss = -py_x.log_prob(y_batch).sum() / batch_size
            loss.backward()
            opt.step()

            metrics = {"loss": loss.item()}

            for key, val in metrics.items():
                tracker[f"classifier_{i}_{key}"].append(val)

            tqdm_iter.set_postfix(metrics)

            if iter_idx > min_es_iters:
                curr_elbo = sum(tracker[f"classifier_{i}_loss"][-100:]) / 100
                ref_elbo = sum(tracker[f"classifier_{i}_loss"][-ref_es_iters - 100 : -ref_es_iters]) / 100
                if es and (curr_elbo - ref_elbo) < abs(es_thresh * ref_elbo):
                    break

    return classifiers, tracker

def train_AdaBoost(classifiers, n, train_dataset,
                   torch_dataset=False,
                   classification=True,
                   max_iters=10_000,
                   lr=1e-3,
                   batch_size=128,
                   es=True,
                   min_es_iters=5_000,
                   ref_es_iters=1_000,
                   es_thresh=1e-3):

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=False
    )

    tracker = defaultdict(list)
    n_samples = len(train_dataset)
    sample_weights = np.ones(n_samples) / n_samples

    for i, classifier in enumerate(classifiers):
        print(f"Training classifier {i+1}/{n}")

        opt = torch.optim.Adam(classifier.parameters(), lr=lr)

        # Train the current classifier with the current sample weights
        for iter_idx in range(max_iters):
            sample_indices = np.random.choice(n_samples, size=batch_size, p=sample_weights)

            x_batch, y_batch = [], []
            for index in sample_indices:
                x, y = train_dataset[index]
                x_batch.append(x)
                y_batch.append(y)

            x_batch = torch.stack(x_batch)
            y_batch = torch.tensor(y_batch)

            # x_batch, y_batch = train_dataset[sample_indices]

            opt.zero_grad()

            py_x = classifier(x_batch)
            loss = -py_x.log_prob(y_batch.long().unsqueeze(1)).sum() / batch_size
            loss.backward()
            opt.step()

            if iter_idx > min_es_iters:
                # Early stopping condition
                curr_loss = sum(tracker[f"classifier_{i}_loss"][-100:]) / 100
                ref_loss = sum(tracker[f"classifier_{i}_loss"][-ref_es_iters - 100 : -ref_es_iters]) / 100
                if es and (curr_loss - ref_loss) < abs(es_thresh * ref_loss):
                    break

        # Compute classifier's performance and update sample weights
        with torch.no_grad():
            all_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
            x_all = []
            y_all = []

            for x_batch, y_batch in all_data_loader:
                x_all.append(x_batch)
                y_all.append(y_batch)

            x_all = torch.cat(x_all)
            y_all = torch.cat(y_all)

            # x_all, y_all = train_dataset[:]
            py_x_all = classifier(x_all)
            y_pred = torch.argmax(py_x_all.probs, dim=1)
            incorrect = (y_pred != y_all).float()

            error = np.dot(sample_weights, incorrect.numpy()) / np.sum(sample_weights)
            classifier_weight = np.log((1 - error) / error) + np.log(n - 1)

            # Update sample weights
            sample_weights *= np.exp(classifier_weight * incorrect.numpy())
            sample_weights /= np.sum(sample_weights)

            # Save classifier weight
            tracker[f"classifier_{i}_weight"].append(classifier_weight)

    return classifiers, tracker



