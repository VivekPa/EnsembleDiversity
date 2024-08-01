import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np

from .utils import *

import torch
from collections import defaultdict
from tqdm import tqdm


def train_LIT(classifiers, n, train_dataset,
              vae=None,
              compute_manifold=False,
              torch_dataset=False,
              grad_quantity='binary_logit_input_gradients',
              classification=True,
              lambda_overlap=0.01,
              max_iters=10_000,
              lr=1e-3,
              batch_size=128,
              es=True,
              min_es_iters=5_000,
              ref_es_iters=1_000,
              es_thresh=1e-3):
    
    # y = train_dataset[:][1]

    # if classification:
    #     if len(y.shape) == 1:
    #         y = onehot(y)

    #     if y.shape[1] > 2 and grad_quantity == 'binary_logit_input_gradients':
    #         grad_quantity = 'cross_entropy_input_gradients'

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

        igrads = [m.input_grad(x_batch) for m in classifiers]
        diverse_loss = sum([torch.sqrt(torch.sum(squared_cos_sim(igrads[i], igrads[j])))
                        for i in range(n)
                        for j in range(i+1, n)]) * lambda_overlap

        manifold_loss = 0
        if compute_manifold:
            for i in range(len(x_batch)):
                x = x_batch[i].unsqueeze(0)
                mgrads = []
                for m in classifiers:
                    input_grad = m.input_grad(x).view(-1, 1)
                    J = vae.manifold_jac(x).view(vae.z_shape[0], -1)
                    manifold_grad = J @ input_grad
                    mgrads.append(manifold_grad)

                x_manifold_loss = sum([torch.sqrt(torch.sum(squared_cos_sim(mgrads[i], mgrads[j])))
                                for i in range(n)
                                for j in range(i+1, n)]) * lambda_overlap
                manifold_loss += x_manifold_loss

        exp_ll_list = []
        for classifier in classifiers:
            py_x = classifier(x_batch)
            # exp_ll = py_x.log_prob(y_batch.long().unsqueeze(1)).sum()
            exp_ll = py_x.log_prob(y_batch).sum()
            exp_ll_list.append(exp_ll)

        regular_loss = -torch.mean(torch.stack(exp_ll_list)) / batch_size
        loss = regular_loss + diverse_loss
        loss.backward()
        opt.step()

        if compute_manifold:
            metrics = {"regular_loss": regular_loss.item(), "diverse_loss": diverse_loss.item(), "total_loss": regular_loss.item() + diverse_loss.item(), "manifold_loss": manifold_loss}
        else:
            metrics = {"regular_loss": regular_loss.item(), "diverse_loss": diverse_loss.item(), "total_loss": regular_loss.item() + diverse_loss.item()}

        for key, val in metrics.items():
            tracker[key].append(val)

        tqdm_iter.set_postfix(metrics)

        if iter_idx > min_es_iters:
            curr_elbo = sum(tracker["total_loss"][-100:]) / 100
            ref_elbo = sum(tracker["total_loss"][-ref_es_iters - 100 : -ref_es_iters]) / 100
            if es and (curr_elbo - ref_elbo) < abs(es_thresh * ref_elbo):
                break

    # for classifier in classifiers:
    #     classifier.vals = [p.detach().numpy() for p in classifier.parameters()]

    return classifiers, tracker

def train_manifold_LIT(classifiers, n, train_dataset,
              vae,
              torch_dataset=False,
              grad_quantity='binary_logit_input_gradients',
              classification=True,
              lambda_overlap=0.01,
              max_iters=10_000,
              lr=1e-3,
              batch_size=128,
              es=True,
              min_es_iters=5_000,
              ref_es_iters=1_000,
              es_thresh=1e-3):
    
    # y = train_dataset[:][1]

    # if classification:
    #     if len(y.shape) == 1:
    #         y = onehot(y)

    #     if y.shape[1] > 2 and grad_quantity == 'binary_logit_input_gradients':
    #         grad_quantity = 'cross_entropy_input_gradients'

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

        igrads = [m.input_grad(x_batch) for m in classifiers]
        diverse_loss = sum([torch.sqrt(torch.sum(squared_cos_sim(igrads[i], igrads[j])))
                        for i in range(n)
                        for j in range(i+1, n)]) * lambda_overlap

        manifold_loss = 0
        for i in range(len(x_batch)):
            x = x_batch[i].unsqueeze(0)
            mgrads = []
            for m in classifiers:
                input_grad = m.input_grad(x).view(-1, 1)
                J = vae.manifold_jac(x).view(vae.z_shape[0], -1)
                manifold_grad = J @ input_grad
                mgrads.append(manifold_grad)

            x_manifold_loss = sum([torch.sqrt(torch.sum(squared_cos_sim(mgrads[i], mgrads[j])))
                            for i in range(n)
                            for j in range(i+1, n)]) * lambda_overlap
            manifold_loss += x_manifold_loss

        manifold_loss = manifold_loss / len(train_loader)

        exp_ll_list = []
        for classifier in classifiers:
            py_x = classifier(x_batch)
            # exp_ll = py_x.log_prob(y_batch.long().unsqueeze(1)).sum()
            exp_ll = py_x.log_prob(y_batch).sum()
            exp_ll_list.append(exp_ll)

        regular_loss = -torch.mean(torch.stack(exp_ll_list)) / len(train_loader)
        loss = regular_loss + manifold_loss*lambda_overlap
        loss.backward()
        opt.step()

        metrics = {"regular_loss": regular_loss.item(),  "manifold_loss": manifold_loss.item(), "total_loss": loss.item()}

        for key, val in metrics.items():
            tracker[key].append(val)

        tqdm_iter.set_postfix(metrics)

        if iter_idx > min_es_iters:
            curr_elbo = sum(tracker["total_loss"][-100:]) / 100
            ref_elbo = sum(tracker["total_loss"][-ref_es_iters - 100 : -ref_es_iters]) / 100
            if es and (curr_elbo - ref_elbo) < abs(es_thresh * ref_elbo):
                break

    return classifiers, tracker

def train_LIT_original(classifiers, n, train_dataset,
              vae=None,
              compute_manifold=False,
              torch_dataset=False,
              grad_quantity='binary_logit_input_gradients',
              classification=True,
              lambda_overlap=0.01,
              max_iters=10_000,
              lr=1e-3,
              batch_size=128,
              es=True,
              min_es_iters=5_000,
              ref_es_iters=1_000,
              es_thresh=1e-3):
    
    # y = train_dataset[:][1]

    # if classification:
    #     if len(y.shape) == 1:
    #         y = onehot(y)

    #     if y.shape[1] > 2 and grad_quantity == 'binary_logit_input_gradients':
    #         grad_quantity = 'cross_entropy_input_gradients'

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

        igrads = [m.input_grad(x_batch) for m in classifiers]
        diverse_loss = sum([torch.sum(squared_cos_sim(igrads[i], igrads[j]))
                        for i in range(n)
                        for j in range(i+1, n)]) * lambda_overlap

        manifold_loss = 0
        if compute_manifold:
            for i in range(len(x_batch)):
                x = x_batch[i].unsqueeze(0)
                mgrads = []
                for m in classifiers:
                    input_grad = m.input_grad(x).view(-1, 1)
                    J = vae.manifold_jac(x).view(vae.z_shape[0], -1)
                    manifold_grad = J @ input_grad
                    mgrads.append(manifold_grad)

                x_manifold_loss = sum([torch.sum(squared_cos_sim(mgrads[i], mgrads[j]))
                                for i in range(n)
                                for j in range(i+1, n)]) * lambda_overlap
                manifold_loss += x_manifold_loss

        exp_ll_list = []
        for classifier in classifiers:
            py_x = classifier(x_batch)
            exp_ll = py_x.log_prob(y_batch.long().unsqueeze(1)).sum()
            exp_ll_list.append(exp_ll)

        regular_loss = -torch.mean(torch.stack(exp_ll_list)) / batch_size
        loss = regular_loss + diverse_loss
        loss.backward()
        opt.step()

        if compute_manifold:
            metrics = {"regular_loss": regular_loss.item(), "diverse_loss": diverse_loss.item(), "total_loss": regular_loss.item() + diverse_loss.item(), "manifold_loss": manifold_loss}
        else:
            metrics = {"regular_loss": regular_loss.item(), "diverse_loss": diverse_loss.item(), "total_loss": regular_loss.item() + diverse_loss.item()}

        for key, val in metrics.items():
            tracker[key].append(val)

        tqdm_iter.set_postfix(metrics)

        if iter_idx > min_es_iters:
            curr_elbo = sum(tracker["total_loss"][-100:]) / 100
            ref_elbo = sum(tracker["total_loss"][-ref_es_iters - 100 : -ref_es_iters]) / 100
            if es and (curr_elbo - ref_elbo) < abs(es_thresh * ref_elbo):
                break

    # for classifier in classifiers:
    #     classifier.vals = [p.detach().numpy() for p in classifier.parameters()]

    return classifiers, tracker


