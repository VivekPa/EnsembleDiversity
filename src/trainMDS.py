import logging
from collections import defaultdict
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import nn
from torch.distributions import Categorical

from tqdm.auto import tqdm

from .classifiers import Classifier
from .vae import VAE
from .utils import *

from pytorch_pretrained_biggan import (one_hot_from_int, truncated_noise_sample)

from .mds import *

def train_mds(
    train_dataset,
    vae,
    classifiers,
    divergence_measure: Callable = torch.distributions.kl_divergence,
    M: int = 128,
    N: int = 128,
    max_iters: int = 10_000,
    lr: float = 1e-3,
    early_stopping: bool = True,
    min_es_iters: int = 5000,
    ref_es_iters: int = 1000,
    es_thresh: float = 1e-3,
    prior_lambda: float = 0,
    mds_lambda: float = 1
) -> Tuple[torch.Tensor, Dict[str, Any]]:

    # # Ensure all parameters of vae and non-training classifiers are detached.
    # for module in [vae]:
    #     for param in module.parameters():
    #         param.requires_grad = False

    for module in classifiers:
        for param in module.parameters():
            param.requires_grad = True

    # Data Loader.
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=N, shuffle=True
    )
    train_loader_iterator = iter(train_loader)

    # Construct optimisation.
    opt = torch.optim.Adam([p for classifier in classifiers for p in classifier.parameters()], lr=lr)

    tracker = defaultdict(list)

    tqdm_iter = tqdm(range(max_iters), desc="Iterations")
    tracker = defaultdict(list)
    for iter_idx in tqdm_iter:
        try:
            (x_batch, y_batch) = next(train_loader_iterator)
        except StopIteration:
            train_loader_iterator = iter(train_loader)
            (x_batch, y_batch) = next(train_loader_iterator)

        objective = 0
        exp_ll_sum = 0
        divergence_sum = 0

        z, class_int_arr, trackers = generate_mds_sets(
            vae,
            classifiers,
            prior_lambda=1e-4,
            conditional=True
        )

        with torch.no_grad():
            x_mds = vae(z, class_vector, 1)

        py_x_set = []
        for classifier in classifiers:
            py_x = classifier(x_batch)
            exp_ll = py_x.log_prob(y_batch).sum()
            exp_ll_sum += exp_ll/N

            py_x_set.append(py_x)

            py_x_mds = classifier(x_mds)
            exp_ll_mds = py_x_mds.log_prob(torch.Tensor(class_int_arr))
            exp_ll_sum += lambda_mds*exp_ll_mds/M

        js_div = JS_divergence(py_x_set)
        divergence_sum += js_div

        objective = (-exp_ll - divergence_sum)

        objective.backward()
        opt.step()

        metrics = {
            "exp_ll": exp_ll_sum.item(),
            "divergence": divergence_sum.item(),
            "objective": objective.item()}

        for key, val in metrics.items():
            tracker[key].append(val)

        tqdm_iter.set_postfix(metrics)

        # Early stopping.
        if iter_idx > min_es_iters:
            curr_elbo = sum(tracker["objective"][-100:]) / 100
            ref_elbo = sum(tracker["objective"][-ref_es_iters - 100 : -ref_es_iters]) / 100
            if es and (curr_elbo - ref_elbo) < abs(es_thresh * ref_elbo):
                break

    return classifiers, tracker

def train_mds_ImageNet(
    train_dataset,
    vae,
    classifiers,
    divergence_measure: Callable = torch.distributions.kl_divergence,
    M: int = 128,
    N: int = 128,
    max_iters: int = 10_000,
    lr: float = 1e-3,
    early_stopping: bool = True,
    min_es_iters: int = 5000,
    ref_es_iters: int = 1000,
    es_thresh: float = 1e-3,
    prior_lambda: float = 0,
    mds_lambda: float = 1
) -> Tuple[torch.Tensor, Dict[str, Any]]:

    # # Ensure all parameters of vae and non-training classifiers are detached.
    # for module in [vae]:
    #     for param in module.parameters():
    #         param.requires_grad = False

    for module in classifiers:
        for param in module.parameters():
            param.requires_grad = True

    # Data Loader.
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=N, shuffle=True
    )
    train_loader_iterator = iter(train_loader)

    # Construct optimisation.
    opt = torch.optim.Adam([p for classifier in classifiers for p in classifier.parameters()], lr=lr)

    tracker = defaultdict(list)

    tqdm_iter = tqdm(range(max_iters), desc="Iterations")
    tracker = defaultdict(list)
    for iter_idx in tqdm_iter:
        try:
            (x_batch, y_batch) = next(train_loader_iterator)
        except StopIteration:
            train_loader_iterator = iter(train_loader)
            (x_batch, y_batch) = next(train_loader_iterator)

        objective = 0
        exp_ll_sum = 0
        divergence_sum = 0

        z, class_int_arr, trackers = generate_mds_sets_pretrained(
            vae,
            classifiers
        )

        class_vector = one_hot_from_int(class_int_arr, batch_size=M)
        class_vector = torch.from_numpy(class_vector)

        with torch.no_grad():
            x_mds = vae(z, class_vector, 1)

        py_x_set = []
        for classifier in classifiers:
            py_x = classifier(x_batch)
            exp_ll = py_x.log_prob(y_batch).sum()
            exp_ll_sum += exp_ll/N

            py_x_set.append(py_x)

            py_x_mds = classifier(x_mds)
            exp_ll_mds = py_x_mds.log_prob(torch.Tensor(class_int_arr))
            exp_ll_sum += lambda_mds*exp_ll_mds/M

        js_div = JS_divergence(py_x_set)
        divergence_sum += js_div

        objective = (-exp_ll - divergence_sum)

        objective.backward()
        opt.step()

        metrics = {
            "exp_ll": exp_ll_sum.item(),
            "divergence": divergence_sum.item(),
            "objective": objective.item()}

        for key, val in metrics.items():
            tracker[key].append(val)

        tqdm_iter.set_postfix(metrics)

        # Early stopping.
        if iter_idx > min_es_iters:
            curr_elbo = sum(tracker["objective"][-100:]) / 100
            ref_elbo = sum(tracker["objective"][-ref_es_iters - 100 : -ref_es_iters]) / 100
            if es and (curr_elbo - ref_elbo) < abs(es_thresh * ref_elbo):
                break

    return classifiers, tracker

def train_mds_single(
    train_dataset,
    vae: VAE,
    classifier1: Classifier,
    classifier2: nn.Module,
    divergence_measure: Callable = torch.distributions.kl_divergence,
    M: int = 128,
    N: int = 128,
    max_iters: int = 10_000,
    lr: float = 1e-3,
    early_stopping: bool = True,
    min_es_iters: int = 5000,
    ref_es_iters: int = 1000,
    es_thresh: float = 1e-3,
    prior_lambda: float = 0,
    mds_lambda: float = 1
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    
    # Perform dimensionality checks.
    assert (
        classifier1.output_dim == classifier2.output_dim
    ), "Classifiers have different output dimensions."
    assert (
        vae.x_shape == classifier1.input_shape
    ), "Classifier input dimensionality should be the same as vae data dimensionality."
    assert (
        vae.x_shape == classifier2.input_shape
    ), "Classifier input dimensionality should be the same as vae data dimensionality."

    # Ensure all parameters of vae and non-training classifiers are detached.
    for module in [vae, classifier2]:
        for param in module.parameters():
            param.requires_grad = False

    for param in classifier1.parameters():
        param.requires_grad = True

    # Data Loader.
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=N, shuffle=True
    )
    train_loader_iterator = iter(train_loader)

    # Construct optimisation.
    opt = torch.optim.Adam(classifier1.parameters(), lr=lr)

    tracker = defaultdict(list)

    tqdm_iter = tqdm(range(max_iters), desc="Iterations")
    tracker = defaultdict(list)
    for iter_idx in tqdm_iter:
        try:
            (x_batch, y_batch) = next(train_loader_iterator)
        except StopIteration:
            train_loader_iterator = iter(train_loader)
            (x_batch, y_batch) = next(train_loader_iterator)

        for param in classifier1.parameters():
            param.requires_grad = False

        z_mds, mds_tracker = generate_mds(
            vae, 
            classifier1, 
            classifier2, 
            divergence_measure, 
            num_samples=M,
            prior_lambda=prior_lambda)

        for param in classifier1.parameters():
            param.requires_grad = True

        x_mds = vae.decode(z_mds)

        igrads = [m.input_grad(x_batch) for m in [classifier1, classifier2]]
        diverse_loss = sum([torch.sqrt(torch.sum(squared_cos_sim(igrads[0], igrads[1])))])

        opt.zero_grad()
        py_x = classifier1(x_batch)
        exp_ll = py_x.log_prob(y_batch.long().unsqueeze(1)).sum()/N

        classifier1_py_x = classifier1(x_mds)
        classifier2_py_x = classifier2(x_mds)
        divergence = divergence_measure(classifier1_py_x, classifier2_py_x).sum()/M
        objective = -exp_ll - mds_lambda*divergence

        objective.backward()

        # for param in classifier1.parameters():
        #     print(param.grad)
        opt.step()

        metrics = {
            "exp_ll": exp_ll.item(),
            "divergence": divergence.item(),
            "objective": objective.item(),
            "LIT_loss": diverse_loss.item()}

        for key, val in metrics.items():
            tracker[key].append(val)

        tqdm_iter.set_postfix(metrics)

        # Early stopping.
        if iter_idx > min_es_iters:
            curr_elbo = sum(tracker["objective"][-100:]) / 100
            ref_elbo = sum(tracker["objective"][-ref_es_iters - 100 : -ref_es_iters]) / 100
            if es and (curr_elbo - ref_elbo) < abs(es_thresh * ref_elbo):
                break

    return classifier1, classifier2, tracker

def train_mds_pairs(
    train_dataset,
    vae: VAE,
    classifier1: Classifier,
    classifier2: nn.Module,
    divergence_measure: Callable = torch.distributions.kl_divergence,
    M: int = 128,
    N: int = 128,
    max_iters: int = 10_000,
    lr: float = 1e-3,
    early_stopping: bool = True,
    min_es_iters: int = 5000,
    ref_es_iters: int = 1000,
    es_thresh: float = 1e-3,
    prior_lambda: float = 0,
    mds_lambda: float = 1
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    
    # Perform dimensionality checks.
    assert (
        classifier1.output_dim == classifier2.output_dim
    ), "Classifiers have different output dimensions."
    assert (
        vae.x_shape == classifier1.input_shape
    ), "Classifier input dimensionality should be the same as vae data dimensionality."
    assert (
        vae.x_shape == classifier2.input_shape
    ), "Classifier input dimensionality should be the same as vae data dimensionality."

    # Ensure all parameters of vae and non-training classifiers are detached.
    for module in [vae]:
        for param in module.parameters():
            param.requires_grad = False

    for module in [classifier1, classifier2]:
        for param in module.parameters():
            param.requires_grad = True

    # Data Loader.
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=N, shuffle=True
    )
    train_loader_iterator = iter(train_loader)

    # Construct optimisation.
    opt = torch.optim.Adam([classifier1.parameters(), classifier2.parameters()], lr=lr)

    tracker = defaultdict(list)

    tqdm_iter = tqdm(range(max_iters), desc="Iterations")
    tracker = defaultdict(list)
    for iter_idx in tqdm_iter:
        try:
            (x_batch, y_batch) = next(train_loader_iterator)
        except StopIteration:
            train_loader_iterator = iter(train_loader)
            (x_batch, y_batch) = next(train_loader_iterator)

        for module in [classifier1, classifier2]:
            for param in module.parameters():
                param.requires_grad = False

        z_mds, mds_tracker = generate_mds(
            vae, 
            classifier1, 
            classifier2, 
            divergence_measure, 
            num_samples=M,
            prior_lambda=prior_lambda)

        for module in [classifier1, classifier2]:
            for param in module.parameters():
                param.requires_grad = True

        x_mds = vae.decode(z_mds)

        igrads = [m.input_grad(x_batch) for m in [classifier1, classifier2]]
        diverse_loss = sum([torch.sqrt(torch.sum(squared_cos_sim(igrads[0], igrads[1])))])

        opt.zero_grad()
        py_x = classifier1(x_batch)
        exp_ll = py_x.log_prob(y_batch.long().unsqueeze(1)).sum()/N

        classifier1_py_x = classifier1(x_mds)
        classifier2_py_x = classifier2(x_mds)
        divergence = divergence_measure(classifier1_py_x, classifier2_py_x).sum()/M
        objective = -exp_ll - mds_lambda*divergence

        objective.backward()

        # for param in classifier1.parameters():
        #     print(param.grad)
        opt.step()

        metrics = {
            "exp_ll": exp_ll.item(),
            "divergence": divergence.item(),
            "objective": objective.item(),
            "LIT_loss": diverse_loss.item()}

        for key, val in metrics.items():
            tracker[key].append(val)

        tqdm_iter.set_postfix(metrics)

        # Early stopping.
        if iter_idx > min_es_iters:
            curr_elbo = sum(tracker["objective"][-100:]) / 100
            ref_elbo = sum(tracker["objective"][-ref_es_iters - 100 : -ref_es_iters]) / 100
            if es and (curr_elbo - ref_elbo) < abs(es_thresh * ref_elbo):
                break

    return classifier1, classifier2, tracker

def train_mds_set(
    train_dataset,
    vae: VAE,
    classifiers,
    divergence_measure: Callable = torch.distributions.kl_divergence,
    M: int = 128,
    N: int = 128,
    max_iters: int = 10_000,
    lr: float = 1e-3,
    early_stopping: bool = True,
    min_es_iters: int = 5000,
    ref_es_iters: int = 1000,
    es_thresh: float = 1e-3,
    prior_lambda: float = 0,
    mds_lambda: float = 1
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    
    # # Perform dimensionality checks.
    # for classifier1 in classifiers:
    #     for classifier2 in classifiers:
    #            assert (
    #             classifier1.output_dim == classifier2.output_dim
    #         ) "Classifiers have different output dimensions."
    # assert (
    #     vae.x_shape == classifier1.input_shape
    # ), "Classifier input dimensionality should be the same as vae data dimensionality."
    # assert (
    #     vae.x_shape == classifier2.input_shape
    # ), "Classifier input dimensionality should be the same as vae data dimensionality."

    # Ensure all parameters of vae and non-training classifiers are detached.
    for module in [vae]:
        for param in module.parameters():
            param.requires_grad = False

    for module in classifiers:
        for param in module.parameters():
            param.requires_grad = True

    # Data Loader.
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=N, shuffle=True
    )
    train_loader_iterator = iter(train_loader)

    # Construct optimisation.
    opt = torch.optim.Adam([p for classifier in classifiers for p in classifier.parameters()], lr=lr)

    tracker = defaultdict(list)

    tqdm_iter = tqdm(range(max_iters), desc="Iterations")
    tracker = defaultdict(list)
    for iter_idx in tqdm_iter:
        try:
            (x_batch, y_batch) = next(train_loader_iterator)
        except StopIteration:
            train_loader_iterator = iter(train_loader)
            (x_batch, y_batch) = next(train_loader_iterator)

        objective = 0
        exp_ll_sum = 0
        divergence_sum = 0

        for classifier1 in classifiers:
            for classifier2 in classifiers:
                if classifier1 == classifier2:
                    pass
                else:
                    # for module in [classifier1, classifier2]:
                    #     for param in module.parameters():
                    #         param.requires_grad = False

                    x_mds = vae.decode(vae.pz.rsample(sample_shape=torch.Size([M])))

                    opt.zero_grad()

                    classifier1_py_x = classifier1(x_mds)
                    classifier2_py_x = classifier2(x_mds)
                    # mean_py_x = 
                    divergence = divergence_measure(classifier1_py_x, classifier2_py_x).sum()/M
                    objective += -mds_lambda*divergence

                    divergence_sum += divergence

            py_x = classifier1(x_batch)
            exp_ll = py_x.log_prob(y_batch).sum()/N
            objective += -exp_ll

            exp_ll_sum += exp_ll

        objective.backward()
        opt.step()

        metrics = {
            "exp_ll": exp_ll_sum.item(),
            "divergence": divergence_sum.item(),
            "objective": objective.item()}

        for key, val in metrics.items():
            tracker[key].append(val)

        tqdm_iter.set_postfix(metrics)

        # Early stopping.
        if iter_idx > min_es_iters:
            curr_elbo = sum(tracker["objective"][-100:]) / 100
            ref_elbo = sum(tracker["objective"][-ref_es_iters - 100 : -ref_es_iters]) / 100
            if es and (curr_elbo - ref_elbo) < abs(es_thresh * ref_elbo):
                break

    return classifiers, tracker
