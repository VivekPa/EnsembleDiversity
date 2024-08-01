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


def generate_mds(
    vae: VAE,
    classifier1: Classifier,
    classifier2: nn.Module,
    divergence_measure: Callable = torch.distributions.kl_divergence,
    z_init: Optional[torch.Tensor] = None,
    num_samples: int = 1,
    max_iters: int = 10_000,
    lr: float = 1e-2,
    early_stopping: bool = True,
    min_es_iters: int = 1000,
    ref_es_iters: int = 100,
    es_thresh: float = 1e-2,
    prior_lambda: float = 0,
    conditional: bool = False,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Generates maximally different samples w.r.t. classifier1 and classifier2 by
    optimising the latent space of the vae.

    Args:
        vae (nn.Module): VAE.
        classifier1 (nn.Module): Classifier 1.
        classifier2 (nn.Module): Classifier 2.
        divergence (Callable): Divergence metric used to evaluate the differences in
            classification of the two.
        z_init (Optional[torch.Tensor], optional): Initial z. Defaults to None.

    Returns:
        torch.Tensor: A maximally different sample.
    """
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

    # Ensure all parameters of vae and classifiers are detached.
    for module in [vae, classifier1, classifier2]:
        for param in module.parameters():
            param.requires_grad = False

    # Construct optimisation.
    if z_init is None:
        z_init = torch.randn((num_samples, vae.z_shape[0]), device=vae.device)
    else:
        if len(z_init.shape) == 1:
            z_init = z_init.unsqueeze(0)

    z = nn.Parameter(z_init)
    opt = torch.optim.Adam([z], lr=lr)

    tracker = defaultdict(list)

    iter_tqdm = tqdm(range(max_iters), desc="iterations")
    logging.info("Beginning optimisation.")
    for iter_idx in iter_tqdm:
        opt.zero_grad()

        x = vae.decode(z)
        classifier1_py_x = classifier1(x)
        classifier2_py_x = classifier2(x)
        divergence = divergence_measure(classifier1_py_x, classifier2_py_x).sum()
        prior = vae.pz.log_prob(z)
        prior_sum = prior_lambda*prior.sum()/len(z)
        objective = divergence + prior_sum

        (-objective).backward()
        opt.step()

        # Track metrics.
        tracker["divergence"].append(divergence.item())
        tracker['z'].append(z.clone().detach().numpy())
        tracker['objective'].append(objective.item())
        iter_tqdm.set_postfix({"divergence": divergence.item()})

        # Check convergence.
        if iter_idx > min_es_iters:
            curr_divergence = sum(tracker["divergence"][-50:]) / 50
            ref_divergence = (
                sum(tracker["divergence"][-ref_es_iters - 50 : -ref_es_iters]) / 50
            )
            if early_stopping and (curr_divergence - ref_divergence) < abs(
                es_thresh * ref_divergence
            ):
                break

    return z.detach(), tracker

def generate_mds_pretrained(
    vae: nn.Module,
    clf1: nn.Module,
    clf1_weights,
    clf2: nn.Module,
    clf2_weights,
    divergence_measure: Callable = torch.distributions.kl_divergence,
    z_init: Optional[torch.Tensor] = None,
    z_shape: int = 128,
    class_int: int = 0,
    num_samples: int = 1,
    max_iters: int = 10_000,
    lr: float = 1e-2,
    early_stopping: bool = True,
    min_es_iters: int = 1000,
    ref_es_iters: int = 100,
    es_thresh: float = 1e-2,
    use_mps: bool = False
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    
    # Ensure all parameters of vae and classifiers are detached.
    for module in [vae, clf1, clf2]:
        for param in module.parameters():
            param.requires_grad = False

    # Construct optimisation.
    if z_init is None:
        z_init = torch.randn((num_samples, z_shape))
    else:
        if len(z_init.shape) == 1:
            z_init = z_init.unsqueeze(0)
        z_init = z_init.to(torch.float32)

    z = nn.Parameter(z_init)
    z_init_p = nn.Parameter(z_init.clone())

    opt = torch.optim.Adam([z], lr=lr)

    class_vector = one_hot_from_int(class_int, batch_size=1)
    class_vector = torch.from_numpy(class_vector)

    if use_mps:
        z_init = z_init.to("mps")
        z = z.to("mps")
        vae = vae.to("mps")
        class_vector = class_vector.to("mps")
        clf1 = clf1.to("mps")
        clf2 = clf2.to("mps")

    tracker = defaultdict(list)

    iter_tqdm = tqdm(range(max_iters), desc="iterations")
    logging.info("Beginning optimisation.")
    for iter_idx in iter_tqdm:
        opt.zero_grad()

        x = vae(z, class_vector, truncation=1)[0]

        clf1_preprocess = clf1_weights.transforms()
        clf1_batch = clf1_preprocess(x).unsqueeze(0)
        clf1_preds = clf1(clf1_batch).squeeze(0).softmax(0)
        clf1_py_x = Categorical(probs=clf1_preds)
        
        clf2_preprocess = clf2_weights.transforms()
        clf2_batch = clf2_preprocess(x).unsqueeze(0)
        clf2_preds = clf2(clf2_batch).squeeze(0).softmax(0)
        clf2_py_x = Categorical(probs=clf2_preds)
        
        divergence = divergence_measure(clf1_py_x, clf2_py_x).sum()

        (-divergence).backward()
        opt.step()

        # Track metrics.
        tracker["divergence"].append(divergence.item())
        tracker['z'].append(z.clone().detach().numpy()[0])
        iter_tqdm.set_postfix({"divergence": divergence.item()})

        # Check convergence.
        if iter_idx > min_es_iters:
            curr_divergence = sum(tracker["divergence"][-50:]) / 50
            ref_divergence = (
                sum(tracker["divergence"][-ref_es_iters - 50 : -ref_es_iters]) / 50
            )
            if early_stopping and (curr_divergence - ref_divergence) < abs(
                es_thresh * ref_divergence
            ):
                break
    return z.detach(), z_init_p.detach(), tracker

def generate_mds_sets(
    vae: VAE,
    classifiers,
    divergence_measure: Callable = JS_divergence,
    z_init: Optional[torch.Tensor] = None,
    num_samples: int = 1,
    max_iters: int = 10_000,
    lr: float = 1e-2,
    early_stopping: bool = True,
    min_es_iters: int = 1000,
    ref_es_iters: int = 100,
    es_thresh: float = 1e-2,
    prior_lambda: float = 0,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Generates maximally different samples w.r.t. classifier1 and classifier2 by
    optimising the latent space of the vae.

    Args:
        vae (nn.Module): VAE.
        classifier1 (nn.Module): Classifier 1.
        classifier2 (nn.Module): Classifier 2.
        divergence (Callable): Divergence metric used to evaluate the differences in
            classification of the two.
        z_init (Optional[torch.Tensor], optional): Initial z. Defaults to None.

    Returns:
        torch.Tensor: A maximally different sample.
    """
    # # Perform dimensionality checks.
    # assert (
    #     classifier1.output_dim == classifier2.output_dim
    # ), "Classifiers have different output dimensions."
    # assert (
    #     vae.x_shape == classifier1.input_shape
    # ), "Classifier input dimensionality should be the same as vae data dimensionality."
    # assert (
    #     vae.x_shape == classifier2.input_shape
    # ), "Classifier input dimensionality should be the same as vae data dimensionality."

    # Ensure all parameters of vae and classifiers are detached.
    for module in [vae]:
        for param in module.parameters():
            param.requires_grad = False

    for module in classifiers:
        for param in module.parameters():
            param.require_grad = False

    # Construct optimisation.
    if z_init is None:
        z_init = torch.randn((num_samples, vae.z_shape[0]), device=vae.device)
    else:
        if len(z_init.shape) == 1:
            z_init = z_init.unsqueeze(0)

    z = nn.Parameter(z_init)
    opt = torch.optim.Adam([z], lr=lr)

    class_int_arr = np.random.randint(0, 999, M)
    class_vector = one_hot_from_int(class_int_arr, batch_size=num_samples)

    class_vector = torch.from_numpy(class_vector)

    tracker = defaultdict(list)

    iter_tqdm = tqdm(range(max_iters), desc="iterations")
    logging.info("Beginning optimisation.")
    for iter_idx in iter_tqdm:
        opt.zero_grad()

        x = vae.decode(z, class_vector)
        classifiers_py_x = [clf(x) for clf in classifiers]
        divergence = divergence_measure(classifiers_py_x).sum()/num_samples
        prior = vae.pz.log_prob(z)
        prior_sum = prior_lambda*prior.sum()/num_samples
        objective = divergence + prior_sum

        (-objective).backward()
        opt.step()

        # Track metrics.
        tracker["divergence"].append(divergence.item())
        tracker['z'].append(z.clone().detach().numpy())
        tracker['objective'].append(objective.item())
        iter_tqdm.set_postfix({"divergence": divergence.item()})

        # Check convergence.
        if iter_idx > min_es_iters:
            curr_divergence = sum(tracker["divergence"][-50:]) / 50
            ref_divergence = (
                sum(tracker["divergence"][-ref_es_iters - 50 : -ref_es_iters]) / 50
            )
            if early_stopping and (curr_divergence - ref_divergence) < abs(
                es_thresh * ref_divergence
            ):
                break

    return z.detach(), class_int_arr, tracker

def generate_mds_sets_pretrained(
    vae: VAE,
    classifiers,
    divergence_measure: Callable = JS_divergence,
    z_init: Optional[torch.Tensor] = None,
    num_samples: int = 1,
    max_iters: int = 10_000,
    lr: float = 1e-2,
    early_stopping: bool = True,
    min_es_iters: int = 1000,
    ref_es_iters: int = 100,
    es_thresh: float = 1e-2,
    prior_lambda: float = 0,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Generates maximally different samples w.r.t. classifier1 and classifier2 by
    optimising the latent space of the vae.

    Args:
        vae (nn.Module): VAE.
        classifier1 (nn.Module): Classifier 1.
        classifier2 (nn.Module): Classifier 2.
        divergence (Callable): Divergence metric used to evaluate the differences in
            classification of the two.
        z_init (Optional[torch.Tensor], optional): Initial z. Defaults to None.

    Returns:
        torch.Tensor: A maximally different sample.
    """
    # # Perform dimensionality checks.
    # assert (
    #     classifier1.output_dim == classifier2.output_dim
    # ), "Classifiers have different output dimensions."
    # assert (
    #     vae.x_shape == classifier1.input_shape
    # ), "Classifier input dimensionality should be the same as vae data dimensionality."
    # assert (
    #     vae.x_shape == classifier2.input_shape
    # ), "Classifier input dimensionality should be the same as vae data dimensionality."

    # Ensure all parameters of vae and classifiers are detached.
    for module in [vae]:
        for param in module.parameters():
            param.requires_grad = False

    for module in classifiers:
        for param in module.parameters():
            param.require_grad = False

    # Construct optimisation.
    if z_init is None:
        z_init = torch.randn((num_samples, vae.z_shape[0]), device=vae.device)
    else:
        if len(z_init.shape) == 1:
            z_init = z_init.unsqueeze(0)

    z = nn.Parameter(z_init)
    opt = torch.optim.Adam([z], lr=lr)

    class_int_arr = np.random.randint(0, 999, M)
    class_vector = one_hot_from_int(class_int_arr, batch_size=num_samples)

    class_vector = torch.from_numpy(class_vector)

    tracker = defaultdict(list)

    iter_tqdm = tqdm(range(max_iters), desc="iterations")
    logging.info("Beginning optimisation.")
    for iter_idx in iter_tqdm:
        opt.zero_grad()

        x_arr = vae(z, class_vector, truncation=1)

        for i in range(num_samples):
            x = x_arr[i]

            clf1_preprocess = clf1_weights.transforms()
            clf1_batch = clf1_preprocess(x).unsqueeze(0)
            clf1_preds = clf1(clf1_batch).squeeze(0).softmax(0)
            clf1_py_x = Categorical(probs=clf1_preds)
            
            clf2_preprocess = clf2_weights.transforms()
            clf2_batch = clf2_preprocess(x).unsqueeze(0)
            clf2_preds = clf2(clf2_batch).squeeze(0).softmax(0)
            clf2_py_x = Categorical(probs=clf2_preds)
            
            divergence = divergence_measure(clf1_py_x, clf2_py_x).sum()/num_samples

            prior = vae.pz.log_prob(z)
            prior_sum = prior_lambda*prior.sum()/num_samples
            objective = divergence + prior_sum

        (-objective).backward()
        opt.step()

        # Track metrics.
        tracker["divergence"].append(divergence.item())
        tracker['z'].append(z.clone().detach().numpy())
        tracker['objective'].append(objective.item())
        iter_tqdm.set_postfix({"divergence": divergence.item()})

        # Check convergence.
        if iter_idx > min_es_iters:
            curr_divergence = sum(tracker["divergence"][-50:]) / 50
            ref_divergence = (
                sum(tracker["divergence"][-ref_es_iters - 50 : -ref_es_iters]) / 50
            )
            if early_stopping and (curr_divergence - ref_divergence) < abs(
                es_thresh * ref_divergence
            ):
                break

    return z.detach(), class_int_arr, tracker

def generate_mds_LI(
    vae: VAE,
    classifier1: Classifier,
    classifier2: nn.Module,
    divergence_measure: Callable = torch.distributions.kl_divergence,
    z_init: Optional[torch.Tensor] = None,
    num_samples: int = 1,
    max_iters: int = 10_000,
    lr: float = 1e-2,
    early_stopping: bool = True,
    min_es_iters: int = 1000,
    ref_es_iters: int = 100,
    es_thresh: float = 1e-2,
    prior_lambda: float = 0,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Generates maximally different samples w.r.t. classifier1 and classifier2 by
    optimising the latent space of the vae.

    Args:
        vae (nn.Module): VAE.
        classifier1 (nn.Module): Classifier 1.
        classifier2 (nn.Module): Classifier 2.
        divergence (Callable): Divergence metric used to evaluate the differences in
            classification of the two.
        z_init (Optional[torch.Tensor], optional): Initial z. Defaults to None.

    Returns:
        torch.Tensor: A maximally different sample.
    """
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

    # Ensure all parameters of vae and classifiers are detached.
    for module in [vae]:
        for param in module.parameters():
            param.requires_grad = False

    for module in [classifier1, classifier2]:
        for param in module.parameters():
            param.requires_grad = False

    # Construct optimisation.
    if z_init is None:
        z_init = torch.randn((num_samples, vae.z_shape[0]), device=vae.device)
    else:
        if len(z_init.shape) == 1:
            z_init = z_init.unsqueeze(0)

    z = nn.Parameter(z_init)
    opt = torch.optim.Adam([z], lr=lr)

    tracker = defaultdict(list)

    iter_tqdm = tqdm(range(max_iters), desc="iterations")
    logging.info("Beginning optimisation.")
    for iter_idx in iter_tqdm:
        opt.zero_grad()

        z.requires_grad = True
        x = vae.decode(z)

        n = 2
        igrads = [m.input_grad(x, create_graph=True) for m in [classifier1, classifier2]]
        diverse_loss = sum([torch.sqrt(torch.sum(squared_cos_sim(igrads[i], igrads[j])))
                        for i in range(n)
                        for j in range(i+1, n)])

        classifier1_py_x = classifier1(x)
        classifier2_py_x = classifier2(x)
        divergence = divergence_measure(classifier1_py_x, classifier2_py_x).sum()
        prior = vae.pz.log_prob(z)
        prior_sum = prior_lambda*prior.sum()/len(z)
        # objective = divergence + prior_sum

        objective = diverse_loss - prior_sum

        objective.backward()
        # print(z.grad)
        opt.step()

        # Track metrics.
        tracker["diverse_loss"].append(diverse_loss.item())
        tracker['z'].append(z.clone().detach().numpy())
        tracker['objective'].append(objective.item())
        tracker['divergence'].append(divergence.item())
        iter_tqdm.set_postfix({"diverse_loss": diverse_loss.item()})

        # Check convergence.
        if iter_idx > min_es_iters:
            curr_divergence = sum(tracker["objective"][-50:]) / 50
            ref_divergence = (
                sum(tracker["objective"][-ref_es_iters - 50 : -ref_es_iters]) / 50
            )
            if early_stopping and (curr_divergence - ref_divergence) < abs(
                es_thresh * ref_divergence
            ):
                break

    return z.detach(), tracker
