from os import makedirs

import torch
import ivon
from torch.nn.utils import parameters_to_vector
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse

from archs import load_architecture
from utilities import get_gd_optimizer, get_gd_directory, get_loss_and_acc, compute_losses, \
    save_files, save_files_final, get_hessian_eigenvalues, iterate_dataset, plot_singular_values, initialize_storage, \
    store_singular_values, initialize_subspace_distances, store_singular_values_and_subspace_distance
from data import load_dataset, take_first, DATASETS


def main(dataset: str = "cifar10-10k", arch_id: str = "resnet32", loss: str = "ce", opt: str = "gd", lr: float = 1e-2, max_steps: int = 10000, neigs: int = 0,
         physical_batch_size: int = 1000, eig_freq: int = -1, iterate_freq: int = -1, save_freq: int = -1,
         save_model: bool = False, beta: float = 0.0, beta2: float=0.99, nproj: int = 0,
         loss_goal: float = None, acc_goal: float = None, abridged_size: int = 5000, seed: int = 0, h0: float =0.1, device_id: int = 1):
    
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    directory = get_gd_directory(dataset, lr, arch_id, seed, opt, loss, beta)
    print(f"output directory: {directory}")
    makedirs(directory, exist_ok=True)

    ## torch load singular value storage
    # singular_value_storage = torch.load(f"{directory}/singular_value_storage")
    # print("singular_value_storage loaded: ", singular_value_storage)
    train_dataset, test_dataset = load_dataset(dataset, loss, device)
    abridged_train = take_first(train_dataset, abridged_size)

    loss_fn, acc_fn = get_loss_and_acc(loss)

    torch.manual_seed(seed)
    network = load_architecture(arch_id, dataset).to(device)
    singular_value_storage = initialize_storage(network)
    subspace_distances = initialize_subspace_distances(network)

    projectors = torch.randn(nproj, len(parameters_to_vector(network.parameters())))
    post_samples=10

    optimizer = get_gd_optimizer(network.parameters(), opt, lr, beta, post_samples,beta2,h0)
    #scheduler = CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=0)

    train_loss, test_loss, train_acc, test_acc = \
        torch.zeros(max_steps), torch.zeros(max_steps), torch.zeros(max_steps), torch.zeros(max_steps)
    iterates = torch.zeros(max_steps // iterate_freq if iterate_freq > 0 else 0, len(projectors))
    eigs = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, neigs)

    for step in range(0, max_steps):
        train_loss[step], train_acc[step] = compute_losses(network, [loss_fn, acc_fn], train_dataset,
                                                           physical_batch_size, device)
        test_loss[step], test_acc[step] = compute_losses(network, [loss_fn, acc_fn], test_dataset, physical_batch_size, device)

        if eig_freq != -1 and step % eig_freq == 0 and step!=0:
            eigs[step // eig_freq, :] = get_hessian_eigenvalues(network, loss_fn, abridged_train, neigs=neigs,
                                                                physical_batch_size=physical_batch_size, device=device )
            print("eigenvalues: ", eigs[step//eig_freq, :])
            

        if iterate_freq != -1 and step % iterate_freq == 0:
            iterates[step // iterate_freq, :] = projectors.mv(parameters_to_vector(network.parameters()).cpu().detach())

        if save_freq != -1 and step % save_freq == 0:
            save_files(directory, [("eigs", eigs[:step // eig_freq]), ("iterates", iterates[:step // iterate_freq]),
                                   ("train_loss", train_loss[:step]), ("test_loss", test_loss[:step]),
                                   ("train_acc", train_acc[:step]), ("test_acc", test_acc[:step])])
        if step % 10 == 0:
            print(f"{step}\t{train_loss[step]:.3f}\t{train_acc[step]:.3f}\t{test_loss[step]:.3f}\t{test_acc[step]:.3f}")

        if (loss_goal != None and train_loss[step] < loss_goal) or (acc_goal != None and train_acc[step] > acc_goal):
            break
        
        if opt=="ivon":
            optimizer.zero_grad()
            for (X, y) in iterate_dataset(train_dataset, physical_batch_size):
                for _ in range(post_samples):
                    with optimizer.sampled_params(train=True):                    
                        loss = loss_fn(network(X.to(device)), y.to(device)) / len(train_dataset)
                        loss.backward()
            optimizer.step()
            #scheduler.step()

        else:
            optimizer.zero_grad()
            for (X, y) in iterate_dataset(train_dataset, physical_batch_size):
                loss = loss_fn(network(X.to(device)), y.to(device)) / len(train_dataset)
                loss.backward()
            optimizer.step()

    save_files_final(directory,
                     [("eigs", eigs[:(step + 1) // eig_freq]), ("iterates", iterates[:(step + 1) // iterate_freq]),
                      ("train_loss", train_loss[:step + 1]), ("test_loss", test_loss[:step + 1]),
                      ("train_acc", train_acc[:step + 1]), ("test_acc", test_acc[:step + 1])])
    if save_model:
        torch.save(network.state_dict(), f"{directory}/snapshot_final")

    ## torch save the variable singular_value_storage in a directory
    # torch.save(singular_value_storage, f"{directory}/singular_value_storage")
    # torch.save(subspace_distances, f"{directory}/subspace_distances")

    # plot_singular_values(singular_value_storage, directory)    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train using gradient descent.")
    parser.add_argument("--dataset", type=str, default="cifar10-10k", choices=DATASETS, help="which dataset to train")
    parser.add_argument("--arch_id", type=str, default="resnet32", help="which network architectures to train")
    parser.add_argument("--loss", type=str, default="ce", choices=["ce", "mse"], help="which loss function to use")
    parser.add_argument("--lr", type=float, default=1e-2, help="the learning rate")
    parser.add_argument("--max_steps", type=int, default=500, help="the maximum number of gradient steps to train for")
    parser.add_argument("--opt", type=str, choices=["gd", "polyak", "nesterov","adam","ivon"],
                        help="which optimization algorithm to use", default="gd")
    parser.add_argument("--seed", type=int, help="the random seed used when initializing the network weights",
                        default=0)
    parser.add_argument("--beta", type=float, help="momentum parameter (used if opt = polyak or nesterov)", default=0.0)
    parser.add_argument("--beta2", type=float, help="momentum parameter for variance", default=0.9)
    parser.add_argument("--physical_batch_size", type=int,
                        help="the maximum number of examples that we try to fit on the GPU at once", default=1000)
    parser.add_argument("--acc_goal", type=float,
                        help="terminate training if the train accuracy ever crosses this value", default=None)
    parser.add_argument("--loss_goal", type=float, help="terminate training if the train loss ever crosses this value", default=None)
    parser.add_argument("--neigs", type=int, help="the number of top eigenvalues to compute", default=2)
    parser.add_argument("--eig_freq", type=int, default=-1,
                        help="the frequency at which we compute the top Hessian eigenvalues (-1 means never)")
    parser.add_argument("--nproj", type=int, default=0, help="the dimension of random projections")
    parser.add_argument("--iterate_freq", type=int, default=-1,
                        help="the frequency at which we save random projections of the iterates")
    parser.add_argument("--abridged_size", type=int, default=5000,
                        help="when computing top Hessian eigenvalues, use an abridged dataset of this size")
    parser.add_argument("--save_freq", type=int, default=-1,
                        help="the frequency at which we save results")
    parser.add_argument("--save_model", type=bool, default=False,
                        help="if 'true', save model weights at end of training")
    parser.add_argument("--h0", type=float, default=0.1,
                        help="posterior variance init")
    parser.add_argument("--device_id", type=int, default=1, help="ID of the GPU to use")
    args = parser.parse_args()

    main(dataset=args.dataset, arch_id=args.arch_id, loss=args.loss, opt=args.opt, lr=args.lr, max_steps=args.max_steps,
         neigs=args.neigs, physical_batch_size=args.physical_batch_size, eig_freq=args.eig_freq,
         iterate_freq=args.iterate_freq, save_freq=args.save_freq, save_model=args.save_model, beta=args.beta, beta2=args.beta2,
         nproj=args.nproj, loss_goal=args.loss_goal, acc_goal=args.acc_goal, abridged_size=args.abridged_size,
         seed=args.seed, h0=args.h0, device_id=args.device_id)
