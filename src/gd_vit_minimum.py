import argparse
import os
import sys

import torch
import torchvision
from peft import LoraConfig, get_peft_model
from torchvision.transforms import (Compose, Normalize, RandomHorizontalFlip,
                                    Resize, ToTensor)
from transformers import AutoImageProcessor, AutoModelForImageClassification

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from archs import load_architecture
from data import DATASETS, load_dataset, take_first
from utilities import (compute_losses, get_hessian_eigenvalues, get_hessian_eigenvalues_trainable,
                       get_loss_and_acc, iterate_dataset)


def main(dataset, arch_id, loss, lr, max_steps, neigs, physical_batch_size, eig_freq,
         abridged_size, seed, weight_decay, device_id, finetune, hessian):

    # Initialization
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Original setting
    train_dataset, test_dataset = load_dataset(dataset, loss, device)
    abridged_train = take_first(train_dataset, abridged_size, device)
    network = load_architecture(arch_id, dataset).to(device)

    # ==========  Uncomment the following lines to use ViT  ==========
    class WrapperModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            return self.model(x).logits

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    network = AutoModelForImageClassification.from_pretrained(
        "WinKawaks/vit-tiny-patch16-224",
        label2id={label: i for i, label in enumerate(classes)},
        id2label=dict(enumerate(classes)),
        ignore_mismatched_sizes=True,
        attn_implementation="eager",
    )

    if finetune == "lora":
        config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
            modules_to_save=["classifier"],
        )
        network = get_peft_model(network, config)
    elif finetune == "lastlayer":
        network.requires_grad_(False)
        network.classifier.requires_grad_(True)
    else:
        assert finetune == "full"

    ft_group = [p for p in network.parameters() if p.requires_grad]

    if hessian == "full":
        network.requires_grad_(True)

    print(f"Total #param: {sum(p.numel() for p in network.parameters())}")
    print(f"Finetune #param: {sum(p.numel() for p in ft_group)}")
    print(f"grad #param: {sum(p.numel() for p in network.parameters() if p.requires_grad)}")

    network = WrapperModel(network)
    network = network.cuda()

    image_processor = AutoImageProcessor.from_pretrained("WinKawaks/vit-tiny-patch16-224")
    transforms = Compose(
        [
            Resize((image_processor.size["height"], image_processor.size["width"])),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
        ]
    )
    if args.loss == "ce":
        target_transform = None
    elif args.loss == "mse":
        target_transform = Compose(
            [torch.tensor, lambda x: torch.nn.functional.one_hot(x, num_classes=10)]
        )
    train_dataset = torchvision.datasets.CIFAR10(
        root="/home/cong/codes/eos-ivon/datasets",
        train=True,
        transform=transforms,
        target_transform=target_transform,
    )
    train_dataset.data = train_dataset.data[:10000]
    abridged_train = train_dataset
    test_dataset = torchvision.datasets.CIFAR10(
        root="/home/cong/codes/eos-ivon/datasets",
        train=False,
        transform=transforms,
        target_transform=target_transform
    )
    # ==========  Uncomment the above lines to use ViT  ==========

    loss_fn, acc_fn = get_loss_and_acc(loss)

    optimizer = torch.optim.SGD(ft_group, lr=lr, weight_decay=weight_decay)

    train_loss, test_loss, train_acc, test_acc = \
        torch.zeros(max_steps), torch.zeros(max_steps), torch.zeros(max_steps), torch.zeros(max_steps)
    eigs = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, neigs)

    for step in range(0, max_steps):

        if eig_freq != -1 and step % eig_freq == 0 and step >= 0:
            if hessian == "full":
                eigs[step // eig_freq, :] = get_hessian_eigenvalues(network, loss_fn, abridged_train, neigs=neigs,
                                                        physical_batch_size=physical_batch_size, device=device )
            elif hessian == "sub":
                eigs[step // eig_freq, :] = get_hessian_eigenvalues_trainable(network, loss_fn, abridged_train, neigs=neigs,
                                                        physical_batch_size=physical_batch_size, device=device )
            print("eigenvalues: ", eigs[step//eig_freq, :])

        if step % 10 == 0:
            train_loss[step], train_acc[step] = compute_losses(network, [loss_fn, acc_fn], train_dataset,
                                                           physical_batch_size, device)
            test_loss[step], test_acc[step] = compute_losses(network, [loss_fn, acc_fn], test_dataset, physical_batch_size, device)
            print(f"{step}\t{train_loss[step]:.3f}\t{train_acc[step]:.3f}\t{test_loss[step]:.3f}\t{test_acc[step]:.3f}")

        optimizer.zero_grad()
        loss_sum = 0
        for (X, y) in iterate_dataset(train_dataset, physical_batch_size):
            loss = loss_fn(network(X.to(device)), y.to(device)) / len(train_dataset)
            loss_sum += loss
            loss.backward()
        print(f"Step {step}, loss: {loss_sum:.3f}")
        optimizer.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train using gradient descent.")
    parser.add_argument("--dataset", type=str, default="cifar10-10k", choices=DATASETS, help="which dataset to train")
    parser.add_argument("--arch_id", type=str, default="fc-tanh", help="which network architectures to train")
    parser.add_argument("--loss", type=str, default="ce", choices=["ce", "mse"], help="which loss function to use")
    parser.add_argument("--lr", type=float, default=1e-2, help="the learning rate")
    parser.add_argument("--max_steps", type=int, default=10000, help="the maximum number of gradient steps to train for")
    parser.add_argument("--seed", type=int, help="the random seed used when initializing the network weights",
                        default=0)
    parser.add_argument("--physical_batch_size", type=int,
                        help="the maximum number of examples that we try to fit on the GPU at once", default=500)
    parser.add_argument("--neigs", type=int, help="the number of top eigenvalues to compute", default=2)
    parser.add_argument("--eig_freq", type=int, default=-1,
                        help="the frequency at which we compute the top Hessian eigenvalues (-1 means never)")
    parser.add_argument("--abridged_size", type=int, default=50000,
                        help="when computing top Hessian eigenvalues, use an abridged dataset of this size")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="weight decay")
    parser.add_argument("--device_id", type=int, default=1, help="ID of the GPU to use")
    parser.add_argument("--finetune", type=str, default="full", choices=["full", "lora", "lastlayer"])
    parser.add_argument("--hessian", type=str, default="full", choices=["full", "sub"])
    args = parser.parse_args()

    main(dataset=args.dataset, arch_id=args.arch_id, loss=args.loss, lr=args.lr, max_steps=args.max_steps,
         neigs=args.neigs, physical_batch_size=args.physical_batch_size, eig_freq=args.eig_freq,
         abridged_size=args.abridged_size, finetune=args.finetune, hessian=args.hessian,
         seed=args.seed, weight_decay=args.weight_decay, device_id=args.device_id)
