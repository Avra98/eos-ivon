import torch
import argparse
import sys
import os 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ivon._ivon import IVON


def quadratic_function(x):
    return 0.5 * (x - 3) ** 2

def main(lr: float = 1e-2, max_steps: int = 10000, seed: int = 0, h0: float = 0.1, post_samples: int = 10, opt="gd", device_id: int = 0):
    
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
  
    torch.manual_seed(seed)
    x = torch.tensor([0.0], requires_grad=True, device=device)
    if opt =='gd':
        optimizer = torch.optim.SGD([x], lr=lr)
        
        for step in range(max_steps):
            optimizer.zero_grad()            
            loss = quadratic_function(x)            
            loss.backward()            
            optimizer.step()


            if step % 10 == 0:
                print(f"Step {step}: x = {x.item():.4f}, loss = {loss.item():.4f}")
        
        print(f"Optimization finished: x = {x.item():.4f}, final loss = {loss.item():.4f}")

    elif opt =='ivon':
        optimizer = IVON([x], lr=lr, ess= 1e1, weight_decay=0.0,mc_samples=post_samples, beta1=0.0,beta2=1.0, hess_init=h0)  
        for step in range(max_steps):
            optimizer.zero_grad()
            for _ in range(post_samples):
                optimizer.zero_grad()
                with optimizer.sampled_params(train=True): 
                    loss = quadratic_function(x)
                    loss.backward()
                    
            optimizer.step()       

            if step % 10 == 0:
                print(f"Step {step}: x = {x.item():.4f}, loss = {loss.item():.4f}")
                print(optimizer.state_dict()['param_groups'][0]['hess'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimize a quadratic function using gradient descent.")
    parser.add_argument("--lr", type=float, default=0.1, help="The learning rate")
    parser.add_argument("--max_steps", type=int, default=500, help="The maximum number of gradient steps to train for")
    parser.add_argument("--seed", type=int, help="The random seed used when initializing the variables", default=0)
    parser.add_argument("--h0", type=float, default=0.1, help="Posterior variance init (not used in this example)")
    parser.add_argument("--post_samples", type=int, default=10, help="Number of posterior samples (not used in this example)")
    parser.add_argument("--opt", type=str, default="gd", help="gd or ivon")
    parser.add_argument("--device_id", type=int, default=0, help="ID of the GPU to use (if available)")
    args = parser.parse_args()

    main(lr=args.lr, max_steps=args.max_steps, seed=args.seed, h0=args.h0, post_samples=args.post_samples,opt = args.opt, device_id=args.device_id)
