import torch
import math
import numpy as np

def cross_entropy(inputs, targets):
    max_val = torch.max(inputs, dim=-1, keepdim=True)[0]
    target_logits = inputs.gather(dim=-1, index=targets.unsqueeze(-1))
    sum_val = torch.sum(torch.exp(inputs - max_val), dim=-1, keepdim=True)
    return -torch.mean(target_logits - max_val - torch.log(sum_val))


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay, betas, eps):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p not in self.state:
                    self.state[p] = {
                        "m" : torch.zeros_like(p),
                        "v" : torch.zeros_like(p),
                        "step" : 0
                    }
                lr = group["lr"]
                weight_decay = group["weight_decay"]
                betas = group["betas"]
                eps = group["eps"]

                m = self.state[p]["m"]
                v = self.state[p]["v"]
                self.state[p]["step"] += 1
                t = self.state[p]["step"]

                m_t = betas[0] * m + (1 - betas[0]) * p.grad
                m_hat = m_t / (1 - betas[0] ** t)
                v_t = betas[1] * v + (1 - betas[1]) * p.grad.pow(2)
                v_hat = v_t / (1 - betas[1] ** t)
            
                p.add_(- lr * weight_decay * p - lr * m_hat / (torch.sqrt(v_hat) + eps))

                self.state[p]["m"] = m_t
                self.state[p]["v"] = v_t
                
def get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
    if it < warmup_iters:
        return it * max_learning_rate / warmup_iters
    elif warmup_iters <= it < cosine_cycle_iters:
        return min_learning_rate + 0.5* (1 + math.cos((it - warmup_iters) * math.pi / (cosine_cycle_iters - warmup_iters))) * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate

def clip_gradients(parameters, max_l2_norm):
    params = list(parameters)
    total_sq = 0.0
    for p in params:
        if p.grad is None:
            continue
        total_sq += p.grad.pow(2).sum()
    global_norm = torch.sqrt(total_sq)

    scale = max_l2_norm / (global_norm + 1e-6)   
    if scale < 1:
        for p in params:
            if p.grad is None:
                continue
            p.grad *= scale

def get_batch(dataset, batch_size: int, context_length: int, device: str):
    n = len(dataset)
    m = context_length
    offsets = np.random.randint(0, n - m, size=batch_size)
    idx = offsets[:, None] + np.arange(m)
    x_np = dataset[idx]
    y_np = dataset[idx + 1]
    x = torch.from_numpy(x_np).to(device)
    y = torch.from_numpy(y_np).to(device)
    return x, y

def save_checkpoint(model, optimizer, iteration: int, out):
    checkpoint = dict(
        model_state_dict = model.state_dict(),
        optimizer_state_dict = optimizer.state_dict(),
        iteration = iteration
    )
    torch.save(checkpoint, out)

def load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]

if __name__ == "__main__":
    pass