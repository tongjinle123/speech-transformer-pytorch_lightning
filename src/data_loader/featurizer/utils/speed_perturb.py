import random
import torch as t


def speed_perturb(sig, min_rate=90, max_rate=110, step=2):
    rate = random.choice([i/100 for i in range(min_rate, max_rate, step)])
    perturbed = t.nn.functional.interpolate(sig.unsqueeze(0), scale_factor=rate)
    return perturbed[0]