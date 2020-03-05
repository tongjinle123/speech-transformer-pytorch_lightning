import random
import torch as t


def speed_perturb(sig, min_rate=90, max_rate=110, step=2, given_rate=None):
    if given_rate is None:
        rate = random.choice([i/100 for i in range(min_rate, max_rate, step)])
    else:
        rate = given_rate
    perturbed = t.nn.functional.interpolate(sig.unsqueeze(0), scale_factor=rate)
    return perturbed[0]

def speed_perturb2(sig, given_rate=None):
    rate = given_rate
    perturbed = t.nn.functional.interpolate(sig.unsqueeze(0), scale_factor=rate)
    return perturbed[0]