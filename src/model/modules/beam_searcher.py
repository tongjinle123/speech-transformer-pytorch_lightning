import torch as t


class BestSaver:
    def __init__(self, best_k):
        self.list = []
        self.best_k = best_k
        self.low_score = -1e20

    def add(self, token, score):
        self.list.append((token, score))
        self.list.sort(key=lambda x: x[1])
        self.list = self.list[-self.best_k:]


class BatchBestSaver:
    def __init__(self, best_k, batch_size, lp_eps=0.0, lp_lambda=5):
        self.batch = [BestSaver(best_k)] * batch_size
        self.batch_low_socre = [-1e30] * batch_size
        self.lp_eps = lp_eps
        self.lp_lambda = lp_lambda

    def add(self, batch_token, batch_score, batch_length):
        # bs, b, seq
        # bs, b, seq
        # bs, b,
        for batch_index, values in enumerate(zip(batch_token, batch_score, batch_length)):
            for b_token, b_score, b_length in zip(*values):
                lp = (self.lp_lambda + b_length) / (self.lp_lambda + 1) ** self.lp_eps
                normalized_score = b_score[-1] / lp
                self.batch[batch_index].add(b_token, normalized_score)


class BeamSteper:
    """
    batched_beam_searcher
    """

    def __init__(self, batch_size, beam_size, bos_id, eos_id, vocab_size, device, k_best=5, lp_eps=0.0, lp_lambda=5):
        self.beam_size = beam_size
        self.batch_size = batch_size
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.device = device
        self.vocab_size = vocab_size
        self.init_containers()
        self.batch_best_saver = BatchBestSaver(k_best, batch_size, lp_eps=lp_eps, lp_lambda=lp_lambda)

    def init_containers(self):
        self.prob_container = t.zeros((self.batch_size, self.beam_size, 1), device=self.device)
        self.token_container = t.ones((self.batch_size, self.beam_size, 1), device=self.device,
                                      dtype=t.long) * self.bos_id
        self.length_container = t.ones((self.batch_size, self.beam_size), dtype=t.long, device=self.device)
        self.continue_mask = t.ones((self.batch_size, self.beam_size), dtype=t.bool, device=self.device)

    def get_first_step_token(self):
        return t.ones((self.batch_size * self.beam_size, 1), dtype=t.long, device=self.device)

    def step(self, step_prob):
        # step_prob [batch_size, beam_size, vocab_size]
        if self.continue_mask.sum() == 0:
            return None
        else:
            step_prob, step_prob_indice = t.topk(step_prob, self.beam_size)
            step_prob.masked_fill_(~self.continue_mask.unsqueeze(-1), -1e10)

            # [batch_size, beam_size, beam_size]
            self.token_container = self.token_container.unsqueeze(-2).repeat(1, 1, self.beam_size, 1)
            # [batch_size, beam_size, beam_size, seqlength]
            self.token_container = t.cat([self.token_container, step_prob_indice.unsqueeze(-1)], dim=-1)
            # [batch_size, beam_size, beam_size, new_sequlenth]
            #         return self.prob_container
            tmp_prob_container = self.prob_container.unsqueeze(-2).repeat(1, 1, self.beam_size, 1)
            # [batch_size, beam_size, beam_size, seqlength]
            tmp_prob_container = t.cat([tmp_prob_container, step_prob.unsqueeze(-1)], dim=-1)
            # batch_size, beam_size, beam_size, seqlength]
            tmp_prob_container = t.sum(tmp_prob_container[:, :, :, -2:], dim=-1, keepdim=True)
            #         return tmp_prob_container
            tmp_prob_container = tmp_prob_container.view(self.batch_size, self.beam_size * self.beam_size)
            # batch_size, beam_size * beam_size]
            tmp_prob_container, index = t.topk(tmp_prob_container, self.beam_size)
            self.prob_container = t.cat([self.prob_container, tmp_prob_container.unsqueeze(-1)], dim=-1)
            # [batch_size, beam_size]
            self.token_container = self.token_container.view(self.batch_size, self.beam_size * self.beam_size, -1)
            self.token_container = [t.index_select(i, 0, v) for i, v in zip(self.token_container, index)]
            self.token_container = t.stack(self.token_container, 0)
            # [batch_size, beam_size, seqlength]
            self.continue_mask.masked_fill_(
                self.token_container[:, :, -1].eq(self.eos_id),
                False)
            self.length_container = self.length_container + self.continue_mask
            self.batch_best_saver.add(self.token_container, self.prob_container, self.length_container)
            return self.token_container, self.length_container
