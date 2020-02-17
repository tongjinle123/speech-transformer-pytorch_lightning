import torch as t


class BeamSteper:
    def __init__(self, batch_size, beam_size, bos_id, eos_id, vocab_size, device):
        self.beam_size = beam_size
        self.batch_size = batch_size
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.device = device
        self.vocab_size = vocab_size
        self.init_containers()

    def init_containers(self):
        self.prob_container = t.zeros((self.batch_size, self.beam_size, 1), device=self.device)
        self.token_container = t.ones((self.batch_size, self.beam_size, 1), device=self.device,
                                      dtype=t.long) * self.bos_id
        self.length_container = t.LongTensor([1] * self.batch_size, device=self.device)

    def get_first_step_token(self):
        return t.ones((self.batch_size * self.beam_size, 1), dtype=t.long)

    def step(self, step_prob):
        # step_prob [batch_size*beam_size, 1, vocab_size]

        self.prob_container = self.prob_container.view(self.batch_size * self.beam_size, 1, 1)
        # [batch_size * beam_size, 1, 1]
        self.prob_container = self.prob_container.repeat(1, 1, self.beam_size)
        # [batch_size * beam_size, 1, beam_size]

        step_prob, step_token = t.topk(step_prob, k=self.beam_size)
        # [batch_size * beam_size, 1, beam_size], [batch_size * beam_size, 1, beam_size]
        self.token_container = self.token_container.view(self.batch_size * self.beam_size, 1, -1).transpose(-1,
                                                                                                            -2).contiguous()
        self.token_container = self.token_container.repeat(1, 1, self.beam_size)
        self.token_container = t.cat([self.token_container, step_token], dim=1)
        # [batch_size * beam_size, 2, beam_size]
        self.token_container = self.token_container.transpose(-1, -2).contiguous()

        # [batch_size * beam_size, beam_size, 2]
        # self.token_container = self.token_container.view(self.batch_size * self.beam_size * self.beam_size, -1)
        # [batch_size * beam_size * beam_size, 2]

        self.prob_container = self.prob_container + step_prob
        #         self.prob_container = t.cat([self.prob_container, step_prob], dim=1)
        # [batch_size*beam_size, 1, beam_size]

        self.prob_container = self.prob_container.view(self.batch_size, self.beam_size, self.beam_size)
        # [batch_size, beam_size, beam_size]
        self.prob_container = self.prob_container.view(self.batch_size, 1, self.beam_size * self.beam_size)
        # [batch_size, 1, beam_size * beam_size]
        prob, index = t.topk(self.prob_container, self.beam_size)
        # [batch_size, 1, beam_size] [batch_size, 1, beam_size]
        self.token_container = self.token_container.view(self.batch_size * self.beam_size * self.beam_size, -1)
        self.token_container = t.index_select(self.token_container, 0,
                                              index.squeeze(1).view(self.batch_size * self.beam_size))
        # self.token_container = self.token_container.view(self.batch_size, self.beam_size, 1)
        #         return prob, index, self.prob_container
        self.prob_container = self.prob_container.squeeze(1).contiguous().view(
            self.batch_size * self.beam_size * self.beam_size, 1)
        self.prob_container = t.index_select(self.prob_container, 0,
                                             index.squeeze(1).view(self.batch_size * self.beam_size))
        self.prob_container = self.prob_container.view(self.batch_size, self.beam_size, 1)
        return self.token_container, self.prob_container

