import torch as t
#
# def length2mask(batch_size, sequence_length, lengths):
#     assert batch_size == len(lengths)
#     position = t.arange(1, sequence_length).repeat(batch_size, 1)
#     return position.le(lengths.unsqueeze(-1).expand_as(position)).byte()


class SpecAugment(t.nn.Module):
    """
    batched spec augment layer, switch off when self.eval()
    """
    def __init__(self, p=0.2, fill_value=0.0, num_freq_mask=1, num_time_mask=1, freq_mask_length=10,
                 time_mask_length=10, max_sequence_length=1000):
        super(SpecAugment, self).__init__()
        self.p = p
        self.fill_value = fill_value
        self.num_freq_mask = num_freq_mask
        self.num_time_mask = num_time_mask
        self.freq_mask_length = freq_mask_length
        self.time_mask_length = time_mask_length

        position = t.arange(1, max_sequence_length)
        self.register_buffer('position', position)

    def get_mask(self, batch_size, sequence_length, lengths):
        position = self.position.repeat(batch_size, 1)[:, :sequence_length]
        return position.le(lengths.unsqueeze(-1).expand_as(position)).unsqueeze(-1)

    def get_time_mask(self, batch_size, sequence_length, lengths, restricted=True):
        device = self.position.device
        position = self.position.repeat(batch_size, 1)[:, :sequence_length]
        start = t.cat([t.randint(0, i, (1,)) for i in lengths]).to(device)
        mask_length = t.cat([t.randint(0, self.time_mask_length, (1,)) for i in lengths]).to(device)
        if restricted:
            mask_length = t.LongTensor([m.item() if m < int(self.p * l.item()) else int(self.p * l.item()) for l, m in
                                        zip(lengths, mask_length)]).to(device)
        end = start + mask_length
        position_mask = (position.le(start.unsqueeze(-1).expand_as(position)).byte() + position.le(
            end.unsqueeze(-1).expand_as(position)).byte()).eq(1)
        return position_mask.unsqueeze(-1)

    def get_freq_mask(self, batch_size, freq_length):
        device = self.position.device
        position = self.position.repeat(batch_size, 1)[:, :freq_length]
        start = t.LongTensor([t.randint(0, freq_length, (1,)) for i in range(batch_size)]).to(device)
        mask_length = t.cat([t.randint(0, self.freq_mask_length, (1,)) for i in range(batch_size)]).to(device)
        end = start + mask_length
        position_mask = (position.le(start.unsqueeze(-1).expand_as(position)).byte() + position.le(
            end.unsqueeze(-1).expand_as(position)).byte()).eq(1)
        return position_mask.unsqueeze(1)
        #

    def forward(self, spec, length=None):
        if self.training:
            assert length is not None
            batch_size, sequence_length, hidden_size = spec.size()
            pad_mask = self.get_mask(batch_size, sequence_length, length)

            for i in range(self.num_time_mask):
                mask = pad_mask * self.get_time_mask(batch_size, sequence_length, length)
                spec = spec.masked_fill(mask.repeat(1, 1, hidden_size), self.fill_value)
            for j in range(self.num_freq_mask):
                mask = mask + self.get_freq_mask(batch_size, hidden_size)
                spec = spec.masked_fill(mask, self.fill_value)

            return spec
        else:
            return spec


# %%


if __name__ == '__main__':
    spec = t.ones((3, 20, 20))
    # spec = t.randn((3,20,20))
    # spec = (spec - spec.mean())/ spec.std()
    length = t.Tensor([2, 3, 4]).long() + 5
    layer = SpecAugment(p=1)
    mask = layer.get_mask(3, 20, length)
    time_mask = layer.get_time_mask(3, 20, length)
    freq_mask = layer.get_freq_mask(3, 20)
    output = layer(spec, length)

#     import time
#     start = time.time()
#     output = layer(spec, length)
#     print(time.time()-start)