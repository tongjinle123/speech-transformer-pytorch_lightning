import sentencepiece as spm
import re


class Vocab:
    """
    vocab for sentence piece
    """
    def __init__(self, from_path):
        self.spm = spm.SentencePieceProcessor()
        self.spm.load(from_path)
        self.pattern = re.compile('(\\[[A-Z]{0,3}\\])|([。，])')

    def _id2piece(self, id):
        return self.spm.id_to_piece(id)

    def str2token(self, string):
        string = string.replace('-', ' ').replace('  ', '')
        string = re.sub(self.pattern, '', string)
        output = self.spm.encode_as_pieces(string)
        if output[0] != '▁':
            return output
        else:
            return output[1:]

    def str2id(self, string):
        string = string.replace('-', ' ').replace('  ', '')
        string = re.sub(self.pattern, '', string)
        output = self.spm.encode_as_ids(string)
        if output[0] != self.token2id('▁'):
            return output
        else:
            return output[1:]

    def token2id(self, token_list):
        return self.spm.piece_to_id(token_list)

    def token2string(self, token_list):
        return self.spm.decode_pieces(token_list)

    def id2token(self, id_list):
        return [self._id2piece(i) for i in id_list]

    def id2string(self, id_list):
        return self.spm.decode_ids(id_list)

    @property
    def pad_id(self):
        return self.spm.pad_id()

    @property
    def unk_id(self):
        return self.spm.unk_id()

    @property
    def blank_id(self):
        return self.spm.piece_to_id(self.blank_token)

    @property
    def pad_token(self):
        return self._id2piece(self.pad_id)

    @property
    def unk_token(self):
        return self._id2piece(self.unk_id)

    @property
    def blank_token(self):
        return '[B]'

    @property
    def vocab_size(self):
        return self.spm.get_piece_size()

    @property
    def bos_id(self):
        return self.spm.piece_to_id('<s>')

    @property
    def eos_id(self):
        return self.spm.piece_to_id('</s>')