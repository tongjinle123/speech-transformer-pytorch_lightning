## end2end chinese-english code-swiching speech recognition in pytorch

This is a mixed project borrowing from many other open projects 
opened recently, trying to use most technique and tricks. And with 
pytorch_lightning, experiments can be carried out in an easy way. 
I will try to code in a clean style, with well named functions and 
class and designed file tree, for new guys to understand easily.
And i will try to make evey calculation in a batched and cleaned way.
(such as add bos & eos into batched target and spec augment)
Any ideas can be put into the issues, and welcome for discussion. 
(This project is still being building and reorganizing)

highlights:
    
    can use multiple chinese datasets, with scripts processing those different dataset into a single style
    use sentencepiece encoding for the extension of code-switching
    use tfrecord for fast dataloading (to be add)
    a batched specaugment layer, can be use as a layer like dropout
    use speed perturb in one line
    use low frame rate
    joint training with ctc & cross entropy (label smoothing)
    lookahead & radam
    fast decoding (TODO)
    dockerfile for quick environment install (to be add)
    
datasets:
    
    aishell1 170h
    aishell2 1000h
    prime 100h
    stcmd 100h
    datatang 200h
    datatang 500h
    datatang mix 200h 
    librispeech 960h

models:
        
    (cnn)transformer encoder-decoder 
    (cnn)transformer ctc-encoder-decoder 
    (cnn)transformer ctc (todo)
    (cnn)transformer-transducer (todo)
    (cnn)transformer-aligner (todo)

reference:

    https://github.com/ZhengkunTian/OpenTransformer
    https://github.com/espnet/espnet
    https://github.com/jadore801120/attention-is-all-you-need-pytorch
    https://github.com/alphadl/lookahead.pytorch
    https://github.com/LiyuanLucasLiu/RAdam
    https://github.com/vahidk/tfrecord
    https://github.com/kaituoxu/Speech-Transformer
    