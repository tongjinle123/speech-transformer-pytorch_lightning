## end2end chinese-english code-swiching speech recognition in pytorch

This is a mixed project borrowing from many other open projects 
opened recently, trying to use most technique and tricks. And with 
pytorch_lightning, experiments can be carried out in an easy way. 
I will try to code in a clean style, for new guys to understand easily.
Any ideas can put into the issue, and welcome for discussion.

highlights:
    
    can use multiple chinese datasets 
    use sentencepiece encoding 
    use tfrecord for fast dataloading
    use batched specaugment layer 
    use speed perturb
    use low frame rate
    joint training with ctc & cross entropy (label smoothing)
    lookahead & radam
    fast decoding (TODO)
    
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
    