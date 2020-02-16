## end2end chinese-english code-swiching speech recognition in pytorch

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
    