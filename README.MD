## end2end chinese-english code-swiching speech recognition in pytorch

highlights:
    
    use multiple datasets
    use sentencepiece encoding 
    use tfrecord for fast dataloading
    use specaugment layer 
    use speed perturb
    use low frame rate

datasets:
    
    aishell1 170h
    aishell2 1000h
    prime 100h
    stcmd 100h
    datatang 200h
    datatang 500h
    datatang mix 200h 
    librispeech 960h

models(todo):
    
    transformer ctc
    transformer encoder-decoder
    transformer ctc-encoder-decoder
    cnn-transformer ctc
    cnn-transformer encoder-decoder
    cnn-transformer ctc-encoder-decoder
    transformer-transducer 
    transformer-aligner

