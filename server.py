import sys
import os
sys.path.append(os.getcwd())
import shutil
from flask import Flask, request
import torch as t
import json
from flask_cors import CORS, cross_origin
# from .vad import vad
import os


print("Loading model...")
import json
import torch as t
from src.model.rnn_lm.rnn_lm import ClassifierWithState
from src.model.rnn_lm.rnn_lm import RNNLM
from src.model.transformer_rezero import LightningModel
import pandas as pd


def load_rnn_lm():
    keys = json.load(open('rnnlm_3/model.json'))
    rnn = RNNLM(n_vocab=keys['n_vocab'], n_units=keys['unit'], n_layers=keys['layer'], n_embed=None, typ=keys['type'])
    rnn_model = ClassifierWithState(rnn)
    rnn_model.load_state_dict(t.load('rnnlm_3/rnnlm.model.best'))
    rnn_model.eval()
    return rnn_model


def load_model():
    model = LightningModel.load_from_metrics(
        'exp/lightning_logs/version_2000/checkpoints/epoch=100_v8.ckpt',
        'exp/lightning_logs/version_2000/meta_tags.csv'
    )
    val_dataloader = model.val_dataloader()
    model.eval()
    return model, val_dataloader


def load_token_list(path='testing_vocab.vocab'):
    with open(path) as reader:
        data = reader.readlines()
        data = [i.split('\t')[0] for i in data]
    return data


def parse_output(output, char_list):
    parsed_outputs = []
    for i in output:
        parsed_output = {}
        token = [clist[j] for j in i['yseq']]
        score = i['score']
        parsed_output = {'token': ''.join(token).replace('<s>', '').replace('</s>', ''), 'score': score}
        parsed_outputs.append(parsed_output)
    return parsed_outputs


def load_manifest(path):
    return pd.read_csv(path)

clist = load_token_list()
rnn_model = load_rnn_lm()
model, val_loader = load_model()
pipe = val_loader.dataset.datasets[0]



print("Model loaded")

app = Flask(__name__)


@app.route("/recognize", methods=["POST"])
@cross_origin(origin='http://172.18.34.25', headers=['Content-Type'])
def recognize():
    f = request.files["file"]
    beam = request.files['beam']
    print(f)
    f.save("test.wav")
    feature, feature_length = pipe.load_wav('test.wav')
    feature = t.from_numpy(feature)
    feature_length = t.LongTensor([feature_length])

    with t.no_grad():
        output = model.transformer.recognize(
            feature.unsqueeze(0), feature_length, beam=10, penalty=0, ctc_weight=0.35, maxlenratio=0.35,
            minlenratio=0, char_list=clist, rnnlm=rnn_model, lm_weight=0.1, nbest=10
        )

    parsed_outputs = parse_output(output, clist)
    return parsed_outputs

#
# @app.route("/recognize_long", methods=["POST"])
# @cross_origin(origin='http://172.18.34.25', headers=['Content-Type'])
# def recognize_long():
#     f = request.files["file"]
#     print(f)
#     f.save("test.wav")
#     if os.path.exists('tmp_long/'):
#         shutil.rmtree('tmp_long/')
#     os.mkdir('tmp_long/')
#     vad(2, 'test.wav', 'tmp_long/')
#     tmp_list = os.listdir('tmp_long/')
#     all_ = []
#     for i in tmp_list:
#         with t.no_grad():
#             feature, length = parser.parser_wav_inference(i)
#             output = model.beam_decode_feature(feature.float().cuda(), length.cuda())
#             all_.append(output)
#     return ' '.join(all_)





if __name__ == '__main__':
    app.run("0.0.0.0", debug=True, port=5000)

