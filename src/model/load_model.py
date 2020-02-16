from src import model


def load_model(model_name, ckpt, metric_csv):
    if model_name == 'transformer':
        m = model.transformer.lightning_model.LightningModel.load_from_metrics(
            ckpt,
            tags_csv=metric_csv)
        m.eval()
    elif model_name == 'transformer_lm':
        pass
    else:
        raise ValueError

    return model
