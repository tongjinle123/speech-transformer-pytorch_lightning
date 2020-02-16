import fire
from src_new import model as Model
from os.path import join
from os import listdir



def load_model(model_name, exp_path):
    ckpts = join(exp_path, 'checkpoints/')
    ckpts = join(ckpts, listdir(ckpts)[0])
    tags = join(exp_path, 'meta_tags.csv')
    model = getattr(Model, model_name)
    model.load_from_metrics(
        ckpts,
        tags_csv=tags
    )
    model.eval()
    test_dataloader = model.test_dataloader()[0]

    return model, test_dataloader








if __name__ == '__main__':
    fire.Fire()