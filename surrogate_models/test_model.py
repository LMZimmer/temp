import glob
import json
import os

import click

from surrogate_models import utils


def load_surrogate_model(model_log_dir):
    # Load config
    data_config = json.load(open(os.path.join(model_log_dir, 'data_config.json'), 'r'))
    model_config = json.load(open(os.path.join(model_log_dir, 'model_config.json'), 'r'))

    # Instantiate model
    surrogate_model = utils.model_dict[model_config['model']](data_root='None', log_dir=None,
                                                              seed=data_config['seed'], data_config=data_config,
                                                              model_config=model_config)

    # Load the model
    surrogate_model.load(os.path.join(model_log_dir, 'surrogate_model.model'))
    return surrogate_model


@click.command()
@click.option('--model_log_dir', type=click.STRING, help='path to saved surrogate model.',
              default='experiments/surrogate_models/lgb/20200130-152015-6/surrogate_model.model')
def validate_and_test_surrogate_model(model_log_dir):
    # Instantiate surrogate model
    surrogate_model = load_surrogate_model(model_log_dir)
    flatten = lambda l: [item for sublist in l for item in sublist]
    aposteriori_analysis = {}
    for split in ['val', 'test']:
        paths = flatten(
            [json.load(open(val_opt)) for val_opt in
             glob.glob(os.path.join(model_log_dir, '*_{}_paths.json'.format(split)))])
        _, preds, true = surrogate_model.evaluate(paths)
        aposteriori_analysis[split] = utils.evaluate_metrics(true, preds, prediction_is_first_arg=False)
    print(aposteriori_analysis)


if __name__ == "__main__":
    validate_and_test_surrogate_model()
