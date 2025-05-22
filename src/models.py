import os
from meegnet.network import Model
import logging


def build_and_train_network(
    dataset,
    data_path,
    n_outputs=2,
    net_option="meegnet",
    name="taskclf_meegnet",
    net_params=None,
    max_epoch=15,
    verbose=1
):
    """
    Build and train the meegnet model.
    """
    input_size = dataset.data[0].shape
    save_path = data_path

    # Only pass net_params for 'mlp' or 'custom'
    if net_option.lower() in ["mlp", "custom"]:
        if net_params is None:
            raise ValueError(
                "net_params is required for MLP and custom networks")
        model = Model(
            name,
            net_option,
            input_size,
            n_outputs,
            save_path,
            net_params=net_params
        )
    else:
        # net_params will be ignored for all other networks
        if net_params is not None:
            raise Warning(
                net_params + " will be ignored for this network type")
        model = Model(
            name,
            net_option,
            input_size,
            n_outputs,
            save_path
        )

    print(model.net)
    model.train(dataset, max_epoch=max_epoch, verbose=verbose)
    return model
