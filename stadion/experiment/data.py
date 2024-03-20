import argparse
from pathlib import Path
import numpy as onp

from stadion.sample import make_data
from stadion.utils.parse import load_data_config, save_data, timer
from stadion.definitions import FOLDER_TRAIN, FOLDER_TEST, NAN_MIN, NAN_MAX
from stadion.core import get_intv_stats

if __name__ == "__main__":
    """
    Generates data for the experiments
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--descr", type=str, required=True)
    parser.add_argument("--data_config_path", type=Path, required=True)
    parser.add_argument("--path_data", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sanity_check_plots", action="store_true")
    parser.add_argument("--grid_id", type=int)
    kwargs = parser.parse_args()

    # sanity check a few datasets
    sanity_check_plots = kwargs.sanity_check_plots and (kwargs.seed <= 5 and kwargs.grid_id is None)

    # load data config
    data_config = load_data_config(kwargs.data_config_path, warn_if_grid=kwargs.grid_id is None,
                                                            warn_if_not_grid=kwargs.grid_id is not None)
    data_folder = kwargs.path_data / f"{kwargs.seed}"

    # select grid option in case there is one
    if kwargs.grid_id is not None:
        key = list(data_config.keys())[kwargs.grid_id]
        data_config = data_config[key]
        data_folder = kwargs.path_data / f"{key}-{kwargs.seed}"

    # generate dataset based on data_config
    with timer() as walltime:
        train_targets, test_targets, meta_data = make_data(seed=kwargs.seed, config=data_config)

    meta_data["walltime"] = walltime() / 60.0 # mins

    # check for nans
    if kwargs.grid_id is None:

        train_means, train_intv = get_intv_stats(train_targets)
        test_means, test_intv = get_intv_stats(test_targets)

        is_nan_train = any([onp.isnan(x).any() for x in train_targets.data])
        assert not is_nan_train, f"train data contains nans:\n" \
                                 f"intv\n{train_intv.astype(onp.int32)}\n" \
                                 f"means\n{train_means}"

        is_nan_test =  any([onp.isnan(x).any() for x in test_targets.data])
        assert not is_nan_test,  f"test data contains nans :\n" \
                                 f"intv\n{test_intv.astype(onp.int32)}\n" \
                                 f"means\n{test_means}"

        is_oob_train = any([((x < NAN_MIN) | (x > NAN_MAX)).any() for x in train_targets.data])
        assert not is_oob_train, f"train data out of bounds:\n" \
                                 f"intv\n{train_intv.astype(onp.int32)}\n" \
                                 f"means\n{train_means}"

        is_oob_test =  any([((x < NAN_MIN) | (x > NAN_MAX)).any() for x in test_targets.data])
        assert not is_oob_test,  f"test data out of bounds :\n" \
                                 f"intv\n{test_intv.astype(onp.int32)}\n" \
                                 f"means\n{test_means}"


    # write to file
    save_data(train_targets, data_folder / FOLDER_TRAIN, meta_data=meta_data, sanity_check_plots=sanity_check_plots)
    save_data(test_targets, data_folder / FOLDER_TEST, meta_data=meta_data, sanity_check_plots=False)

    print(f"{kwargs.descr}: {kwargs.seed} finished successfully.")