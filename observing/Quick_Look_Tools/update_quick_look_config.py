import sys
import os
import ruamel.yaml


def update_yaml(key, value):
    with open('hector_display_config.yaml') as f:
        yaml.preserve_quotes = True
        doc = ruamel.yaml.load(f, Loader=ruamel.yaml.RoundTripLoader)
        doc[key] = value

    with open('hector_display_config.yaml', 'w') as f:
        yaml.preserve_quotes = True
        f.write(ruamel.yaml.dump(doc, Dumper=ruamel.yaml.RoundTripDumper))


if __name__ == "__main__":
    import yaml
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--observing_run_dates", default=None, type=str, help="The dates of the observing run")
    parser.add_argument("--file_prefix", default=None, type=str, help="The prefix of the data files")
    parser.add_argument("--robot_file_name", default=None, type=str, help="The name of the robot file")
    parser.add_argument("--data_type", default=None, type=str, help="Specify the data type (recommended type is tramline_map)")
    parser.add_argument("--red_or_blue", default=None, type=str, help="Specify which arms (red, blue or both) to use for Quick Look constructions")
    parser.add_argument("--sigma_clip", default=None, type=str, help="Do you want to perform sigma-clipping?")
    parser.add_argument("--centroid", default=None, type=str, help="Do you want to fit centroids?")
    parser.add_argument("--make_plots", default=None, type=str, help="Set to true to save all the diagnostic plots")

    args = parser.parse_args()
    observing_run_dates = args.observing_run_dates
    file_prefix = args.file_prefix
    robot_file_name = args.robot_file_name
    data_type = args.data_type
    red_or_blue = args.red_or_blue
    sigma_clip = args.sigma_clip
    centroid = args.centroid
    make_plots = args.make_plots


    if observing_run_dates is not None:
        data_dir = "/data/hector/reduction/{}/".format(observing_run_dates)
        update_yaml('data_dir', os.path.abspath(data_dir))
        print('---> Observing run dates keyword updated to {}.'.format(observing_run_dates))

    elif file_prefix is not None:
        update_yaml('file_prefix', file_prefix)
        print('---> File prefix keyword updated to {}.'.format(file_prefix))

    elif robot_file_name is not None:
        update_yaml('robot_file_name', file_prefix)
        print('---> Robot file name keyword updated to {}.'.format(robot_file_name))

    elif data_type is not None:
        if (data_type == "raw") or (data_type == 'reduced'):
            print("WARNING: The recommended data type to use is tramline_map")

        update_yaml('data_type', data_type)
        print('---> Data type keyword updated to {}.'.format(data_type))

    elif red_or_blue is not None:
        update_yaml('red_or_blue', red_or_blue)
        print('---> {} arms will be used in the quick look tool constructions.'.format(red_or_blue))

    elif sigma_clip is not None:
        update_yaml('sigma_clip', sigma_clip)
        print('---> Sigma clip keyword updated to {}.'.format(sigma_clip))

    elif centroid is not None:
        update_yaml('centroid', centroid)
        print('---> Centroid keyword updated to {}.'.format(centroid))

    elif make_plots is not None:
        update_yaml('make_plots', make_plots)
        print('---> Make plots keyword updated to {}.'.format(make_plots))

    else:
        print('---> Nothing updated...')

