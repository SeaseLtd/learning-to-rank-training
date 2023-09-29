import getopt
import sys
from trainingSetBuilder import training_set_builder
from trainingSetBuilder import training_set_builder_subset


def main(argv):
    unix_options = "ho:d:m:e:"
    gnu_options = ["help", "output_dir=", "dataset_name=" "mapping=", "experiment="]
    args1 = []
    args2 = []
    chosen_experiment = "2"

    try:
        arguments, values = getopt.getopt(argv, unix_options, gnu_options)
    except getopt.error as err:
        # output error, and return with an error code
        print(str(err))
        sys.exit(2)
    for currentArgument, currentValue in arguments:
        if currentArgument in ("-h", "--help"):
            print("'main_trainingSetBuilder.py -o <output_dir> -d <dataset_name> -m <mapping> -e <experiment>'")
            sys.exit(1)
        elif currentArgument in ("-o", "--output_dir"):
            args1.append(currentValue)
            args2.append(currentValue)
        elif currentArgument in ("-d", "--dataset_name"):
            args1.append(currentValue)
        elif currentArgument in ("-m", "--mapping"):
            args1.append(currentValue)
        elif currentArgument in ("-e", "--experiment"):
            chosen_experiment = currentValue

    if chosen_experiment == "1":
        training_set_builder.training_set_builder(*args1)
    elif chosen_experiment == "2":
        training_set_builder_subset.training_set_builder_subset(*args2)
    else:
        print("Error, the selected experiment does not exist")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main(sys.argv[1:])
