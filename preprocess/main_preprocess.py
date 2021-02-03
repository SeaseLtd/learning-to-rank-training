import getopt
import sys
from preprocess import preprocessing
from preprocess import preprocessing_new_query_ID


def main(argv):
    unix_options = "ho:d:e:x:"
    gnu_options = ["help", "output_dir=", "dataset_path=", "encoding=", "experiment="]
    args1 = []
    chosen_experiment = "2"

    try:
        arguments, values = getopt.getopt(argv, unix_options, gnu_options)
    except getopt.error as err:
        # output error, and return with an error code
        print(str(err))
        sys.exit(2)
    for currentArgument, currentValue in arguments:
        if currentArgument in ("-h", "--help"):
            print("'main_preprocess.py -o <output_dir> -d <dataset_path> -e <encoding> -x <experiment>'")
            sys.exit(1)
        elif currentArgument in ("-o", "--output_dir"):
            args1.append(currentValue)
        elif currentArgument in ("-d", "--dataset_path"):
            args1.append(currentValue)
        elif currentArgument in ("-e", "--encoding"):
            args1.append(currentValue)
        elif currentArgument in ("-x", "--experiment"):
            chosen_experiment = currentValue

    if chosen_experiment == "1":
        preprocessing.preprocessing(*args1)
    elif chosen_experiment == "2":
        preprocessing_new_query_ID.preprocessing_new_query_ID(*args1)
    else:
        print("Error, the selected experiment does not exist")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main(sys.argv[1:])