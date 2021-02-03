import getopt
import sys

from streamsOrdering import streams_ordering
from streamsOrdering import streams_ordering_subset

def main(argv):
    unix_options = "hd:l:e:"
    gnu_options = ["help", "dataset_path=","largest_streams_dataset_path=", "experiment="]
    args1 = []
    chosen_experiment = "1"

    try:
        arguments, values = getopt.getopt(argv, unix_options, gnu_options)
    except getopt.error as err:
        # output error, and return with an error code
        print(str(err))
        sys.exit(2)

    for currentArgument, currentValue in arguments:
        if currentArgument in ("-h", "--help"):
            print("'main_streams.py -d <dataset_path> -l <largest_streams_dataset_path> -e <experiment>'")
            sys.exit(1)
        elif currentArgument in ("-d", "--dataset_path"):
            args1.append(currentValue)
        elif currentArgument in ("-l", "--largest_streams_dataset_path"):
            args1.append(currentValue)
        elif currentArgument in ("-e", "--experiment"):
            chosen_experiment = currentValue

    if chosen_experiment == "1":
        streams_ordering.streams_ordering(*args1)
    elif chosen_experiment == "2":
        streams_ordering_subset.streams_ordering_subset(*args1)
    else:
        print("Error, the selected experiment does not exist")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main(sys.argv[1:])