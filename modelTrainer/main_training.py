import getopt
import sys
from modelTrainer import training

def main(argv):
    unix_options = "ho:t:s:n:e:"
    gnu_options = ["help", "output_dir=","training_set=", "test_set=", "name=", "eval_metric="]
    args1 = []
    try:
        arguments, values = getopt.getopt(argv, unix_options, gnu_options)
    except getopt.error as err:
        # output error, and return with an error code
        print(str(err))
        sys.exit(2)
    for currentArgument, currentValue in arguments:
        if currentArgument in ("-h", "--help"):
            print("'main_training.py -o <output_dir> -t <training_set> -s <test_set> -n <name> -e <eval_metric>'")
            sys.exit(1)
        elif currentArgument in ("-o", "--output_dir"):
            args1.append(currentValue)
        elif currentArgument in ("-t", "--training_set"):
            args1.append(currentValue)
        elif currentArgument in ("-s", "--test_set"):
            args1.append(currentValue)
        elif currentArgument in ("-n", "--name"):
            args1.append(currentValue)
        elif currentArgument in ("-e", "--eval_metric"):
            args1.append(currentValue)

    training.train_model(*args1)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main(sys.argv[1:])