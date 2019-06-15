# CMAR-ES
MACES: Building Efficient Classification Based on Multiple Association Rules with Evolution Strategy

## Requirements
* Python 3
* Numpy
* Scikit-learn
* Pycma

## How to run?
Run main.py script with following parameters:

* `--train` the path of training data file.
* `--test`  the path of testing data file.
* `--minsup`  the minimum support threshold, in range [0, 1] .
* `--nloop`  the maximum number of iterations that optimization must run.
* `--lambda`  lambda coefficient.
* `--beta`  beta coefficient.
* `--class` index of class label in the dataset (0 for the first index and -1 for the last index).

For example:

```
python main.py --train data//credit/credits.data.csv.train.0 --test data/credit/credits.data.csv.test.0 --minsup 0.01 --nloop 100 --lambda 0.001 --beta 0.01 --class 0
```
