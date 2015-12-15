export PYTHONPATH=/Users/spinoco/scikit-learn

DATA=$1
INGREDIENTS=$2
DATA_TEST=$3
INGREDIENTS_TEST=$4
NETWORK="net_$(date "+%Y-%m-%d-%H-%M")"
PREDICTION="predict_$(date "+%Y-%m-%d-%H-%M")"
python2.7 mlpclassifier.py -d train.json -m $DATA -i $INGREDIENTS -n $NETWORK
python2.7 recallTest.py -d test.json -m $DATA_TEST -i $INGREDIENTS_TEST -n $NETWORK > $PREDICTION
