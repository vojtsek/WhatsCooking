export PYTHONPATH=/Users/spinoco/scikit-learn

DATA=$1
INGREDIENTS=$2
DATA_TEST=$3
INGREDIENTS_TEST=$4
NETWORK="net_$(date "+%Y-%m-%d-%H-%M")"
PREDICTION="predict_$(date "+%Y-%m-%d-%H-%M")"
python mlpclassifier.py -d dataset/train.json -m $DATA -i $INGREDIENTS -n $NETWORK
python recallTest.py -d dataset/test.json -m $DATA_TEST -i $INGREDIENTS_TEST -n $NETWORK > $PREDICTION
