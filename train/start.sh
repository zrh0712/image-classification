echo "downloading example mxnet resnet50 model"
mkdir -p model/
cd model/
wget http://data.mxnet.io/models/imagenet/resnet/50-layers/resnet-50-0000.params
wget http://data.mxnet.io/models/imagenet/resnet/50-layers/resnet-50-symbol.json
cd ..

echo "downloading example dataset cifar10"
mkdir -p data/
wget http://data.mxnet.io/data/cifar10/cifar10_train.rec
wget http://data.mxnet.io/data/cifar10/cifar10_val.rec

echo "start to run training script"
python mxnet_train.py log.log -f -s 
