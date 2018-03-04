echo "downloading example mxnet resnet50 model"
mkdir -p model/
cd model/
wget http://data.mxnet.io/models/imagenet/resnet/50-layers/resnet-50-0000.params
wget http://data.mxnet.io/models/imagenet/resnet/50-layers/resnet-50-symbol.json
cd ..

echo "downloading example dataset cifar10"
mkdir -p data/
cd data/
wget http://data.mxnet.io/data/cifar10/cifar10_train.rec
wget http://data.mxnet.io/data/cifar10/cifar10_val.rec

echo "creating log file folder"
mkdir -p log/

# echo "start to run training script"
# nohup python mxnet_train.py log/cifar10_0304_1331.log -f --gpus 0,1,2,3  --batch-size 512  --network resnet-50 --num-epochs 20 --model-prefix /workspace/run/master/image-classification/train/model/my_model &
