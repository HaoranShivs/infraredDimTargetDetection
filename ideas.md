# how we set CNN architecture
## multiscale features fusion
### sum or cat?
if we simply sum them with the largest channel of these features, the information will lost.
so, the best way is catting them. But the shallow features are not prepared to fusion. So, 1*1 Conv is indispensable and Final channel is sum of original channels of features.
### more or less parameters work better?
in our experience, no matter the 128 or 256(deepest conv_layer's channel),they are work same worse. and in same architecture, 64 is worse than 128 about 10% 
python train.py --lr 0.001 --net-name basenet_8-32-32-64-128 --batch-size 32 --epochs 100