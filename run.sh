pip install scikit-image tensorboardX visdom==0.2.1

mkdir -p outputs/dcgan
cp -r /home2/pytorch-broad-models/3D-GAN/dcgan_pretrained/first_test outputs/dcgan/.

python main.py --test=True --num_iter 200 --num_warmup 20 --jit --channels_last 1 --device cuda --precision float16 --profile
