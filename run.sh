mkdir -p outputs/dcgan
cp -r /home2/pytorch-broad-models/3D-GAN/dcgan_pretrained/first_test outputs/dcgan/.

python main.py --test=True --num_iter 10 --num_warmup 1
