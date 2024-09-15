# WGAN-GP


1.requirements

Tensorflow-gpu == 1.13.1

Tensorlayer == 1.10.1

To run the code, an NVIDIA GeForce RTX2070 Super GPU video card with 8GB video memory is required. 

Software development environment should be any Python integrated development environment used on an NVIDIA video card. 

Programming language: Python 3.6. 


2.How to use？


First, preprocess the image: Cut the porous media slice into 64 * 64 * 64 size pictures. Each training image consists of 64 pictures of size 64 * 64, stored in a separate folder. Then use use scripts/preparedata.py to convert the image into .npy format.


Secondly, set the network parameters such as batchsize, learning rate and storage location. The executable .py file of WGAN-GP, the path is: WGAN-GP/WGAN-GP.py. After configuring the parameters and environment, you can run directly: python WGAN-GP.py


Finally, in WGAN-GP/savepoint/Test, find the loss images during the training process and the .npy format of the porous media three-dimensional structure images of different rounds. Use scripts/loadnpy to convert .npy to .txt format for later analysis and processing.

