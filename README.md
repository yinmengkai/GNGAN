# GNGAN

Python implementation code for the paper titled,

Title: Modeling stochastic porous media using gradient normalization based generative adversarial network

Authors: Ting Zhang1, Mengkai Yin1, Yuqi Wu 2, 3, *, Yi Du4, **

1.College of Computer Science and Technology, Shanghai University of Electric Power, Shanghai 200090, China 

2.National Key Laboratory of Deep Oil and Gas, China University of Petroleum (East China), Qingdao 266580, China

3.School of Geosciences, China University of Petroleum (East China), Qingdao 266580, China

4.School of Computer and Information Engineering, Institute for Artificial Intelligence, Shanghai Polytechnic University, Shanghai 201209, China

(*Corresponding authors, E-mail: [wuyuqi@upc.edu.cn](mailto:wuyuqi@upc.edu.cn) (Y. Wu), **Corresponding author, E-mail: duyi@sspu.edu.cn (Y. Du))

Ting Zhang Email: tingzh@shiep.edu.cn, Affiliation: College of Computer Science and Technology, Shanghai University of Electric Power, Shanghai 200090, China

Mengkai Yin Email: yinmengkai@mail.shiep.edu.cn, Affiliation: College of Computer Science and Technology, Shanghai University of Electric Power, Shanghai 200090, China

Yuqi Wu Email: wuyuqi@upc.edu.cn, Affiliation: National Key Laboratory of Deep Oil and Gas, China University of Petroleum (East China), Qingdao 266580, China

Yi Du Email: duyi0701@126.com, Affiliation: School of Computer and Information Engineering, Institute for Artificial Intelligence, Shanghai Polytechnic University, Shanghai 201209, China

1.requirements

- pytorch ==1.10.0


- To run the code, an NVIDIA GeForce RTX3080 GPU video card with 10 GB video memory is required. 


- Software development environment should be any Python integrated development environment used on an NVIDIA video card. Programming language: Python 3.7.

2.How to use？

- Use `utils/create_training_images.py` to create the subvolume training images. This will create the sub-volume training images as an hdf5 format which can then be used for training.  

- Train the GAN,use `main_train.py` to train the GAN network.
