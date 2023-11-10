# ANI-1 Molecular Potential Prediction using Pre-trained EGNN and Transformer-Encoder

![Alt Text!](images/architecture.png)<br>
This repository holds an Equivariant Graph Neural Network (EGNN) + Transformer-Encoder model used for end-to-end ANI-1 molecular potential prediction. The details for pretraining, fine-tuning and visualization could be found below.

Note: The EGNN portion of the model used is a modified version from [Reference 1/3](#reference), which does not perform node-coordinates update through layers. Therefore, it possesses an invariant property instead of E(n), which was described in the original paper.

### Model Training and Evaluation Workflow:
1) <b>Data Preparation</b>: Place your dataset in the `./Data` folder. Adjust the necessary parameters in the config.yaml file.
1) <b>Data Reading and Splitting</b>: The model imports coordinates, atom species, and energies from the files in `./Data`. It then divides the data into training/validation/test subsets as defined in the configuration.
2) <b>Energy Processing</b>: Additional energy processing including subtraction of self interaction energy from the total energy.
3) <b>Data Packaging</b>: Organize the processed data into `PyGDataLoader` to create a `train_loader`, which is then ready for the training process.
4) <b>Training Setup</b>: Training function configured using parameters specified in the configuration file.
5) <b>Normalization</b>: Normalize the target data (y-values) using training dataset statistics (mean and standard deviation), and record these values.
6) <b>Logging and Output</b>: Set up a logging function, file writing, and TensorBoard writer for monitoring the training process.
7) <b>Model Training</b>: Train the model and save the model parameters with the lowest validation loss. For evaluation, the y-values are denormalized using the previously recorded standard deviation and mean.
8) <b>Evaluation on Test Set</b>: Use the best-performing model to evaluate the test set and analyze the results.

<a name="reference"></a>
## Reference:
1. Satorras et al., <i>E(n) Equivariant Graph Neural Networks.</i> [[Paper]](https://arxiv.org/abs/2102.09844) [[GitHub]](https://github.com/vgsatorras/egnn)
2. Vaswani et al., <i>Attention is All You Need.</i>
[[Paper]](https://arxiv.org/abs/1706.03762)
3. Wang et al., <i>Denoise Pre-training on Non-equilibrium Molecules for Accurate and Transferable Neural Potentials.</i><br> [[Paper]](https://arxiv.org/abs/2303.02216) [[GitHub]](https://github.com/yuyangw/Denoise-Pretrain-ML-Potential)
4. Smith et al., <i>ANI-1: An extensible neural network potential with DFT accuracy at force field computational cost.</i><br>
[[Paper]](https://pubs.rsc.org/en/content/articlelanding/2017/sc/c6sc05720a) [[GitHub]](https://github.com/isayev/ANI1_dataset)
5. Smith et al., <i>ANI-1, A data set of 20 million calculated off-equilibrium conformations for organic molecules.</i><br>
[[Paper]](https://www.nature.com/articles/sdata2017193)