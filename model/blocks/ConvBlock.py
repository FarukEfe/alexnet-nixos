'''
Architecture
--------
Overlapping pooling: stride < kernel_size achieved by setting s=2 and z=3

Local response normalization: used for generalization, mimics lateral inhibition in real neurons.
                              this is esp. useful for generalization of features. Take squared sum of all activations with
                              kernel i applied at (x,y). Hyperparameters: k=2, n=5, alpha=10e-4, beta=0.75

Double GPU training: convolutional layers are split across 2 GPUs, fully connected layers are repliated on each one.
                     from 2nd conv layer to 3rd, there is cross-GPU communication to share feature maps.

Architecture:

1. Input 3x224x224
2. Conv1: 96 filters (split across 2 GPUs, each with 48 filters), 11x11 kernel, stride 4, padding 0 -> ReLU -> LRN -> MaxPool 3x3, stride 2
3. Conv2: 128 filters each GPU (total 256), 5x5 kernel, stride 1, padding 2 -> ReLU -> LRN -> MaxPool 3x3, stride 2
4. Conv3: 192 filters each GPU (total 384), 3x3 kernel, stride 1, padding 1 -> ReLU
5. Conv4: same as Conv3
6. Conv5: 128 filters each GPU (total 256), 3x3 kernel, stride 1, padding 1 -> ReLU -> MaxPool 3x3, stride 2
7. Dense1: 4096 neurons -> ReLU -> Dropout (p=0.5)
8. Dense2: 4096 neurons -> ReLU -> Dropout (p=0.5)
9. Dense3: 1000 neurons -> Softmax
'''

