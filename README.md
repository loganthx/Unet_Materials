# Unet_Materials

A loose UNET implementation of image classification of liquid crystals

# Architecture

a UNET outputs a (learnable) grayscale image that serves as inputs to 'simplified' fully connected layers

# Reasons

UNETs are excelent in image segmentation tasks. The unet here provided passes segmentation tasks benchmarks so we think we could try to use this architecture to 'see' where the network is looking for activations.
Our network can't predict above 65% yet, since temperatures around a relative big range can't be easily predicted like binary classification tasks. 

Our current results on the materials (our main goal):

# Plots

![Image Alt text](/logs/plots_30_epochs.png)

# Cats and Dogs Dataset
We see that our net makes good segmentation decisions on cats dogs dataset, but in the current case the 'simplified' connected layers are detrimental to te accuracy improvement.

Segmentation Results after 5 epochs:

![Image Alt text](/logs/plots_5_epochs.png)


