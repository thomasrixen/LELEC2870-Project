# -*- coding: utf-8 -*-

import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


from sklearn.manifold import TSNE

import warnings
# Ignore all warning messages
warnings.filterwarnings('ignore')


#----- HELPER FUNCTIONS FOR P4 -----#


# Displays grid of sample images from dataset
def visualize_sample_images(dataset, gridsize=(3,8)):
    
    plt.close('all')
    
    data_loader = DataLoader(dataset, batch_size=100, shuffle=True)
    
    nrows, ncols = gridsize
    width, height = gridsize[::-1]
    fig, axes = plt.subplots(nrows, ncols, figsize=(width*1.5,height*1.5))

    for i, (images, labels) in enumerate(data_loader):
        if i == nrows:
            break

        for j in range(ncols):
            ax = axes[i, j]
            image = images[j].squeeze().numpy()
            label = labels[j].item()

            ax.imshow(image, cmap='gray')
            ax.set_title(f"Risk: {label}")
            ax.axis('off')
    plt.tight_layout()
    plt.show()
    

# Visualize image data in 2D with TSNE
def visualize_dataset_tSNE(dataset, extract_features=False, feature_extractor=None):
       
    # Create dataloader
    data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    images, labels = next(iter(data_loader))
    
    # Apply TSNE 
    tsne = TSNE(perplexity=100)
    
    # If extract feature is True, first extract features of images then apply TSNE
    if extract_features:
        img_features = feature_extractor.extract_features(dataset.images, dataset.images_directory)
        X = tsne.fit_transform(img_features)
    # If extract feature is False, apply TSNE on original images (flattened as 1D vectors) 
    else:
        images = images.squeeze().numpy()
        labels = labels.squeeze().numpy()
        X = tsne.fit_transform(images.reshape(len(images),-1))
  
    # Define custom colormap
    colors = ["green","yellow","orange","red","black"]
    nodes = [0,0.15,0.3,0.5,1]
    mycmap = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))
    
    # Create plot
    fig, ax = plt.subplots()
    scatter = plt.scatter(X[:,0],X[:,1], c=labels, cmap=mycmap, vmin=0, vmax=1)
    plt.axis('off')
    plt.colorbar(label="Risk")

    # Define annotation box and initialize with dummy values
    im = OffsetImage(images[0].reshape(48,48), cmap='gray')
    ab = AnnotationBbox(offsetbox = im, # image
                        xy = (0,0), # reference coordinate
                        xybox=(-40, 40), # coordinate of box wrt to reference xy
                        xycoords='data', 
                        boxcoords="offset points",
                        pad=0,
                        arrowprops=dict(arrowstyle="->"))
    ab.set_visible(False) # by default, annotation box is invisible
    ax.add_artist(ab) # add anotation box to axes
    
    # Define event to display the image corresponding to the point which is hovered over
    def motion_hover(event):
        ab_visible = ab.get_visible() # is the annotation currently visible ?
        if event.inaxes == ax: # if mouse is contained in axes
            is_contained, annotation_index = scatter.contains(event) 
            # is_contained : if point is hovered over 
            # annotation_index : index of point
            if is_contained: # if the mouse hovers over a point
                idx = annotation_index['ind'][0] # retrieve index of point
                data_point_location = scatter.get_offsets()[idx] # retrieve coordinates of point
                ab.offsetbox = OffsetImage(images[idx].reshape(48,48), zoom=2, cmap='gray') # match point to corresponding image
                ab.xy = data_point_location # set reference to coordinates of point
                ab.set_visible(True) # display annotation box
                fig.canvas.draw_idle() # redraw figure
            else: # is mouse is in figure but not over a point
                if ab_visible: # if annotation box is currently visible
                    ab.set_visible(False) # hide annotation box
                    fig.canvas.draw_idle() # redraw figure
                    
    # Connect event to figure
    fig.canvas.mpl_connect('motion_notify_event', motion_hover)

    plt.show()


# Convolves image with given custom_kernel
def visualize_2Dconvolution(image, custom_kernel):
    
    plt.close('all')
    
    # Normalize pixels between 0 and 1
    max_pixel_value = torch.max(image)
    min_pixel_value = torch.min(image)
    image = (image-min_pixel_value)/(max_pixel_value-min_pixel_value)
   
    # Convert the custom kernel to a PyTorch tensor
    kernel = torch.FloatTensor(custom_kernel).unsqueeze(0).unsqueeze(0)

    # Apply the custom kernel using 2D convolution
    transformed_image = nn.functional.conv2d(image, kernel)

    # Display the original and transformed images
    plt.figure(figsize=(12, 4))
    
    # Define own colormap
    colors = ["magenta","black","green"]
    nodes = [0, 0.5, 1]
    mycmap = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))
    
    # Display original (normalized) image
    plt.subplot(131)
    image = image.squeeze().numpy()
    plt.title(f"Original Image {image.shape}")
    plt.axis('off')
    plt.imshow(image, cmap='gray')
    
    # Display Kernel
    plt.subplot(132)
    kernel = kernel.squeeze().numpy()
    plt.title(f"Kernel {kernel.shape}")
    plt.axis('off')
    vmax = np.max(kernel)
    vmin = np.min([np.min(kernel),-vmax])
    mynorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    plt.imshow(kernel, cmap=mycmap, norm=mynorm)
    for (j,i),value in np.ndenumerate(kernel):
        plt.text(i,j,value,ha='center',va='center',color="white")
    
    # Display transformed image
    plt.subplot(133)
    transformed_image = transformed_image.squeeze().numpy()
    plt.title(f"Transformed Image {transformed_image.shape}")
    plt.axis('off')
    vmax = np.max(transformed_image)
    vmin = np.min([np.min(transformed_image),-vmax])         
    mynorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    plt.imshow(transformed_image, cmap=mycmap, norm=mynorm)
    
    plt.show()
    
# Verifies if my_guess correspond to the true number of parameters in a given model
def verify_number_of_parameters(my_guess, model):
    true_answer = sum(p.numel() for p in model.parameters())
    if int(my_guess) == int(true_answer):
        print(f"You are correct! There is indeed {my_guess} parameters in this neural network.")
    else:
        if my_guess<true_answer:
            print(f"You are incorrect, there is more than {my_guess} parameters in this neural network.\nTry again!")
        else:
            print(f"You are incorrect, there is less than {my_guess} parameters in this neural network.\nTry again!")




# Visualizes regression results and brief residual analysis
def visualize_regression_results(y_true,y_pred):
    
    plt.close('all')

    fig = plt.figure(figsize=(12, 4))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])

    ax1 = plt.subplot(gs[0])
    ax1.set_title("Predicted risks vs test risks")
    ax1.scatter(y_true, y_pred, marker='o', color='b', s=10)
    ax1.plot([0, 1], [0, 1], 'b--', label='Perfect predictions', alpha=0.3)
    ax1.set_xlabel('y_test')
    ax1.set_ylabel('y_pred')
    ax1.legend()

    ax2 = plt.subplot(gs[1])
    ax2.set_title("Error vs test risks")
    ax2.scatter(y_true, (y_true-y_pred), marker='o', color='m', s=10)
    ax2.plot([0, 1], [0, 0], 'm--', label='Zero error', alpha=0.3)
    ax2.set_xlabel('y_test')
    ax2.set_ylabel('Error')
    ax2.legend(loc='best')

    num_bins = 20
    ax3 = plt.subplot(gs[2])
    ax3.set_title("Distribution of predicted and test risks")
    hist_true, bins, _ = ax3.hist(y_true, bins=num_bins, alpha=0.3, color='green', edgecolor="green", label='y_test')
    hist_pred, _, _ = ax3.hist(y_pred, bins=bins, alpha=0.3, color='orange', edgecolor="orange", label='y_pred')
    ax3.set_xlim(-0.05,1.05)
    ax3.set_xlabel('Risk')
    ax3.set_ylabel('Frequency')
    ax3.legend(loc='best')

    # Show the plot
    plt.tight_layout()
    plt.show()
