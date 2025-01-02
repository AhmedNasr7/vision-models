import matplotlib.pyplot as plt
import torch

# Function to plot random samples from the dataset
def plot_random_samples(dataset, class_names, fig_size=(5, 5), n=16):
    """
    Plots random samples from the dataset in a grid.

    Parameters:
        dataset (torchvision.datasets): Dataset to sample from.
        class_names (list): List of class names.
        n (int): Number of samples to display. Must be a perfect square.
    """
    if n % int(n**0.5) != 0:
        raise ValueError("n must be a perfect square for a grid layout.")

    # Select random indices
    indices = torch.randint(0, len(dataset), (n,))
    samples = [dataset[i] for i in indices]

    # Prepare grid
    grid_size = int(n**0.5)
    fig, axes = plt.subplots(grid_size, grid_size, figsize=fig_size)
    fig.tight_layout(pad=2)

    for i, ax in enumerate(axes.flat):
        image, label = samples[i]
        image = image.permute(1, 2, 0)  # Convert (C, H, W) to (H, W, C) for plotting
        image = image * torch.tensor([0.5, 0.5, 0.5]) + torch.tensor([0.5, 0.5, 0.5])  # Unnormalize
        image = image.clamp(0, 1)  # Clamp values to [0, 1] for display

        ax.imshow(image)
        ax.set_title(class_names[label], fontsize=8)
        ax.axis("off")

    plt.show()

