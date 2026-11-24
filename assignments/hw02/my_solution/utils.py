import matplotlib.pyplot as plt


def view_configuration_space(grid, cspace, resolution=0.025):
    height, width = grid.shape

    extent = (0, width * resolution,
              0, height * resolution)

    fig, axes = plt.subplots(
        1, 2,
        figsize=(16, 7),
        sharex=True,
        sharey=True
    )

    # Original map
    axes[0].imshow(
        grid != 0,
        cmap='gray_r',
        origin='lower',
        interpolation='nearest',
        extent=extent,
        vmin=0,
        vmax=1
    )
    axes[0].set_title('Original Map')

    # Configuration space
    axes[1].imshow(
        cspace,
        cmap='gray_r',
        origin='lower',
        interpolation='nearest',
        extent=extent,
        vmin=0,
        vmax=1
    )
    axes[1].set_title('Configuration Space')

    for ax in axes:
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()