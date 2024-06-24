import flax.linen as nn
import jax
import jax.numpy as jnp

from avid.diffusion import DiffusionBackbone, SinusoidalEmbedding


class UNetResidualBlock(nn.Module):
    """
    Implements a residual block within a U-Net architecture using JAX.

    This module defines a residual block with convolutional layers and normalization, followed by a residual connection. It's a fundamental building block in constructing deeper and more complex U-Net architectures for tasks like image segmentation and generation.

    Attributes:
        width (int): The number of output channels for the convolutional layers within the block.

    Methods:
        __call__(x: jnp.ndarray): Processes the input tensor through the residual block and returns the result.
    """

    width: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        input_width = x.shape[-1]

        # Define layers
        convolution_1 = nn.Conv(self.width, kernel_size=(1, 1, 1))
        convolution_2 = nn.Conv(self.width, kernel_size=(3, 3, 3), padding='CIRCULAR')
        convolution_3 = nn.Conv(self.width, kernel_size=(3, 3, 3), padding='CIRCULAR')
        norm = nn.GroupNorm(num_groups=2, epsilon=1e-5, use_bias=False, use_scale=False)

        # Residual connection
        residual = convolution_1(x) if input_width != self.width else x

        # Forward pass
        x = norm(x)
        x = nn.swish(x)
        x = convolution_2(x)
        x = nn.swish(x)
        x = convolution_3(x)

        return x + residual


class UNetDownBlock(nn.Module):
    """
    Implements a down-sampling block in a U-Net architecture using JAX.

    This module consists of a sequence of residual blocks followed by an average pooling operation to reduce the spatial dimensions. It's used to capture higher-level features at reduced spatial resolutions in the encoding pathway of a U-Net.

    Attributes:
        width (int): The number of output channels for the convolutional layers within the block.
        block_depth (int): The number of residual blocks to include in the down-sampling block.

    Methods:
        setup(): Initializes the sequence of residual blocks.
        __call__(x: jnp.ndarray): Processes the input tensor through the down-sampling block and returns the result.
    """

    width: int
    block_depth: int

    def setup(self):
        self.residual_blocks = [UNetResidualBlock(self.width) for _ in range(self.block_depth)]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for block in self.residual_blocks:
            x = block(x)
        x = nn.avg_pool(x, window_shape=(2, 2, 2), strides=(2, 2, 2))
        return x


class UNetUpBlock(nn.Module):
    """
    Implements an up-sampling block in a U-Net architecture using JAX.

    This module consists of a sequence of residual blocks and a bilinear up-sampling operation to increase the spatial dimensions. It's used in the decoding pathway of a U-Net to progressively recover spatial resolution and detail in the output image.

    Attributes:
        width (int): The number of output channels for the convolutional layers within the block.
        block_depth (int): The number of residual blocks to include in the up-sampling block.

    Methods:
        setup(): Initializes the sequence of residual blocks.
        __call__(x: jnp.ndarray, skip: jnp.ndarray): Processes the input tensor and a skip connection from the encoding pathway through the up-sampling block and returns the result.
    """

    width: int
    block_depth: int

    def setup(self):
        self.residual_blocks = [UNetResidualBlock(self.width) for _ in range(self.block_depth)]

    def __call__(self, x: jnp.ndarray, skip: jnp.ndarray) -> jnp.ndarray:
        B, D, H, W, C = x.shape
        upsampled_shape = (B, D * 2, H * 2, W * 2, C)
        x = jax.image.resize(x, shape=upsampled_shape, method='bilinear')
        x = jnp.concatenate([x, skip], axis=-1)
        for block in self.residual_blocks:
            x = block(x)
        return x


class UNet(DiffusionBackbone):
    """
    Implements the U-Net architecture for image processing tasks using JAX.

    This model is widely used for tasks such as image segmentation, denoising, and super-resolution. It features a symmetric encoder-decoder structure with skip connections between corresponding layers in the encoder and decoder to preserve spatial information.

    Attributes:
        image_size (Tuple[int, int]): The size of the input images (height, width).
        widths (List[int]): The number of output channels for each block in the U-Net architecture.
        block_depth (int): The number of residual blocks in each down-sampling and up-sampling block.
        embed_dims (int): The dimensionality of the sinusoidal embeddings for encoding positional information.
        embed_min_freq (float): The minimum frequency for the sinusoidal embeddings.
        embed_max_freq (float): The maximum frequency for the sinusoidal embeddings.

    Methods:
        setup(): Initializes the U-Net architecture including the sinusoidal embedding layer, down-sampling blocks, residual blocks, and up-sampling blocks.
        __call__(noisy_images: jnp.ndarray, noise_variances: jnp.ndarray): Processes noisy images and their associated noise variances through the U-Net and returns the denoised images.
    """

    widths: tuple[int]
    block_depth: int
    embed_dims: int
    embed_min_freq: float
    embed_max_freq: float

    @nn.compact
    def __call__(
        self, noisy_images: jnp.ndarray, noise_variances: jnp.ndarray, training: bool = False
    ) -> jnp.ndarray:
        sinusoidal_embedding = SinusoidalEmbedding(
            self.embed_dims, self.embed_min_freq, self.embed_max_freq
        )
        down_blocks = [UNetDownBlock(width, self.block_depth) for width in self.widths[:-1]]
        residual_blocks = [UNetResidualBlock(self.widths[-1]) for _ in range(self.block_depth)]
        up_blocks = [UNetUpBlock(width, self.block_depth) for width in reversed(self.widths[:-1])]
        convolution_1 = nn.Conv(self.widths[0], kernel_size=(1, 1, 1))
        convolution_2 = nn.Conv(
            noisy_images.shape[-1], kernel_size=(1, 1, 1), kernel_init=nn.initializers.zeros
        )
        e = sinusoidal_embedding(noise_variances)
        upsampled_shape = (
            # B, D, H, W
            *noisy_images.shape[:4],
            self.embed_dims,
        )
        e = jax.image.resize(e, upsampled_shape, method='nearest')

        x = convolution_1(noisy_images)
        x = jnp.concatenate([x, e], axis=-1)

        skips = []
        for block in down_blocks:
            skips.append(x)
            x = block(x)

        for block in residual_blocks:
            x = block(x)

        for block, skip in zip(up_blocks, reversed(skips)):
            x = block(x, skip)

        outputs = convolution_2(x)
        return outputs
