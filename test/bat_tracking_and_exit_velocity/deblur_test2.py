import torch
import torch.nn.functional as F
from skimage.restoration import denoise_tv_chambolle


def richardson_lucy(
    observation: torch.Tensor,
    x_0: torch.Tensor,
    k: torch.Tensor,
    steps: int,
    clip: bool = True,
    filter_epsilon: float = 1e-12,
    tv: bool = False,
) -> torch.Tensor:
    """
    Performs Richardson-Lucy deconvolution on an observed image.

    Args:
        observation (torch.Tensor): The observed image.
        x_0 (torch.Tensor): The initial estimate of the deconvolved image.
        k (torch.Tensor): The point spread function (PSF) kernel.
        steps (int): The number of iterations to perform.
        clip (bool): Whether to clip the deconvolved image values between -1 and 1.
        filter_epsilon (float): The epsilon value for filtering small values in the deconvolution process.

    Returns:
        torch.Tensor: The deconvolved image.

    """
    with torch.no_grad():

        psf = k.clone().float()

        # For RGB images
        if x_0.shape[1] == 3:
            psf = psf.repeat(1, 3, 1, 1)

        im_deconv = x_0.clone().float()
        k_T = torch.flip(psf, dims=[2, 3])

        eps = 1e-12
        pad = (psf.size(2) // 2, psf.size(2) // 2, psf.size(3) // 2, psf.size(3) // 2)

        for _ in range(steps):
            conv = F.conv2d(F.pad(im_deconv, pad, mode="replicate"), psf) + eps
            if filter_epsilon:
                relative_blur = torch.where(
                    conv < filter_epsilon, 0.0, observation / conv
                )
            else:
                relative_blur = observation / conv
            im_deconv *= F.conv2d(F.pad(relative_blur, pad, mode="replicate"), k_T)

            if tv:
                if x_0.shape[1] == 3:
                    multichannel = True
                else:
                    multichannel = False

                im_deconv = denoise_tv_chambolle_torch(
                    im_deconv, weight=0.02, n_iter_max=50, multichannel=multichannel
                )
        if clip:
            im_deconv = torch.clamp(im_deconv, -1, 1)

        return im_deconv


def blind_richardson_lucy(
    observation: torch.Tensor,
    x_0: torch.Tensor,
    k_0: torch.Tensor,
    steps: int,
    x_steps: int,
    k_steps: int,
    clip: bool = True,
    filter_epsilon: float = 1e-12,
    tv: bool = False,
) -> torch.Tensor:
    """
    Performs blind Richardson-Lucy deconvolution algorithm to estimate the latent image and the blur kernel.

    Args:
        observation (torch.Tensor): The observed blurry image.
        x_0 (torch.Tensor): The initial estimate of the latent image.
        k_0 (torch.Tensor): The initial estimate of the blur kernel.
        steps (int): The number of iterations to perform.
        clip (bool): Whether to clip the values of the estimated latent image between 0 and 1.
        filter_epsilon (float): A small value used for numerical stability in the algorithm.

    Returns:
        torch.Tensor: The estimated latent image.
        torch.Tensor: The estimated blur kernel.
    """

    observation_L = torch.sum(observation, dim=1, keepdim=True)

    with torch.no_grad():

        k = k_0.clone().float()

        im_deconv = x_0.clone().float()
        im_deconv_L = torch.sum(im_deconv, dim=1, keepdim=True)

        k_T = torch.flip(k, dims=[2, 3])
        im_deconv_L_T = torch.flip(im_deconv_L, dims=[2, 3])

        eps = 1e-12
        pad_im = (k.size(2) // 2, k.size(2) // 2, k.size(3) // 2, k.size(3) // 2)
        pad_k = (
            im_deconv.size(2) // 2,
            im_deconv.size(2) // 2,
            im_deconv.size(3) // 2,
            im_deconv.size(3) // 2,
        )

        for i in range(steps):

            # Kernel estimation
            # The issue with the offset is probably here, as there is no offset when using k as initialization

            for m in range(k_steps):

                k = k.swapaxes(0, 1)
                conv11 = F.conv2d(F.pad(im_deconv_L, pad_im, mode="replicate"), k) + eps

                if filter_epsilon:
                    relative_blur = torch.where(
                        conv11 < filter_epsilon, 0.0, observation_L / conv11
                    )
                else:
                    relative_blur = observation_L / conv11

                k = k.swapaxes(0, 1)
                im_deconv_L_T = im_deconv_L_T.swapaxes(0, 1)
                im_mean = F.conv2d(torch.ones_like(F.pad(k, pad_k)), im_deconv_L_T)
                # im_mean = F.conv2d(F.pad(torch.ones_like(k), pad_k, mode='replicate'), im_deconv_T)

                if filter_epsilon:
                    k = torch.where(im_mean < filter_epsilon, 0.0, k / im_mean)
                else:
                    k /= im_mean

                conv12 = (
                    F.conv2d(
                        F.pad(relative_blur, pad_k, mode="replicate"), im_deconv_L_T
                    )
                    + eps
                )
                conv12 = conv12[
                    :,
                    :,
                    conv12.size(2) // 2
                    - k.size(2) // 2 : conv12.size(2) // 2
                    + k.size(2) // 2
                    + 1,
                    conv12.size(3) // 2
                    - k.size(3) // 2 : conv12.size(3) // 2
                    + k.size(3) // 2
                    + 1,
                ]
                k *= conv12
                k_T = torch.flip(k, dims=[2, 3])

            # For RGB images
            if x_0.shape[1] == 3:
                k = k.repeat(1, 3, 1, 1)
                k_T = k_T.repeat(1, 3, 1, 1)
                groups = 3
            else:
                groups = 1
            # Image estimation

            for n in range(x_steps):

                k = k.swapaxes(0, 1)

                conv21 = (
                    F.conv2d(
                        F.pad(im_deconv, pad_im, mode="replicate"), k, groups=groups
                    )
                    + eps
                )

                if filter_epsilon:
                    relative_blur = torch.where(
                        conv21 < filter_epsilon, 0.0, observation / conv21
                    )
                else:
                    relative_blur = observation / conv21

                # k_mean = F.conv2d(F.pad(torch.ones_like(im_deconv), pad_im, mode='replicate'), k_T)
                k_T = k_T.swapaxes(0, 1)
                k_mean = F.conv2d(
                    torch.ones_like(F.pad(im_deconv, pad_im)), k_T, groups=groups
                )
                if filter_epsilon:
                    im_deconv = torch.where(
                        k_mean < filter_epsilon, 0.0, im_deconv / k_mean
                    )
                else:
                    im_deconv /= k_mean

                im_deconv *= (
                    F.conv2d(
                        F.pad(relative_blur, pad_im, mode="replicate"),
                        k_T,
                        groups=groups,
                    )
                    + eps
                )

                if tv:
                    if x_0.shape[1] == 3:
                        multichannel = True
                    else:
                        multichannel = False

                    # im_deconv = denoise_tv_chambolle_torch(im_deconv, weight=0.02, n_iter_max=50, multichannel=multichannel)
                    im_deconv = denoise_tv_chambolle(
                        im_deconv, weight=0.02, n_iter_max=50, multichannel=multichannel
                    )

                k_T = k_T.swapaxes(0, 1)
                k = k.swapaxes(0, 1)
            k = k[:, 0:1, :, :]
            k_T = k_T[:, 0:1, :, :]
            im_deconv_L = torch.sum(im_deconv, dim=1, keepdim=True)

            if clip:
                im_deconv = torch.clamp(im_deconv, 0, 1)

            im_deconv_T = torch.flip(im_deconv, dims=[2, 3])

    return im_deconv, k


"""
Created on Sun Oct 13 14:30:46 2019
Last edited on 06/ Nov/ 2019

author: Wei-Chung

description: this is the denoise function "denoise_tv_chambolle" in skimage.
It only supports numpy array, this function transfer it and it support torch.tensor.
"""


def diff(image, axis):
    """
    Take the difference of different dimension(1~4) of images
    """
    ndim = image.ndim
    if ndim == 3:
        if axis == 0:
            return image[1:, :, :] - image[:-1, :, :]
        elif axis == 1:
            return image[:, 1:, :] - image[:, :-1, :]
        elif axis == 2:
            return image[:, :, 1:] - image[:, :, :-1]

    elif ndim == 2:
        if axis == 0:
            return image[1:, :] - image[:-1, :]
        elif axis == 1:
            return image[:, 1:] - image[:, :-1]
    elif ndim == 4:
        if axis == 0:
            return image[1:, :, :, :] - image[:-1, :, :, :]
        elif axis == 1:
            return image[:, 1:, :, :] - image[:, :-1, :, :]
        elif axis == 2:
            return image[:, :, 1:, :] - image[:, :, :-1, :]
        elif axis == 3:
            return image[:, :, :, 1:] - image[:, :, :, :-1]
    elif ndim == 1:
        if axis == 0:
            return image[1:] - image[:-1]


def _denoise_tv_chambolle_nd_torch(image, weight=0.1, eps=2.0e-4, n_iter_max=200):
    """
    image : torch.tensor
        n-D input data to be denoised.
    weight : float, optional
        Denoising weight. The greater `weight`, the more denoising (at
        the expense of fidelity to `input`).
    eps : float, optional
        Relative difference of the value of the cost function that determines
        the stop criterion. The algorithm stops when:
            (E_(n-1) - E_n) < eps * E_0
    n_iter_max : int, optional
        Maximal number of iterations used for the optimization.
    Returns
    -------
    out : torch.tensor
        Denoised array of floats.

    """

    ndim = image.ndim
    pt = torch.zeros((image.ndim,) + image.shape, dtype=image.dtype).to(image.device)
    gt = torch.zeros_like(pt)
    dt = torch.zeros_like(image)

    i = 0
    while i < n_iter_max:
        if i > 0:
            # dt will be the (negative) divergence of p
            dt = -pt.sum(0)
            slices_dt = [
                slice(None),
            ] * ndim
            slices_pt = [
                slice(None),
            ] * (ndim + 1)
            for ax in range(ndim):
                slices_dt[ax] = slice(1, None)
                slices_pt[ax + 1] = slice(0, -1)
                slices_pt[0] = ax
                dt[tuple(slices_dt)] += pt[tuple(slices_pt)]
                slices_dt[ax] = slice(None)
                slices_pt[ax + 1] = slice(None)
            out = image + dt
        else:
            out = image
        Et = torch.mul(dt, dt).sum()

        # gt stores the gradients of out along each axis
        # e.g. gt[0] is the first order finite difference along axis 0
        slices_gt = [
            slice(None),
        ] * (ndim + 1)
        for ax in range(ndim):
            slices_gt[ax + 1] = slice(0, -1)
            slices_gt[0] = ax
            gt[tuple(slices_gt)] = diff(out, ax)
            slices_gt[ax + 1] = slice(None)

        norm = torch.sqrt((gt**2).sum(axis=0)).unsqueeze(0)
        Et = Et + weight * norm.sum()
        tau = 1.0 / (2.0 * ndim)
        norm = norm * tau / weight
        norm = norm + 1.0
        pt = pt - tau * gt
        pt = pt / norm
        Et = Et / float(image.view(-1).shape[0])
        if i == 0:
            E_init = Et
            E_previous = Et
        else:
            if torch.abs(E_previous - Et) < eps * E_init:
                break
            else:
                E_previous = Et
        i += 1

    return out


def denoise_tv_chambolle_torch(
    image, weight=0.1, eps=2.0e-4, n_iter_max=200, multichannel=False
):
    """Perform total-variation denoising on n-dimensional images.
    Parameters
    ----------
    image : torch.tensor of ints, uints or floats
        Input data to be denoised. `image` can be of any numeric type,
        but it is cast into an torch.tensor of floats for the computation
        of the denoised image.
    weight : float, optional
        Denoising weight. The greater `weight`, the more denoising (at
        the expense of fidelity to `input`).
    eps : float, optional
        Relative difference of the value of the cost function that
        determines the stop criterion. The algorithm stops when:
            (E_(n-1) - E_n) < eps * E_0
    n_iter_max : int, optional
        Maximal number of iterations used for the optimization.
    multichannel : bool, optional
        Apply total-variation denoising separately for each channel. This
        option should be true for color images, otherwise the denoising is
        also applied in the channels dimension.
    Returns
    -------
    out : torch.tensor
        Denoised image.

    """
    # im_type = image.dtype
    # if not im_type.kind == 'f':
    #     image = image.type(torch.float64)
    #     image = image/torch.abs(image.max()+image.min())

    if multichannel:
        out = torch.zeros_like(image)
        for c in range(image.shape[1]):
            out[:, c, ...] = _denoise_tv_chambolle_nd_torch(
                image[:, c, ...], weight, eps, n_iter_max
            )
    else:
        out = _denoise_tv_chambolle_nd_torch(image, weight, eps, n_iter_max)

    return out

# This code is mostly based on the code from the following repository:
# https://github.com/LeviBorodenko/motionblur
# It was adapted by Thibaut Modrzyk to output torch tensors instead of PIL images

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from numpy.random import uniform, triangular, beta
from math import pi
from pathlib import Path
from scipy.signal import convolve
import torch
import torch.nn as nn

# tiny error used for nummerical stability
eps = 0.001


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def norm(lst: list) -> float:
    """[summary]
    L^2 norm of a list
    [description]
    Used for internals
    Arguments:
        lst {list} -- vector
    """
    if not isinstance(lst, list):
        raise ValueError("Norm takes a list as its argument")

    if lst == []:
        return 0

    return (sum((i**2 for i in lst)))**0.5


def polar2z(r: np.ndarray, θ: np.ndarray) -> np.ndarray:
    """[summary]
    Takes a list of radii and angles (radians) and
    converts them into a corresponding list of complex
    numbers x + yi.
    [description]

    Arguments:
        r {np.ndarray} -- radius
        θ {np.ndarray} -- angle

    Returns:
        [np.ndarray] -- list of complex numbers r e^(i theta) as x + iy
    """
    return r * np.exp(1j * θ)


class MotionBlur(nn.Module):
    """[summary]
    Class representing a motion blur kernel of a given intensity.

    [description]
    Keyword Arguments:
            size {tuple} -- Size of the kernel in px times px
            (default: {(100, 100)})

            intensity {float} -- Float between 0 and 1.
            Intensity of the motion blur.

            :   0 means linear motion blur and 1 is a highly non linear
                and often convex motion blur path. (default: {0})

    Attribute:
    kernelMatrix -- Numpy matrix of the kernel of given intensity

    Properties:
    applyTo -- Applies kernel to image
               (pass as path, pillow image or np array)

    Raises:
        ValueError
    """

    def __init__(self, size: int = 61, intensity: float=0, channels:int=1):
        super().__init__()

        # check if intensity is float (int) between 0 and 1
        if type(intensity) not in [int, float, np.float32, np.float64]:
            raise ValueError("Intensity must be a number between 0 and 1")
        elif intensity < 0 or intensity > 1:
            raise ValueError("Intensity must be a number between 0 and 1")

        # saving args
        self.SIZE = (size, size)
        self.kernel_size = size
        self.INTENSITY = intensity

        # deriving quantities

        # we super size first and then downscale at the end for better
        # anti-aliasing
        self.SIZEx2 = tuple([2 * i for i in self.SIZE])
        self.x, self.y = self.SIZEx2

        # getting length of kernel diagonal
        self.DIAGONAL = (self.x**2 + self.y**2)**0.5

        # flag to see if kernel has been calculated already
        self.kernel_is_generated = False

        self.seq = nn.Sequential(
            nn.Conv2d(channels, channels, self.kernel_size, stride=1, padding=self.kernel_size//2, padding_mode='replicate', bias=False, groups=channels)
        )
    
        self.weights_init()
    
    def forward(self, x):
        """
        Perform a forward pass of the Gaussian blur operation.

        Args:
            x (torch.Tensor): The input tensor to be blurred.

        Returns:
            torch.Tensor: The blurred output tensor.
        """
        
        return self.seq(x)

    def _createPath(self):
        """[summary]
        creates a motion blur path with the given intensity.
        [description]
        Proceede in 5 steps
        1. Get a random number of random step sizes
        2. For each step get a random angle
        3. combine steps and angles into a sequence of increments
        4. create path out of increments
        5. translate path to fit the kernel dimensions

        NOTE: "random" means random but might depend on the given intensity
        """

        # first we find the lengths of the motion blur steps
        def getSteps():
            """[summary]
            Here we calculate the length of the steps taken by
            the motion blur
            [description]
            We want a higher intensity lead to a longer total motion
            blur path and more different steps along the way.

            Hence we sample

            MAX_PATH_LEN =[U(0,1) + U(0, intensity^2)] * diagonal * 0.75

            and each step: beta(1, 30) * (1 - self.INTENSITY + eps) * diagonal)
            """

            # getting max length of blur motion
            self.MAX_PATH_LEN = 0.75 * self.DIAGONAL * \
                (uniform() + uniform(0, self.INTENSITY**2))

            # getting step
            steps = []

            while sum(steps) < self.MAX_PATH_LEN:

                # sample next step
                step = beta(1, 30) * (1 - self.INTENSITY + eps) * self.DIAGONAL
                if step < self.MAX_PATH_LEN:
                    steps.append(step)

            # note the steps and the total number of steps
            self.NUM_STEPS = len(steps)
            self.STEPS = np.asarray(steps)

        def getAngles():
            """[summary]
            Gets an angle for each step
            [description]
            The maximal angle should be larger the more
            intense the motion is. So we sample it from a
            U(0, intensity * pi)

            We sample "jitter" from a beta(2,20) which is the probability
            that the next angle has a different sign than the previous one.
            """

            # same as with the steps

            # first we get the max angle in radians
            self.MAX_ANGLE = uniform(0, self.INTENSITY * pi)

            # now we sample "jitter" which is the probability that the
            # next angle has a different sign than the previous one
            self.JITTER = beta(2, 20)

            # initialising angles (and sign of angle)
            angles = [uniform(low=-self.MAX_ANGLE, high=self.MAX_ANGLE)]

            while len(angles) < self.NUM_STEPS:

                # sample next angle (absolute value)
                angle = triangular(0, self.INTENSITY *
                                   self.MAX_ANGLE, self.MAX_ANGLE + eps)

                # with jitter probability change sign wrt previous angle
                if uniform() < self.JITTER:
                    angle *= - np.sign(angles[-1])
                else:
                    angle *= np.sign(angles[-1])

                angles.append(angle)

            # save angles
            self.ANGLES = np.asarray(angles)

        # Get steps and angles
        getSteps()
        getAngles()

        # Turn them into a path
        ####

        # we turn angles and steps into complex numbers
        complex_increments = polar2z(self.STEPS, self.ANGLES)

        # generate path as the cumsum of these increments
        self.path_complex = np.cumsum(complex_increments)

        # find center of mass of path
        self.com_complex = sum(self.path_complex) / self.NUM_STEPS

        # Shift path s.t. center of mass lies in the middle of
        # the kernel and a apply a random rotation
        ###

        # center it on COM
        center_of_kernel = (self.x + 1j * self.y) / 2
        self.path_complex -= self.com_complex

        # randomly rotate path by an angle a in (0, pi)
        self.path_complex *= np.exp(1j * uniform(0, pi))

        # center COM on center of kernel
        self.path_complex += center_of_kernel

        # convert complex path to final list of coordinate tuples
        self.path = [(i.real, i.imag) for i in self.path_complex]

    def weights_init(self, save_to: Path=None, show: bool=False):
        # check if we haven't already generated a kernel
        if self.kernel_is_generated:
            return None

        # get the path
        self._createPath()

        # Initialise an image with super-sized dimensions
        # (pillow Image object)
        kernel_image = Image.new("RGB", self.SIZEx2)

        # ImageDraw instance that is linked to the kernel image that
        # we can use to draw on our kernel_image
        self.painter = ImageDraw.Draw(kernel_image)

        # draw the path
        self.painter.line(xy=self.path, width=int(self.DIAGONAL / 150))

        # applying gaussian blur for realism
        self.kernel_image = kernel_image.filter(
            ImageFilter.GaussianBlur(radius=int(self.DIAGONAL * 0.01)))

        # Resize to actual size
        kernel_image = kernel_image.resize(
            self.SIZE, resample=Image.LANCZOS)

        # convert to gray scale
        kernel_image = kernel_image.convert("L")

        self.k = np.asarray(kernel_image, dtype=np.float32)
        self.k /= np.sum(self.k)
        self.k = torch.from_numpy(self.k)
        
        for name, f in self.named_parameters():
            f.data.copy_(self.k)

    def get_kernel(self):
        """
        Get the Gaussian kernel used for blurring.

        Returns:
            torch.Tensor: The Gaussian kernel.
        """
        return self.k

import numpy as np
import matplotlib.pyplot as plt
import torch

from PIL import Image

ref = Image.open('input/baseball.png')
ref = torch.from_numpy(np.array(ref)).unsqueeze(0).float() / 255.0
ref = ref[:, :-1, :-1, :-1]
ref = ref.permute(0, 3, 1, 2)

k_ref = gaussianblur.get_kernel().view(1, 1, k_size, k_size)
res = richardson_lucy(ref, ref, k_ref, steps=50, clip=True, filter_epsilon=1e-6)


plt.imshow(res[0].permute(1, 2, 0).cpu().numpy())