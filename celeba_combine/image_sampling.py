from tensorflow.python.platform import flags

import numpy as np
from scipy.misc import imsave
from torchvision import transforms
from PIL import Image
import torch

flags.DEFINE_float('sampling_step_lr', 20.0, 'size of gradient descent size')
flags.DEFINE_integer('sampling_num_steps', 120, 'number of steps to optimize the label')

FLAGS = flags.FLAGS


def sample_embedded_vector(z_target, model_list, select_idx, outfile=None, n=8, start_im=None):
    """
    Approximates an embedded space vector in the input space. Approximate inverse operation of applying the model.

    @param z_target: The embedded space vector to be approximated.
    @param model_list: Single-element list that specifies the model object to use for sampling
    @param select_idx: Single-element list that specifies the condition variable for the model
    @param outfile: File name where to save the image. If None, the image will not be saved
    @param n: Number of times to repeat the sampling process. The one with least error to z_target, is returned.
    @param start_im: Image to start the sampling from. If None sampling will be initialized randomly.
                     In case of walks we can use the previous image as start_im.
    @return: An image that produces an output z close to z_target when input to the models
             specified by model_list and select_idx.
    """

    labels = []

    for six in select_idx:
        label_ix = np.eye(2)[six]
        label_batch = np.tile(label_ix[None, :], (n, 1))
        label = torch.Tensor(label_batch).cuda()
        labels.append(label)

    z_target_gpu = torch.from_numpy(z_target).cuda()
    im = torch.rand(n, 3, 128, 128)
    im_noise = torch.randn_like(im).detach().cuda()

    def get_color_distortion(s=1.0):
        # s is the strength of color distortion.
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.4 * s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([
            rnd_color_jitter,
            rnd_gray])
        return color_distort

    color_transform = get_color_distortion()

    im_size = 128
    transform = transforms.Compose(
        [transforms.RandomResizedCrop(im_size, scale=(0.08, 1.0)), transforms.RandomHorizontalFlip(), color_transform,
         transforms.ToTensor()])

    if start_im is not None:
        im = (im + torch.from_numpy(start_im[None, :]).float().permute(0, 3, 1, 2)).cuda()
    else:
        im = im.cuda()
        # First get good initializations for sampling
        for i in range(10):
            for i in range(20):
                im_noise.normal_()
                im = im + 0.001 * im_noise
                im.requires_grad_(requires_grad=True)
                z = None

                for model, label in zip(model_list, labels):
                    z_next = model.forward_bottom(im, label)
                    z = z_next if z is None else z_next + z

                loss = ((z - z_target_gpu) ** 2).sum() / n
                im_grad = torch.autograd.grad([loss], [im])[0]

                im = im - FLAGS.sampling_step_lr * im_grad
                im = im.detach()

                im = torch.clamp(im, 0, 1)

            im = im.detach().cpu().numpy().transpose((0, 2, 3, 1))
            im = (im * 255).astype(np.uint8)

            ims = []
            for i in range(im.shape[0]):
                im_i = np.array(transform(Image.fromarray(np.array(im[i]))))
                ims.append(im_i)

            im = torch.Tensor(np.array(ims)).cuda()

    # Then refine the images
    for i in range(FLAGS.sampling_num_steps):
        im_noise.normal_()
        im = im + 0.005 * im_noise
        im.requires_grad_(requires_grad=True)
        z = None

        for model, label in zip(model_list, labels):
            z_next = model.forward_bottom(im, label)
            z = z_next if z is None else z_next + z

        loss = ((z - z_target_gpu) ** 2).sum() / n
        im_grad = torch.autograd.grad([loss], [im])[0]

        im = im - FLAGS.sampling_step_lr * im_grad
        im = im.detach()

        im = torch.clamp(im, 0, 1)

    z = None
    for model, label in zip(model_list, labels):
        z_next = model.forward_bottom(im, label)
        z = z_next if z is None else z_next + z
    losses = ((z.detach().cpu().numpy() - z_target) ** 2).sum(axis=model_list[0].embedded_dims)  # (1, 2, 3)
    best = np.argmin(losses)

    if outfile is not None:
        output = im.detach().cpu().numpy()
        output = output.transpose((0, 2, 3, 1))
        output = output.reshape((-1, n, 128, 128, 3)).transpose((0, 2, 1, 3, 4)).reshape((-1, 128 * n, 3))
        imsave(outfile, output)
    return im[best].detach().cpu().numpy().transpose((1, 2, 0))
