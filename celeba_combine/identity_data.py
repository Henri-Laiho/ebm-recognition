import os
import random

import pandas as pd
import numpy as np
from tensorflow.python.platform import flags
from torch.utils.data import Dataset
from scipy.misc import imread, imresize

FLAGS = flags.FLAGS


class CelebAPairsWithIdentity(Dataset):
    """
    Identity-based CelebA pair dataloader. The pairs consist of 2 images A and B. Provides 4-tuples of:
    2-tuple of file name A, file name B,
    2-tuple of annotations for A, annotations for B,
    2-tuple of identity of A, identity of B,
    label as integer equal to identity of A == identity of B.

    To load the image use self.load_im(file_name)
    Assumes that the dataset is in the directory CelebA: images in CelebA/img_align_celeba/, annotations in CelebA/Anno/
    """

    def __init__(self, samples_per_ground=None, cycles_per_side=2, minimum_occurrences=5,
                 random_state=None, pos_probability=0.5, num_identities=10):
        if samples_per_ground is None:
            samples_per_ground = FLAGS.batch_size * 3
        self.rand = random.Random(random_state)
        self.path = os.path.join("CelebA", "img_align_celeba")
        self.ident = pd.read_csv("CelebA/Anno/identity_CelebA.txt", sep="\s+", names=['file', 'celeb_id'])
        self.annotations = pd.read_csv("CelebA/Anno/list_attr_celeba.csv", sep="\s+")
        self.samples_per_ground = samples_per_ground
        self.cycles_per_side = cycles_per_side
        self.pos_probability = pos_probability
        self.side_state = None
        self.cycle = 0
        self.side = 0
        self.sample_counter = 0
        self.num_identities = num_identities

        ivc = self.ident.celeb_id.value_counts()
        self.ivcthr = ivc[ivc >= minimum_occurrences]

        if num_identities is not None:
            self.ivcthr = self.ivcthr[:num_identities]

        imgs_of_celeb = {}
        for x in self.ivcthr.index:
            imgs_of_celeb[x] = self.ident[self.ident.celeb_id == x].file
        self.imgs_of_celeb = imgs_of_celeb

        self.fnames = pd.Series()
        img_to_celeb = {}
        for x in imgs_of_celeb:
            labels = [y for y in imgs_of_celeb[x]]
            for label in labels:
                img_to_celeb[label] = x
            self.fnames = self.fnames.append(imgs_of_celeb[x])
        self.img_to_celeb = img_to_celeb
        self.fnames = self.fnames.reset_index(drop=True)

    def __len__(self):
        return self.fnames.shape[0]

    def __getitem__(self, index):

        label = self.rand.choices([0, 1], weights=[1 - self.pos_probability, self.pos_probability])[0]
        if label:  # positive
            if self.side_state is not None:
                side_fname, celeb = self.side_state
                main_fname = self.imgs_of_celeb[celeb].iloc[
                    len(self.imgs_of_celeb[celeb]) - 1 - index % len(self.imgs_of_celeb[celeb])]
            else:
                main_fname = self.fnames[index]
                celeb = self.get_id_for_fname(main_fname)
                side_fname = self.imgs_of_celeb[celeb].iloc[
                    len(self.imgs_of_celeb[celeb]) - 1 - index % len(self.imgs_of_celeb[celeb])]
        else:  # negative
            if self.side_state is not None:
                side_fname, celeb = self.side_state
                while self.get_id_for_fname(self.fnames[index % len(self)]) == celeb:
                    index += len(self.imgs_of_celeb)
                main_fname = self.fnames[index % len(self)]
            else:
                main_fname = self.fnames[index]
                celeb = self.get_id_for_fname(main_fname)
                side_fname = self.rand.choice(self.fnames)
                while self.get_id_for_fname(side_fname) == celeb:
                    side_fname = self.rand.choice(self.fnames)

        im1, im2 = (main_fname, side_fname) if self.side else (side_fname, main_fname)
        annotation1 = self.annotations.loc[im1]
        annotation2 = self.annotations.loc[im2]
        id1 = self.get_id_for_fname(im1)
        id2 = self.get_id_for_fname(im2)

        self.side_state = side_fname, celeb

        self.sample_counter += 1
        if self.sample_counter % self.samples_per_ground == 0:
            self.side_state = None
            self.cycle += 1
            if self.cycle % self.cycles_per_side == 0:
                self.side = not self.side

        label = np.eye(2)[label]
        return (im1, im2), (annotation1, annotation2), (id1, id2), label

    def get_id_for_fname(self, fname):
        return self.img_to_celeb[fname]

    def load_im(self, fname):
        path = os.path.join(self.path, fname)
        im = imread(path)
        im = imresize(im, (128, 128))
        image_size = 128
        im = im / 255.
        return im, image_size


class CelebAIdentities(Dataset):
    """
    Simple identity-based CelebA dataloader. Provides triplets of:
    image file name as a str,
    annotations as a list,
    identity as an integer.

    To load the image use self.load_im(file_name)
    Assumes that the dataset is in the directory CelebA: images in CelebA/img_align_celeba/, annotations in CelebA/Anno/
    """

    def __init__(self, minimum_occurrences=5,
                 random_state=None, num_identities=10, data_mapper=None):
        self.data_mapper = data_mapper
        self.rand = random.Random(random_state)
        self.path = os.path.join("CelebA", "img_align_celeba")
        self.ident = pd.read_csv("CelebA/Anno/identity_CelebA.txt", sep="\s+", names=['file', 'celeb_id'])
        self.annotations = pd.read_csv("CelebA/Anno/list_attr_celeba.csv", sep="\s+")
        self.num_identities = num_identities
        self.side_state = None
        self.cycle = 0
        self.side = 0
        self.sample_counter = 0

        ivc = self.ident.celeb_id.value_counts()
        self.ivcthr = ivc[ivc >= minimum_occurrences]

        if num_identities is not None:
            self.ivcthr = self.ivcthr[:num_identities]

        imgs_of_celeb = {}
        for x in self.ivcthr.index:
            imgs_of_celeb[x] = self.ident[self.ident.celeb_id == x].file
        self.imgs_of_celeb = imgs_of_celeb

        self.fnames = pd.Series()
        img_to_celeb = {}
        for x in imgs_of_celeb:
            labels = [y for y in imgs_of_celeb[x]]
            for label in labels:
                img_to_celeb[label] = x
            self.fnames = self.fnames.append(imgs_of_celeb[x])
        self.img_to_celeb = img_to_celeb
        self.fnames = self.fnames.reset_index(drop=True)

    def __len__(self):
        return self.fnames.shape[0]

    def __getitem__(self, index):
        main_fname = self.fnames[index]
        celeb = self.get_id_for_fname(main_fname)
        annotation = self.annotations.loc[main_fname]

        if self.data_mapper is not None:
            return self.data_mapper(self, main_fname, annotation, celeb)
        return main_fname, annotation, celeb

    def get_id_for_fname(self, fname):
        return self.img_to_celeb[fname]

    def load_im(self, fname):
        path = os.path.join(self.path, fname)
        im = imread(path)
        im = imresize(im, (128, 128))
        image_size = 128
        im = im / 255.
        return im, image_size
