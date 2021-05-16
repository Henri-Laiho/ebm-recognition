import random

from sklearn.metrics import accuracy_score, f1_score
from tensorflow.python.platform import flags

from image_sampling import sample_embedded_vector
from identity_data import CelebAPairsWithIdentity
from models import CelebASplitModel, CelebASplit2Model
import os.path as osp
import numpy as np
import os
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import scipy.stats
from matplotlib.colors import hsv_to_rgb
from scipy.misc import imsave
from experiment_conf import search

print('cuda:', torch.cuda.is_available(), torch.cuda.device_count())
device = torch.device("cuda:0")
print(device)


class DFLAGS:
    num_steps = None
    test_size = None
    step_valley_depth = None
    step_valley_sigma = None
    step_lr = None
    step_noise = None


flags.DEFINE_integer('seed', 1, 'random seed')
flags.DEFINE_integer('samples_per_ground', 1, 'for how many samples we should keep the other image constant')
flags.DEFINE_integer('cycles_per_side', 1,
                     'number of samples_per_ground cycles before swapping the side of the constant image; 1 means ignore this feature')
flags.DEFINE_integer('min_occurrences', 10,
                     'if there are less than this many occurrences of a celebrity in the dataset, then drop all images of that celebrity')
flags.DEFINE_float('pos_neg_balance', 0.5,
                   'the balance of negative and positive image pairs: 1.0=100% positive, 0.5 = 50/50, 0.0=100% negative')
flags.DEFINE_float('gradient_clip', 0.01, 'threshold for gradient clipping')
flags.DEFINE_integer('verbose', 0, 'verbosity, 1 or 0')
flags.DEFINE_integer('figsize_w', 10, 'plot width X10px')
flags.DEFINE_integer('figsize_h', 7, 'plot height X10px')

# walks
flags.DEFINE_integer('render_resolution', None,
                     'Energy rendering resolution; 350 was used for plots; Use None for no energy landscape rendering')
flags.DEFINE_bool('plot_walks', True, 'Whether to plot walks. Useful if test size is low.')
flags.DEFINE_bool('refine_walk_steps_to_images', False, 'Whether to plot walks. Useful if test size is low.')
flags.DEFINE_integer('num_identities', None,
                     'Number of identities to use from the dataset (use None for all). Use this to restrict to only 3 identities for example.')
flags.DEFINE_integer('model_version', 1,
                     'Embedded space version. The embedded space variations a, b, c described in the thesis, are defined here as follows: ' +
                     'a:(model_version=0, scales=(2,)), ' +
                     'b:(model_version=1, scales=(2,)), ' +
                     'c:(model_version=0, scales=(0,1,2)); ' +
                     'where scales is defined in experiment_conf.py')
FLAGS = flags.FLAGS

figsize = [FLAGS.figsize_w, FLAGS.figsize_h]  # plt plot size


def id_to_rgb(id):
    return hsv_to_rgb(((id % 256) / 256, 1.0, 1.0))


def energy_distance(model, label, z1, z2, n=9):
    result = 0
    tz1 = torch.from_numpy(z1).cuda()
    tz2 = torch.from_numpy(z2).cuda()
    tdz = (tz2 - tz1) / (n + 1)
    tz_mod = tz1 + tdz
    for i in range(n):
        energy = model.forward_top(tz_mod, label)
        tz_mod += tdz
        result += energy.detatch().cpu().numpy()[0]
        del energy
    del tz1, tz2, tz_mod
    return result / n


def walk_eval(model_list, select_idx, dataset):
    n = 1
    labels = []

    for six in select_idx:
        label_ix = np.eye(2)[six]
        label_batch = np.tile(label_ix[None, :], (n, 1))
        label = torch.Tensor(label_batch).cuda()
        labels.append(label)

    def walk_pair(model, label, z_mod1, z_mod2):
        walk_zs = []
        for i in range(DFLAGS.num_steps):
            im_noise = torch.randn_like(z_mod1).detach()

            im_noise.normal_()
            z_mod1 = z_mod1 + DFLAGS.step_noise * im_noise

            z_mod1.requires_grad_(requires_grad=True)

            energy_next_z1 = model.forward_top(z_mod1, label)
            gaussian_distance = torch.multiply(torch.tensor(-DFLAGS.step_valley_depth * model.depth_mod),
                                               torch.exp(torch.divide(
                                                   torch.sum(torch.square(torch.subtract(z_mod1, z_mod2)),
                                                             dim=model.embedded_dims),  # (1, 2, 3)
                                                   torch.tensor(-DFLAGS.step_valley_sigma * model.sigma_mod)).double()))

            if FLAGS.verbose > 1:
                print("step: ", i, energy_next_z1.mean(), "gd: ", gaussian_distance.mean())

            im_grad1 = torch.autograd.grad([torch.add(energy_next_z1.sum(), gaussian_distance.sum())], [z_mod1])[0]

            z_mod1 = z_mod1 - DFLAGS.step_lr * model.lr_mod * torch.clamp(im_grad1, -FLAGS.gradient_clip,
                                                                          FLAGS.gradient_clip)
            z_mod1 = z_mod1.detach()

            if FLAGS.plot_walks:
                walk_zs.append(z_mod1.cpu().numpy())

        return z_mod1, z_mod2, walk_zs

    start_distances = []
    distances = []
    distance_pairs_list = []
    y_true = []
    direct_energies = []
    final_energies = []
    energy_decreases = []
    identities = []
    annotations = []

    test_size = DFLAGS.test_size
    rand = random.Random(x=FLAGS.seed)  # seed
    testset_index = rand.choices(range(len(dataset)), k=test_size)
    start_zs = []
    end_zs = []
    end_zrs = []
    start_energies = []
    end_energies = []
    end_energiesr = []
    walk_zs = []
    img_fnames = []

    for i, idx in enumerate(testset_index):
        (img1_fname, img2_fname), (anno1, anno2), (id1, id2), id_label = dataset[idx]

        identities.append((id1, id2))
        annotations.append((anno1, anno2))
        img_fnames.append((img1_fname, img2_fname))
        img1, image_size = dataset.load_im(img1_fname)
        img2, image_size2 = dataset.load_im(img2_fname)

        if FLAGS.verbose:
            print('\rtesting: %d/%d, label=%s ' % (i + 1, test_size, id_label[1]), end='')

        z1s = []
        z2s = []
        z1_finals = []
        z2_finals = []
        z1r_finals = []
        z2r_finals = []
        start_energy1s = []
        start_energy2s = []
        end_energy1s = []
        end_energy2s = []
        end_energy1rs = []
        end_energy2rs = []
        models_walk_zs = []
        models_walk_zrs = []
        for model, label in zip(model_list, labels):
            x1 = torch.from_numpy(img1[None, :]).float().permute(0, 3, 1, 2).cuda()
            x2 = torch.from_numpy(img2[None, :]).float().permute(0, 3, 1, 2).cuda()
            z1 = model.forward_bottom(x1, label)
            z2 = model.forward_bottom(x2, label)
            z1s.append(z1.detach().cpu())
            z2s.append(z2.detach().cpu())

            z1f, z2f, z_mods = walk_pair(model, label, z1, z2)
            z1fr, z2fr, zr_mods = walk_pair(model, label, z2, z1)

            models_walk_zs.append(z_mods)
            models_walk_zrs.append(zr_mods)
            z1r_finals.append(z1fr.detach().cpu())
            z2r_finals.append(z2fr.detach().cpu())

            z1_finals.append(z1f.detach().cpu())
            z2_finals.append(z2f.detach().cpu())

            start_energy1s.append(model.forward_top(z1, label).detach().cpu())
            start_energy2s.append(model.forward_top(z2, label).detach().cpu())
            end_energy1s.append(model.forward_top(z1f, label).detach().cpu())
            end_energy2s.append(model.forward_top(z2f, label).detach().cpu())
            end_energy1rs.append(model.forward_top(z1fr, label).detach().cpu())
            end_energy2rs.append(model.forward_top(z2fr, label).detach().cpu())
            del z1
            del z2
            del z1f
            del z2f
            del z1fr
            del z2fr
            torch.cuda.empty_cache()

        distance_pairs = [(np.linalg.norm(z_f1 - z_f2), np.linalg.norm(z_f1r - z_f2r)) for
                          z_f1, z_f2, z_f1r, z_f2r in zip(z1_finals, z2_finals, z1r_finals, z2r_finals)]
        distance = sum(np.max(x) for x in distance_pairs)
        distance_pairs_list.append(distance_pairs)
        start_distance = sum(np.linalg.norm(z1 - z2) for z1, z2 in zip(z1s, z2s))

        distances.append(distance)
        start_distances.append(start_distance)
        for j, (de1, de2, fe1, fe2) in enumerate(zip(start_energy1s, start_energy2s, end_energy1s, end_energy2s)):
            direct_energies.append([de1, de2])
            final_energies.append([fe1, fe2])
            energy_decreases.append([de1 - fe1, de2 - fe2])

        y_true.append(id_label[1])

        start_zs.append(tuple([z.numpy() for z in zs] for zs in (z1s, z2s)))
        start_energies.append(tuple([e.numpy()[0][0] for e in es] for es in (start_energy1s, start_energy1s)))
        end_zs.append(tuple([z.numpy() for z in zs] for zs in (z1_finals, z2_finals)))
        end_energies.append(tuple([e.numpy()[0][0] for e in es] for es in (end_energy1s, end_energy2s)))
        end_zrs.append(tuple([z.numpy() for z in zs] for zs in (z2r_finals, z1r_finals)))
        end_energiesr.append(tuple([e.numpy()[0][0] for e in es] for es in (end_energy2rs, end_energy1rs)))

        if FLAGS.plot_walks:
            walk_zs.append(tuple(z_mods for z_mods in (models_walk_zs, models_walk_zrs)))

    distance_pairs_flat = np.array(distance_pairs_list).reshape(test_size * len(model_list), 2)
    correlation = scipy.stats.pearsonr(distance_pairs_flat[:, 0], distance_pairs_flat[:, 1])

    start_zs = np.array(start_zs)
    start_energies = np.array(start_energies)
    end_zs = np.array(end_zs)
    end_energies = np.array(end_energies)

    end_zrs = np.array(end_zrs)
    end_energiesr = np.array(end_energiesr)
    walk_zs = np.array(walk_zs)

    shape = [test_size * 2, len(model_list)] + list(start_zs.shape[3:])
    energies_shape = [test_size * 2, len(model_list)] + list(start_energies.shape[3:])

    start_zs = start_zs.reshape(shape)
    start_energies = start_energies.reshape(energies_shape)
    end_zs = end_zs.reshape(shape)
    end_energies = end_energies.reshape(energies_shape)
    end_zrs = end_zrs.reshape(shape)
    end_energiesr = end_energiesr.reshape(energies_shape)

    start_zs_flat = start_zs.reshape(shape[:2] + [np.product(shape[2:])])
    end_zs_flat = end_zs.reshape(shape[:2] + [np.product(shape[2:])])
    end_zrs_flat = end_zrs.reshape(shape[:2] + [np.product(shape[2:])])
    if FLAGS.plot_walks:
        walk_zs_flat = walk_zs.reshape(list(walk_zs.shape[:4]) + [np.product(walk_zs.shape[4:])])
    ids_flat = np.array(identities).flatten()

    for i in range(shape[1]):

        if FLAGS.refine_walk_steps_to_images:
            for pair in range(DFLAGS.test_size):
                for direction in range(2):
                    fname = img_fnames[pair][direction]
                    img1, image_size = dataset.load_im(fname)
                    walk_imgs = [img1]
                    for step in range(DFLAGS.num_steps):
                        img = sample_embedded_vector(walk_zs[pair, direction, i, step], [model_list[i]],
                                                     [select_idx[i]],
                                                     start_im=walk_imgs[-1])
                        walk_imgs.append(img)

                    output = np.array(walk_imgs)
                    output = output.reshape((-1, DFLAGS.num_steps + 1, 128, 128, 3)).transpose((0, 2, 1, 3, 4)).reshape(
                        (-1, 128 * (DFLAGS.num_steps + 1), 3))
                    name = 'walk=%d_model=%d_direction=%d_start=%s.png' % (pair, i, direction, fname)
                    imsave(os.path.join('celeba_combine', 'walk_images', name), output)
                    print('wrote file', name, 'start-identity =', identities[pair][direction],
                          id_to_rgb(identities[pair][direction]))

        plt.rcParams['figure.figsize'] = figsize
        model = model_list[i]
        label = labels[i]
        pca = PCA(n_components=2)

        ids = np.concatenate((ids_flat, ids_flat, ids_flat))
        z_flat = np.concatenate((start_zs_flat[:, i, :], end_zs_flat[:, i, :], end_zrs_flat[:, i, :]))
        energies = np.concatenate((start_energies[:, i], end_energies[:, i], end_energiesr[:, i]))

        if FLAGS.plot_walks:
            pca_walk_zs = walk_zs_flat[:, :, i, :, :]
            pca_walk_zs_shape = pca_walk_zs.shape
            pca_walk_zs = pca_walk_zs.reshape([np.product(pca_walk_zs_shape[:-1])] + [pca_walk_zs_shape[-1]])
            pca.fit(np.concatenate((z_flat, pca_walk_zs)))
            pca_walk_pcs = pca.transform(pca_walk_zs)
            walk_pcs = pca_walk_pcs.reshape(list(pca_walk_zs_shape[:-1]) + [2])
        else:
            pca.fit(z_flat)

        pc = pca.transform(z_flat)
        if FLAGS.verbose:
            print('Variance ratios:', pca.explained_variance_ratio_)

        energies = -energies
        x_data = np.array([x[0] for x in pc])
        y_data = np.array([x[1] for x in pc])

        render_resolution = FLAGS.render_resolution
        x_render = []
        y_render = []
        render_energies = []
        scaled_render_energies = None
        if render_resolution is not None:
            xi, xa, yi, ya = x_data.min(), x_data.max(), y_data.min(), y_data.max()
            dx, dy = (xa - xi) * 0.1, (ya - yi) * 0.1
            for i, x in enumerate(np.linspace(xi - dx, xa + dx, num=render_resolution)):
                if FLAGS.verbose:
                    print('\rRendering... %d/%d' % (i, render_resolution), end='')
                for y in np.linspace(yi - dy, ya + dy, num=render_resolution):
                    x_render.append(x)
                    y_render.append(y)
                    z = np.reshape(pca.inverse_transform([(x, y)]), newshape=[1] + model.embedded_shape)
                    render_energies.append(model.forward_top(torch.from_numpy(z).float().cuda(), label)
                                           .detach().cpu().numpy()[0][0])
            render_energies = np.array(render_energies)
            scaled_render_energies = (render_energies - render_energies.min()) / (
                    render_energies.max() - render_energies.min())
            if FLAGS.verbose:
                print()

        zs_inverse_pc = pca.inverse_transform(pc)
        ppc = pca.fit_transform(zs_inverse_pc)
        if FLAGS.verbose:
            print('PCA error:', np.mean(np.linalg.norm(zs_inverse_pc - z_flat, axis=1)))
        x_data_ppca = np.array([x[0] for x in ppc])
        y_data_ppca = np.array([x[1] for x in ppc])

        z = np.reshape(zs_inverse_pc, newshape=[len(zs_inverse_pc)] + model.embedded_shape)
        energies_ppca = model.forward_top(torch.from_numpy(z).float().cuda(), label).detach().cpu().numpy()[:, 0]

        for j, (marker, e_scale) in enumerate(zip(['o', '1'] + (['2']), [1, 2, 2])):
            es = energies[j * shape[0]:(j + 1) * shape[0]]
            data = {
                'a': x_data[j * shape[0]:(j + 1) * shape[0]], 'b': y_data[j * shape[0]:(j + 1) * shape[0]],
                'c': [id_to_rgb(x) for x in ids[j * shape[0]:(j + 1) * shape[0]]],
                'd': ((es - es.min()) / (es.max() - es.min()) * 90 + 50) * e_scale,
            }
            plt.scatter('a', 'b', c='c', s='d', marker=marker, data=data, alpha=0.7)
        if FLAGS.plot_walks:
            for pair, ids in zip(walk_pcs, identities):
                for direction, id in zip(pair, ids):
                    data_x = np.array([x[0] for x in direction])
                    data_y = np.array([x[1] for x in direction])
                    dataw = {
                        'a': data_x,
                        'b': data_y,
                    }
                    plt.plot('a', 'b', c=id_to_rgb(id), marker='>', data=dataw, linewidth=2, markersize=4, alpha=0.6)

            for j in range(0, shape[0], 2):
                x = [x_data[j], x_data[j + 1]]
                y = [y_data[j], y_data[j + 1]]
                plt.plot(x, y, c='black', alpha=0.6, linewidth=2)

            if render_resolution is not None:
                data2 = {
                    'a': x_render, 'b': y_render, 'c': [(e, 1 - e, 0.2) for e in scaled_render_energies],
                }
                plt.scatter('a', 'b', c='c', s=1, data=data2, alpha=0.25)
            data3 = {
                'a': x_data_ppca, 'b': y_data_ppca, 'c': [id_to_rgb(x) for x in ids],
                'd': (energies_ppca - energies_ppca.min()) / (energies_ppca.max() - energies_ppca.min()) * 100,
            }
            # plt.scatter('a', 'b', c='c', s='d', marker='*', data=data3, alpha=0.4)
            plt.title('%s (%s)' % (model.name, 'true' if int(label[0][1]) else 'false'))
            plt.show()

    distances = np.array(distances)
    start_distances = np.array(start_distances)
    y_true = np.array(y_true)

    pos = distances[y_true == 1]
    neg = distances[y_true == 0]

    # pos = pos[pos < 1E308]  # nan & inf removal
    # pos = pos[pos > -1E308]

    spos = start_distances[y_true == 1]
    sneg = start_distances[y_true == 0]

    pos_mean = np.mean(pos)
    pos_std = np.std(pos)
    neg_mean = np.mean(neg)
    neg_std = np.std(neg)
    spos_mean = np.mean(spos)
    spos_std = np.std(spos)
    sneg_mean = np.mean(sneg)
    sneg_std = np.std(sneg)

    if FLAGS.verbose:
        print('Fwd/Bwd walk correlation:', correlation)
        print('   with walk positives mean distance:', pos_mean, 'negatives:', neg_mean)
        print('   with walk positives stdv distance:', pos_std, 'negatives:', neg_std)
        print('without walk positives mean distance:', spos_mean, 'negatives:', sneg_mean)
        print('without walk positives stdv distance:', spos_std, 'negatives:', sneg_std)
        print('With walk diff:', neg_mean - pos_mean, 'without walk diff:', sneg_mean - spos_mean)
        print('rating:', neg_mean - pos_mean - sneg_mean + spos_mean)
        print('mean energy decrease during walk:', np.mean(energy_decreases))

    walk_preds = distances <= (pos_mean + neg_mean) / 2
    acc = accuracy_score(y_true, walk_preds)
    f1 = f1_score(y_true, walk_preds)
    if FLAGS.verbose:
        print('actual positives:', sum(y_true))
        print('walk accuracy:', acc, 'walk f1-score:', f1)
        print('walk positives:', sum(walk_preds))
        print('acc,f1,wpmd,wnmd,wpstd,wnstd,spmd,snmd,spstd,snstd:')
    print(acc, f1, pos_mean, neg_mean, pos_std, neg_std, spos_mean, sneg_mean, spos_std, sneg_std, sep='\t')


def walk_main(models, resume_iters, select_idx, configs, scales=(2,)):
    model_list = []

    for model, resume_iter, (lrm, sigmam, depthm) in zip(models, resume_iters, configs):
        model_path = osp.join("celeba_combine", model, "model_{}.pth".format(resume_iter))
        checkpoint = torch.load(model_path)
        FLAGS_model = checkpoint['FLAGS']
        if FLAGS.model_version == 0:
            model_base = CelebASplitModel(FLAGS_model, scales=scales, name='%s_%s' % (model, resume_iter), lr_mod=lrm,
                                          sigma_mod=sigmam, depth_mod=depthm)
        elif FLAGS.model_version == 1:
            model_base = CelebASplit2Model(FLAGS_model, scales=scales, name='%s_%s' % (model, resume_iter), lr_mod=lrm,
                                           sigma_mod=sigmam, depth_mod=depthm)
        else:
            raise RuntimeError
        model_base.load_state_dict(checkpoint['ema_model_state_dict_0'])
        model_base = model_base.cuda()
        model_list.append(model_base)

    dataset = CelebAPairsWithIdentity(num_identities=FLAGS.num_identities,
                                      minimum_occurrences=FLAGS.min_occurrences,
                                      random_state=FLAGS.seed,
                                      samples_per_ground=FLAGS.samples_per_ground,
                                      cycles_per_side=FLAGS.cycles_per_side,
                                      pos_probability=FLAGS.pos_neg_balance)

    walk_eval(model_list, select_idx, dataset)


if __name__ == "__main__":
    print('Short output format for piping to a tsv file:')
    print(
        'config\tscales\twalk_rate\tvalley_depth\tvalley_sigma\tnoise\tnum_steps\ttest_size\t\taccuracy\tf1-score\tpost-walk_positives_mean\tpost-walk_negatives_mean\tpost-walk_positives_stdev\tpost-walk_negatives_stdev\tpre-walk_positives_mean\tpre-walk_negatives_mean\tpre-walk_positives_stdev\tpre-walk_negatives_stdev')
    for config, scales, walk_rate, valley_depth, valley_sigma, noise, num_steps, test_size in search:
        DFLAGS.step_lr = walk_rate
        DFLAGS.step_valley_depth = valley_depth
        DFLAGS.step_valley_sigma = valley_sigma
        DFLAGS.step_noise = noise
        DFLAGS.num_steps = num_steps
        DFLAGS.test_size = test_size
        if FLAGS.verbose:
            print('running with params: ', end='')
        print(config, scales, walk_rate, valley_depth, valley_sigma, noise, num_steps, test_size,
              sep='\t', end='\n' if FLAGS.verbose else '\t\t')

        model_info = {
            'male': {
                'fq_name': 'celeba_128_male_2',
                'resume_iter': "latest",
            },
            'old': {
                'fq_name': 'celeba_128_old_2',
                'resume_iter': "6000",
            },
            'smiling': {
                'fq_name': 'celeba_128_smiling_2',
                'resume_iter': "13000",
            },
            'wavy_hair': {
                'fq_name': 'celeba_128_wavy_hair_2',
                'resume_iter': "9000",
            }
        }
        ##################################
        # Settings for the visualisation (which models to use)
        models = [model_info[x[0]]['fq_name'] for x in config]
        resume_iters = [model_info[x[0]]['resume_iter'] for x in config]
        select_idx = [int(x[1]) for x in config]
        configs = [x[2] for x in config]

        DFLAGS.step_lr = DFLAGS.step_lr / len(models)

        walk_main(models, resume_iters, select_idx, configs, scales=scales)
