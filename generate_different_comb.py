import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from dataloader import get_loader
from models.model_main import ModelMain
from models.transformers import denumericalize
from options import get_parser_main_model
from data_utils.svg_utils import render
from models.util_funcs import svg2img, cal_iou
from itertools import combinations, cycle, repeat
import pickle
from tqdm import tqdm
from time import time
import torch.multiprocessing as mp
from copy import deepcopy
from typing import Any


# CURRENTLY NOT WORKING!
# PROBLEM WITH TEST LOADER PARALLELIZATION


class ProcessingRes:
    def __init__(self, ind_min, ind_max, iou_max, comb_mean) -> None:
        self.ind_min = ind_min
        self.val_min = iou_max[ind_min]

        self.ind_max = ind_max
        self.val_min_max = iou_max[ind_max]

        self.comb_mean = comb_mean


# Generate combinations of letters for n fonts
# checking eng alphabet


def stat_main_model(opts):

    dir_res = os.path.join("./experiments/", opts.name_exp, "results")
    try:
        os.mkdir(dir_res)
    except:
        pass
    path_ckpt = os.path.join(
        "experiments", opts.name_exp, "checkpoints", opts.name_ckpt
    )
    charset = open(f"./data/char_set/{opts.language}.txt", "r").read()

    test_loader = get_loader(
        opts.data_root,
        opts.img_size,
        opts.language,
        opts.char_num,
        opts.max_seq_len,
        opts.dim_seq,
        opts.batch_size,
        "test",
    )
    # checking eng alphabet
    # evaluates best match for letter
    # per combination
    # - min iou with letter
    # - max iou with letter
    # - mean iou
    # acccumulate min, max, and mean for each letter separately

    best_mean_comb = (0, "a" * opts.ref_nshot)

    def prep_model(x):
        x.load_state_dict(torch.load(path_ckpt)["model"])
        x.cuda()
        x.eval()

    models = [ModelMain(opts) for _ in range(opts.num_processes)]
    map(prep_model, models)
    rng = cycle(models)

    with torch.no_grad():

        for test_idx, test_data in enumerate(test_loader):
            for key in test_data:
                test_data[key] = test_data[key].cuda()

            print("testing font %04d ..." % test_idx)
            # (min, max, mean)
            letters_tpl_ious = {}
            # eng alph len
            for i in range(opts.char_num):
                letters_tpl_ious[i] = []

            best_repr_count = np.zeros(opts.char_num)
            worst_repr_count = np.zeros(opts.char_num)

            comb_generator = combinations(range(opts.char_num), opts.ref_nshot)
            # options, test_idx, test_data dir_res, main_model, combination
            # converting to list won't scale in terms of memory efficiency for larger number of combinations
            # better use 'zip' with 'mupltiprocessing.Pool.imap'
            comb_and_model = list(
                zip(
                    repeat(opts),
                    repeat(test_idx),
                    repeat(test_data),
                    repeat(dir_res),
                    rng,
                    comb_generator,
                )
            )
            # to work with CUDA (instead of helping –– doesn't let the model to run)

            with mp.Pool(processes=opts.num_processes) as pool:
                # Map the process_combination function to the combinations and model instances in parallel
                for tpl_res in tqdm(
                    pool.starmap(process_combination, comb_and_model),
                    total=len(comb_and_model),
                ):
                    ind_min, ind_max = tpl_res[0].ind_min, tpl_res[0].ind_max
                    comb_mean = tpl_res[0].comb_mean
                    best_repr_count[ind_min] += 1
                    worst_repr_count[ind_max] += 1

                    comb = tpl_res[1]

                    for letter_ind in comb:
                        letters_tpl_ious[letter_ind].append(tpl_res[0])

                    if comb_mean > best_mean_comb[0]:
                        best_mean_comb = (comb_mean, comb)

            # save result over the font into a npy
            with open(f"stat_{test_idx}.pkl", "wb") as f:
                dump_state = (
                    letters_tpl_ious,
                    best_repr_count,
                    worst_repr_count,
                    charset,
                    opts,
                    test_idx,
                )
                pickle.dump(dump_state, f)

            # get the worst letters to recreate
            plot_worst_repr_letters(worst_repr_count, charset, opts, test_idx)
            # get the best letters to recreate
            plot_best_repr_letters(best_repr_count, charset, opts, test_idx)
            # get letters with the best mean
            plot_best_mean_letters(letters_tpl_ious, charset, opts, test_idx)


def process_combination(
    opts, test_idx, test_data, dir_res, model_main, comb
) -> tuple[ProcessingRes, Any]:
    new_opts = deepcopy(opts)
    new_opts.ref_char_ids = ",".join(str(el) for el in comb)

    dir_comb = "_".join(str(el) for el in comb)
    dir_save_name = f"{test_idx}__{dir_comb}"

    dir_save = os.path.join(dir_res, dir_save_name)
    if not os.path.exists(dir_save):
        os.mkdir(dir_save)
        os.mkdir(os.path.join(dir_save, "imgs"))
        os.mkdir(os.path.join(dir_save, "svgs_single"))
        os.mkdir(os.path.join(dir_save, "svgs_merge"))

    # idx_best_sample = np.zeros(opts.char_num)
    iou_max = calculate_iou_max(new_opts, model_main, test_data, dir_save)

    # idx_best_sample[i] = sample_idx
    ind_min = iou_max.argmin(axis=0)
    ind_max = iou_max.argmax(axis=0)
    comb_mean = iou_max.mean()
    ret = ProcessingRes(ind_min, ind_max, iou_max, comb_mean)
    return (ret, comb)


def calculate_iou_max(opts, model_main, test_data, dir_save):
    model_main.opts = opts

    iou_max = np.zeros(opts.char_num)
    # model_st = time()
    ret_dict_test, loss_dict_test = model_main(test_data, mode="test")
    # print("model process time:", time() - model_st)

    # svg_st = time()
    svg_sampled = ret_dict_test["svg"]["sampled_1"]
    sampled_svg_2 = ret_dict_test["svg"]["sampled_2"]

    img_trg = ret_dict_test["img"]["trg"]
    img_output = ret_dict_test["img"]["out"]

    for char_idx in range(opts.char_num):
        img_gt = (1.0 - img_trg[char_idx, ...]).data
        save_file_gt = os.path.join(dir_save, "imgs", f"{char_idx:02d}_gt.png")
        save_image(img_gt, save_file_gt, normalize=True)

        img_sample = (1.0 - img_output[char_idx, ...]).data
        save_file = os.path.join(
            dir_save, "imgs", f"{char_idx:02d}_{opts.img_size}.png"
        )
        save_image(img_sample, save_file, normalize=True)

    # write results w/o parallel refinement
    svg_dec_out = svg_sampled.clone().detach()
    for i, one_seq in enumerate(svg_dec_out):
        syn_svg_outfile = os.path.join(
            os.path.join(dir_save, "svgs_single"),
            f"syn_{i:02d}_wo_refine.svg",
        )

        syn_svg_f_ = open(syn_svg_outfile, "w")
        try:
            svg = render(one_seq.cpu().numpy())
            syn_svg_f_.write(svg)
            if i > 0 and i % 13 == 12:
                syn_svg_f_.write("<br>")

        except:
            continue
        syn_svg_f_.close()

    # write results w/ parallel refinement
    svg_dec_out = sampled_svg_2.clone().detach()
    for i, one_seq in enumerate(svg_dec_out):
        syn_svg_outfile = os.path.join(
            os.path.join(dir_save, "svgs_single"),
            f"syn_{i:02d}_refined.svg",
        )

        syn_svg_f = open(syn_svg_outfile, "w")
        try:
            svg = render(one_seq.cpu().numpy())
            syn_svg_f.write(svg)

        except:
            continue
        # calculate iou
        syn_svg_f.close()
        syn_img_outfile = syn_svg_outfile.replace(".svg", ".png")
        svg2img(syn_svg_outfile, syn_img_outfile, img_size=opts.img_size)
        iou_tmp, l1_tmp = cal_iou(
            syn_img_outfile,
            os.path.join(dir_save, "imgs", f"{i:02d}_{opts.img_size}.png"),
        )
        iou_tmp = iou_tmp
        if iou_tmp > iou_max[i]:
            iou_max[i] = iou_tmp
    # print("svg process time: ", time() - svg_st)
    return iou_max


def plot_cached(cached_dict_name, charset, opts):
    with open(cached_dict_name, "rb") as f:
        (
            letters_tpl_ious,
            best_repr_count,
            worst_repr_count,
            charset,
            opts,
            test_idx,
        ) = pickle.load(f)

        # get the worst letters to recreate
        plot_worst_repr_letters(worst_repr_count, charset, opts, test_idx)
        # get the best letters to recreate
        plot_best_repr_letters(best_repr_count, charset, opts, test_idx)
        # get letters with the best mean
        plot_best_mean_letters(letters_tpl_ious, charset, opts, test_idx)


def plot_worst_repr_letters(worst_counts, charset, opts, font_idx):
    alphabet_to_plot = {}
    for idx in range(len(charset)):
        alphabet_to_plot[charset[idx]] = worst_counts[idx]

    plt.bar(alphabet_to_plot.keys(), alphabet_to_plot.values())
    plt.savefig(f"best_repr_{opts.ref_nshot}_{font_idx}.jpg")


def plot_best_repr_letters(best_counts, charset, opts, font_idx):
    alphabet_to_plot = {}
    for idx in range(len(charset)):
        alphabet_to_plot[charset[idx]] = best_counts[idx]

    plt.bar(alphabet_to_plot.keys(), alphabet_to_plot.values())
    plt.savefig(f"best_repr_{opts.ref_nshot}_{font_idx}.jpg")


def plot_best_mean_letters(letters_tpls, charset, opts, font_idx):
    alphabet_to_plot = {}
    for key, arr_vals in letters_tpls:
        # tuples have format (min, max, mean)
        sum_mean = sum(t[2] for t in arr_vals)
        mean_over_mean = float(sum_mean) / len(arr_vals)
        alphabet_to_plot[charset[key]] = mean_over_mean

    plt.bar(alphabet_to_plot.keys(), alphabet_to_plot.values())
    plt.savefig(f"best_mean_{opts.ref_nshot}_{font_idx}.jpg")


def main():

    opts = get_parser_main_model().parse_args()
    opts.name_exp = opts.name_exp + "_" + opts.model_name
    experiment_dir = os.path.join("./experiments", opts.name_exp)
    print(f"Testing on experiment {opts.name_exp}...")
    # Dump options
    stat_main_model(opts)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
