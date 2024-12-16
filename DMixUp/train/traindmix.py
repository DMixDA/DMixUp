import time
import torch
import torch.nn as nn
import src.utils as utils


def train_dmix(args, loaders, optimizers, models_ld, sp_params, losses, epoch):
    print("Epoch: [{}/{}]".format(epoch, args.epochs))
    start = time.time()
    src_train_loader, tgt_train_loader = loaders[0], loaders[1]
    optimizer_ld, optimizer_ld = optimizers[0], optimizers[1]
    sp_param_ld, sp_param_td = sp_params[0], sp_params[1]
    ce, mse = losses[0], losses[1]

    for step, (src_data, tgt_data) in enumerate(zip(src_train_loader, tgt_train_loader)):
        src_imgs, src_labels = src_data
        tgt_imgs, tgt_labels = tgt_data
        src_imgs, src_labels = src_imgs.cuda(non_blocking=True), src_labels.cuda(non_blocking=True)
        tgt_imgs, tgt_labels = tgt_imgs.cuda(non_blocking=True), tgt_labels.cuda(non_blocking=True)

        x_ld= models_ld(tgt_imgs)

        pseudo_ld, top_prob_ld, threshold_ld = utils.get_target_preds(args, x_ld)
        dmix_ld_loss = utils.get_dmix_loss(models_ld, src_imgs, tgt_imgs, src_labels, pseudo_ld, args.lam_dmix)


        total_loss = dmix_ld_loss

        if step == 0:
            print('Dynamic MixUp Loss: {:.4f}'.format(dmix_ld_loss.item()))

        if epoch > args.cdkd_start_start:
            cdkd_start_ld = torch.ge(top_prob_ld, threshold_ld)
            cdkd_start_ld = torch.nonzero(cdkd_start_ld).squeeze()


            if cdkd_start_ld.dim() > 0 and cdkd_start_ld.dim() > 0:
                if cdkd_start_ld.numel() > 0 and cdkd_start_ld.numel() > 0:
                    cdkd = min(cdkd_start_ld.size(0), cdkd_start_ld.size(0))
                    dmix_ld_loss = ce(x_ld[cdkd_start_ld[:cdkd]], pseudo_ld[cdkd_start_ld[:cdkd]].cuda().detach())

                    total_loss += dmix_ld_loss

                    if step == 0:
                        print('confidence-based crossdomin knowledge-diffusion Loss: {:.4f}'.format(cdkd_start_ld.item()))

        if epoch <= args.idkd_start_start:
            idkd_start_ld = torch.lt(top_prob_ld, threshold_ld)
            idkd_start_ld = torch.nonzero(idkd_start_ld).squeeze()

            if idkd_start_ld.dim() > 0 and idkd_start_ld.dim() > 0:
                if idkd_start_ld.numel() > 0 and idkd_start_ld.numel() > 0:
                    idkd = min(idkd_start_ld.size(0), idkd_start_ld.size(0))
                    idkd_ld_loss = utils.get_sp_loss(x_ld[idkd_start_ld[:idkd]], pseudo_ld[idkd_start_ld[:idkd]], sp_param_ld)

                    total_loss += idkd_ld_loss

                    if step == 0:
                        print('confidence-based intradomin knowledge-diffusion: {:.4f}', idkd_ld_loss.item())


        optimizer_ld.zero_grad()
        total_loss.backward()


    print("Train time: {:.2f}".format(time.time() - start))
