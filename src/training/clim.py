# TODO: process mosaicked image
import torch
import torch.nn.functional as F


class CLIM:
    mosaic_choices = [2, 3, 4]

    def __init__(self):
        super().__init__()

    def __call__(self, batch, model, dist_model, loss, device, cast_dtype,
                 distributed, args):
        if distributed:
            model = model.module
        images, texts = batch
        images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        mosaicked_images, pseudo_boxes_list, single_images \
            = self.split_a_batch(images, args.train_image_size)
        single_image_features = model.encode_image(single_images, normalize=True)
        with torch.no_grad():
            text_features = model.encode_text(texts, normalize=True)
        logit_scale = model.logit_scale.exp()

        pseudo_region_features = model.encode_pseudo_boxes(
            mosaicked_images, pseudo_boxes_list, normalize=True, extract_type=args.extract_type)
        image_features = torch.cat([pseudo_region_features, single_image_features], dim=0)

        contrast_loss = loss(image_features,
                             text_features,
                             logit_scale,
                             output_dict=False, )

        losses = dict(loss_contrast=contrast_loss * args.contrast_weight)

        return losses, len(images), logit_scale


    @staticmethod
    def _generate_normed_boxes(M, N):
        grid_x, grid_y = torch.meshgrid(torch.linspace(0, 1, N + 1), torch.linspace(0, 1, M + 1),
                                        indexing='xy')
        x0y0s = torch.stack([grid_x[:M, :N], grid_y[:M, :N]], dim=-1)
        x1y1s = torch.stack([grid_x[1:, 1:], grid_y[1:, 1:]], dim=-1)
        pseudo_boxes = torch.cat([x0y0s, x1y1s],
                                 dim=-1).view(-1, 4)
        return pseudo_boxes

    def split_a_batch(self, images, train_image_size):
        batch_size = images.shape[0]
        choices = self.mosaic_choices
        min_images = sum([c**2 for c in choices])

        assert batch_size >= min_images
        num_single = batch_size % min_images
        num_groups = batch_size // min_images
        # assert num_single == 0
        split = [c for c in choices for _ in range(num_groups)]
        # split = [2] * num_groups + [3] * num_groups + [4] * num_groups
        pseudo_boxes_list = [self._generate_normed_boxes(s, s).to(images) for s in split]

        images_list = torch.split(images, [s**2 for s in split] + [num_single], dim=0)

        mosaicked_images_list = [
            F.interpolate(self._mosaic_a_minibatch(imgs, s, s), size=train_image_size, mode='bicubic')
            for imgs, s in zip(images_list[:-1], split)]

        mosaicked_images = torch.cat(mosaicked_images_list)

        return mosaicked_images, pseudo_boxes_list, images_list[-1]

    @staticmethod
    def _mosaic_a_minibatch(images, M, N):
        bs, _, h, w = images.shape
        assert bs % (M * N) == 0
        num_mosaic = bs // (M*N)
        images_for_mosaic = images.permute(0, 2, 3, 1)
        images_for_mosaic = images_for_mosaic.view(num_mosaic, M, N, h, w, 3)
        images_for_mosaic = images_for_mosaic.permute(0, 1, 3, 2, 4, 5).contiguous()
        mosaicked_images = images_for_mosaic.view(num_mosaic, M * h, N * w, 3)
        mosaicked_images = mosaicked_images.permute(0, 3, 1, 2)

        return mosaicked_images
