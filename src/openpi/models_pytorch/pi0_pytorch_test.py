from types import SimpleNamespace

import torch

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch


class FakePI0Pytorch:
    _embed_image_batch = PI0Pytorch._embed_image_batch
    _log_packed_image_fallback = PI0Pytorch._log_packed_image_fallback
    _can_pack_image_streams = PI0Pytorch._can_pack_image_streams
    _get_zero_image_embedding_template = PI0Pytorch._get_zero_image_embedding_template
    _embed_images_iterative = PI0Pytorch._embed_images_iterative
    _embed_images_packed = PI0Pytorch._embed_images_packed

    def __init__(self):
        projector_linear = SimpleNamespace(out_features=4, weight=torch.zeros(4, 4, dtype=torch.float32))
        vision_embeddings = SimpleNamespace(num_patches=2)
        vision_model = SimpleNamespace(embeddings=vision_embeddings)
        vision_tower = SimpleNamespace(vision_model=vision_model)
        model = SimpleNamespace(
            vision_tower=vision_tower,
            multi_modal_projector=SimpleNamespace(linear=projector_linear),
        )

        self.paligemma_with_expert = SimpleNamespace(
            paligemma=SimpleNamespace(model=model),
            embed_image=self._fake_embed_image,
        )
        self._packed_image_fallback_logged = False
        self.fallback_reasons = []
        self.embed_image_calls = 0

    def _apply_checkpoint(self, func, *args, **kwargs):
        return func(*args, **kwargs)

    def _fake_embed_image(self, image: torch.Tensor) -> torch.Tensor:
        self.embed_image_calls += 1
        flattened = image.reshape(image.shape[0], -1).to(torch.float32)
        first_token = flattened[:, :4]
        second_token = flattened[:, 4:8]
        return torch.stack((first_token, second_token), dim=1)

    def _log_packed_image_fallback(self, reason: str) -> None:
        self.fallback_reasons.append(reason)
        PI0Pytorch._log_packed_image_fallback(self, reason)


def _make_streams():
    base = torch.arange(3 * 1 * 2 * 4, dtype=torch.float32).reshape(3, 1, 2, 4)
    images = [base + offset for offset in (0, 100, 200)]
    masks = [
        torch.tensor([True, False, True]),
        torch.tensor([False, True, True]),
        torch.tensor([True, True, False]),
    ]
    return images, masks


def test_embed_images_packed_matches_iterative_on_active_rows():
    model = FakePI0Pytorch()
    images, masks = _make_streams()

    iterative_embs, iterative_pad_masks, iterative_att_masks = model._embed_images_iterative(images, masks)
    packed_embs, packed_pad_masks, packed_att_masks = model._embed_images_packed(images, masks)

    assert packed_att_masks == iterative_att_masks
    assert len(packed_embs) == len(iterative_embs)

    for iterative_emb, packed_emb, iterative_pad_mask, packed_pad_mask, mask in zip(
        iterative_embs,
        packed_embs,
        iterative_pad_masks,
        packed_pad_masks,
        masks,
        strict=True,
    ):
        assert torch.equal(iterative_pad_mask, packed_pad_mask)
        assert torch.equal(iterative_emb[mask], packed_emb[mask])
        assert torch.count_nonzero(packed_emb[~mask]) == 0


def test_embed_images_packed_handles_zero_active_images_without_encoder_call():
    model = FakePI0Pytorch()
    images, _ = _make_streams()
    masks = [torch.zeros(image.shape[0], dtype=torch.bool) for image in images]

    packed_embs, packed_pad_masks, packed_att_masks = model._embed_images_packed(images, masks)

    assert model.embed_image_calls == 0
    assert packed_att_masks == [0] * (len(images) * 2)
    for packed_emb, packed_pad_mask in zip(packed_embs, packed_pad_masks, strict=True):
        assert torch.count_nonzero(packed_emb) == 0
        assert not packed_pad_mask.any()


def test_embed_images_packed_falls_back_for_heterogeneous_streams():
    model = FakePI0Pytorch()
    images, masks = _make_streams()
    images[1] = images[1].to(torch.bfloat16)

    iterative = model._embed_images_iterative(images, masks)
    packed = model._embed_images_packed(images, masks)

    assert model.fallback_reasons
    iterative_embs, iterative_pad_masks, iterative_att_masks = iterative
    packed_embs, packed_pad_masks, packed_att_masks = packed
    for iterative_emb, packed_emb in zip(iterative_embs, packed_embs, strict=True):
        assert torch.equal(iterative_emb, packed_emb)
    for iterative_pad_mask, packed_pad_mask in zip(iterative_pad_masks, packed_pad_masks, strict=True):
        assert torch.equal(iterative_pad_mask, packed_pad_mask)
    assert iterative_att_masks == packed_att_masks
