import torch

from src.encoder import Decoder, Encoder


def test_encoder_output_shape():
    enc = Encoder(embed_dim=1024)
    x = torch.zeros(4, 3, 64, 64)
    out = enc(x)
    assert out.shape == (4, 1024)


def test_decoder_output_shape():
    dec = Decoder(feature_dim=230)
    feat = torch.zeros(4, 230)
    out = dec(feat)
    assert out.shape == (4, 3, 64, 64)


def test_encoder_decoder_pipeline_runs():
    enc = Encoder(embed_dim=1024)
    dec = Decoder(feature_dim=230)
    x = torch.zeros(2, 3, 64, 64)
    e = enc(x)
    assert e.shape == (2, 1024)
    fake_state = torch.zeros(2, 230)
    y = dec(fake_state)
    assert y.shape == x.shape
