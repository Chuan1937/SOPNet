from __future__ import annotations

import torch

from sopnet.models import SOPNet, SOPNetCls, SOPNetConfig, count_parameters


def test_sopnet_output_shape_and_range():
    model = SOPNet(SOPNetConfig())
    model.eval()
    x = torch.randn(4, 1, 400)
    with torch.no_grad():
        field = model(x)
    assert field.shape == (4, 1, 400)
    assert field.min() >= -1.0 and field.max() <= 1.0
    assert torch.isfinite(field).all()


def test_sopnet_parameter_budget():
    model = SOPNet(SOPNetConfig())
    parameters = count_parameters(model)
    assert parameters < 2_000_000, f"too many parameters: {parameters}"


def test_sopnet_predict_reads_field():
    model = SOPNet(SOPNetConfig())
    model.eval()
    x = torch.randn(3, 1, 400)
    with torch.no_grad():
        output = model.predict(x)
    assert output["p_position"].shape == (3,)
    assert output["polarity"].shape == (3,)
    assert output["confidence"].shape == (3,)
    assert ((output["polarity"] >= -1) & (output["polarity"] <= 1)).all()


def test_classifier_shares_encoder_interface():
    config = SOPNetConfig()
    classifier = SOPNetCls(config)
    classifier.eval()
    x = torch.randn(2, 1, 400)
    with torch.no_grad():
        logits = classifier(x)
    assert logits.shape == (2, 3)

    field_model = SOPNet(config)
    classifier_params = dict(classifier.encoder.named_parameters())
    field_params = dict(field_model.encoder.named_parameters())
    assert classifier_params.keys() == field_params.keys()
    for key in classifier_params:
        assert classifier_params[key].shape == field_params[key].shape


def test_model_handles_short_and_long_windows():
    config = SOPNetConfig()
    model = SOPNet(config)
    model.eval()
    for length in (200, 400, 800):
        with torch.no_grad():
            field = model(torch.randn(1, 1, length))
        assert field.shape[-1] == length
