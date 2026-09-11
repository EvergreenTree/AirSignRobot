import pytest
import torch
from ebim_phase2.model import ACT, ACTConfig, loss_fn


def test_masked_posterior_ignores_padded_actions_and_inference_needs_no_targets():
    torch.set_num_threads(2)
    model = ACT(ACTConfig(42, 17, image_size=32, chunk_size=4, hidden_dim=32, heads=4,
                          encoder_layers=1, decoder_layers=1, dropout=0), pretrained=False).eval()
    image = torch.zeros(2,3,3,32,32,dtype=torch.uint8)
    state = torch.zeros(2,42)
    action = torch.randn(2,4,17)
    mask = torch.tensor([[True,True,False,False],[True,True,True,True]])
    altered = action.clone(); altered[0,2:] = 10000
    torch.manual_seed(1); pred, mu, lv = model(image,state,action,mask)
    torch.manual_seed(1); pred2, mu2, lv2 = model(image,state,altered,mask)
    torch.testing.assert_close(mu,mu2)
    torch.testing.assert_close(pred,pred2)
    loss, _, _ = loss_fn(pred,action,mask,mu,lv)
    assert torch.isfinite(loss)
    assert model(image,state)[0].shape == (2,4,17)
def test_joint_residual_anchor_preserves_native_measured_pose_at_initialization():
    import numpy as np
    from ebim_phase2.schema import JOINT_STATE_INDICES, JOINT_ACTION_INDICES
    config = ACTConfig(42, 17, image_size=32, hidden_dim=32, heads=4,
                       encoder_layers=1, decoder_layers=1, chunk_size=4, joint_residual=True)
    model = ACT(config, pretrained=False).eval()
    stats = {'state_mean': np.linspace(-1, 1, 42), 'state_std': np.linspace(.1, 2, 42),
             'action_mean': np.linspace(2, 3, 17), 'action_std': np.linspace(.2, 1, 17)}
    model.configure_joint_anchor(stats)
    state = torch.linspace(-2, 2, 42)[None]
    with torch.inference_mode():
        prediction, _, _ = model(torch.zeros(1, 3, 3, 32, 32), state)
    native = prediction[0].numpy()*stats['action_std'] + stats['action_mean']
    measured = state[0].numpy()*stats['state_std'] + stats['state_mean']
    np.testing.assert_allclose(native[:, JOINT_ACTION_INDICES],
                               np.tile(measured[JOINT_STATE_INDICES], (4, 1)), atol=1e-6)
    np.testing.assert_allclose(prediction[0, :, [7, 15, 16]], 0, atol=0)
