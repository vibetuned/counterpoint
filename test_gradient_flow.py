"""
Test script to verify gradient flow through the MaskedMultiCategoricalMixin.

This checks if the learning signal (gradients) properly propagate through
the action masking operations.
"""

import torch
import torch.nn as nn
import numpy as np

# Add path if needed
import sys
sys.path.insert(0, '/home/flux/projects/counterpoint-steps/src')

from counterpoint.policies.mixins import MaskedMultiCategoricalMixin, MASK_VALUE


def test_gradient_flow():
    """Test that gradients flow through the masking operations."""
    print("=" * 60)
    print("Testing Gradient Flow Through MaskedMultiCategoricalMixin")
    print("=" * 60)
    
    batch_size = 4
    device = torch.device("cpu")
    
    # Create a simple test class that uses the mixin
    class TestMixin(MaskedMultiCategoricalMixin):
        def __init__(self):
            super().__init__()
    
    mixin = TestMixin()
    
    # Create input logits with requires_grad=True
    logits = torch.randn(batch_size, 20, requires_grad=True)
    
    # Create action mask: [fingers_black(5), num_notes(1), finger_mask(5)] = 11
    # fingers_black: all 0 (white keys)
    # num_notes: 1
    # finger_mask: all 1 (allow all fingers)
    action_mask = torch.zeros(batch_size, 11)
    action_mask[:, 5] = 1.0  # num_notes
    action_mask[:, 6:11] = 1.0  # all fingers allowed
    
    # Create finger_mask from priority head (simulating Gumbel-Softmax output)
    # Use soft values to test gradient flow
    finger_mask = torch.zeros(batch_size, 5, requires_grad=True)
    finger_mask_data = torch.zeros(batch_size, 5)
    finger_mask_data[:, 0] = 1.0  # Select first finger
    finger_mask = finger_mask_data.clone().detach().requires_grad_(True)
    
    print("\n1. Testing basic gradient flow through logits...")
    print(f"   Input logits shape: {logits.shape}")
    print(f"   Action mask shape: {action_mask.shape}")
    print(f"   Finger mask shape: {finger_mask.shape}")
    
    # Apply action mask
    masked_logits = mixin._apply_action_mask(logits, action_mask, finger_mask)
    
    # Check if output requires grad
    print(f"\n2. After masking:")
    print(f"   masked_logits.requires_grad: {masked_logits.requires_grad}")
    
    # Create a dummy loss
    target = torch.randint(0, 2, (batch_size, 10))  # Random binary targets
    
    # Simple loss: cross-entropy on each action branch
    loss = 0
    for i in range(10):
        branch_logits = masked_logits[:, 2*i:2*i+2]
        branch_target = target[:, i]
        loss += nn.functional.cross_entropy(branch_logits, branch_target)
    loss = loss / 10
    
    print(f"\n3. Loss value: {loss.item():.4f}")
    
    # Backpropagate
    loss.backward()
    
    # Check gradients
    print("\n4. Gradient check:")
    print(f"   logits.grad exists: {logits.grad is not None}")
    if logits.grad is not None:
        print(f"   logits.grad.shape: {logits.grad.shape}")
        print(f"   logits.grad non-zero: {(logits.grad != 0).sum().item()} / {logits.grad.numel()}")
        print(f"   logits.grad mean abs: {logits.grad.abs().mean().item():.6f}")
        print(f"   logits.grad max abs: {logits.grad.abs().max().item():.6f}")
    
    print(f"\n   finger_mask.grad exists: {finger_mask.grad is not None}")
    if finger_mask.grad is not None:
        print(f"   finger_mask.grad: {finger_mask.grad}")
    
    # Test with soft finger mask (Gumbel-Softmax style)
    print("\n" + "=" * 60)
    print("5. Testing with soft (differentiable) finger mask...")
    print("=" * 60)
    
    logits2 = torch.randn(batch_size, 20, requires_grad=True)
    
    # Simulate Gumbel-Softmax with soft values
    raw_logits = torch.randn(batch_size, 5, requires_grad=True)
    soft_finger_mask = torch.softmax(raw_logits, dim=-1)
    
    # Convert to hard mask while maintaining gradient (straight-through estimator)
    hard_finger_mask = torch.zeros_like(soft_finger_mask)
    indices = soft_finger_mask.argmax(dim=-1, keepdim=True)
    hard_finger_mask.scatter_(1, indices, 1.0)
    # Straight-through: use hard values in forward, but soft gradients in backward
    st_finger_mask = (hard_finger_mask - soft_finger_mask).detach() + soft_finger_mask
    
    masked_logits2 = mixin._apply_action_mask(logits2, action_mask, st_finger_mask)
    
    loss2 = 0
    for i in range(10):
        branch_logits = masked_logits2[:, 2*i:2*i+2]
        branch_target = target[:, i]
        loss2 += nn.functional.cross_entropy(branch_logits, branch_target)
    loss2 = loss2 / 10
    
    loss2.backward()
    
    print(f"\n   raw_logits.grad exists: {raw_logits.grad is not None}")
    if raw_logits.grad is not None:
        print(f"   raw_logits.grad non-zero: {(raw_logits.grad != 0).sum().item()} / {raw_logits.grad.numel()}")
        print(f"   raw_logits.grad mean abs: {raw_logits.grad.abs().mean().item():.6f}")
    
    print(f"\n   logits2.grad exists: {logits2.grad is not None}")
    if logits2.grad is not None:
        print(f"   logits2.grad non-zero: {(logits2.grad != 0).sum().item()} / {logits2.grad.numel()}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    has_issues = False
    
    if logits.grad is None or (logits.grad == 0).all():
        print("❌ PROBLEM: Gradients are NOT flowing to logits!")
        has_issues = True
    else:
        print("✅ Gradients ARE flowing through logits")
    
    if raw_logits.grad is None or (raw_logits.grad == 0).all():
        print("❌ PROBLEM: Gradients are NOT flowing to raw_logits (priority head)!")
        has_issues = True
    else:
        print("✅ Gradients ARE flowing through priority head (raw_logits)")
    
    if not has_issues:
        print("\n✅ Learning signal should propagate correctly!")
    else:
        print("\n❌ There are gradient flow issues that need to be fixed.")
    
    return not has_issues


def test_torch_where_gradient():
    """Specifically test torch.where gradient behavior."""
    print("\n" + "=" * 60)
    print("Testing torch.where Gradient Behavior")
    print("=" * 60)
    
    # Test 1: torch.where with constant tensors (current implementation)
    print("\nTest A: torch.where with constant tensors (problematic)")
    x = torch.randn(4, requires_grad=True)
    condition = torch.tensor([True, False, True, False])
    
    # This is what we do in mixins.py
    CONST = -1e8
    result = torch.where(condition, torch.tensor(CONST), torch.tensor(0.0))
    
    # Can we backprop through this?
    try:
        loss = result.sum()
        loss.backward()
        print(f"   x.grad: {x.grad}")
        print("   ⚠️ Note: torch.where with constants gives no gradients to x (expected)")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 2: How it affects total gradient when added to x
    print("\nTest B: x + mask (where mask computed with torch.where)")
    x2 = torch.randn(4, requires_grad=True)
    condition2 = torch.tensor([True, False, True, False])
    mask = torch.where(condition2, torch.tensor(-1e8), torch.tensor(0.0))
    result2 = x2 + mask  # Adding mask to x
    
    loss2 = result2.sum()
    loss2.backward()
    print(f"   x2.grad: {x2.grad}")
    print(f"   x2.grad all ones? {(x2.grad == 1.0).all()}")
    if (x2.grad == 1.0).all():
        print("   ✅ Gradients flow through x even when mask is constant!")
    
    # Test 3: What about when mask depends on x?
    print("\nTest C: Checking if env_finger_mask affects gradients properly")
    x3 = torch.randn(4, 5, requires_grad=True)
    env_mask = torch.tensor([[1.0, 1.0, 0.0, 1.0, 1.0]] * 4)  # One finger forbidden
    
    # This is similar to our effective_selected computation
    selected = torch.softmax(x3, dim=-1)
    effective = selected * env_mask  # Zero out forbidden finger
    
    loss3 = effective.sum()
    loss3.backward()
    
    print(f"   x3.grad exists: {x3.grad is not None}")
    print(f"   x3.grad non-zero count: {(x3.grad != 0).sum().item()}")
    print(f"   ✅ Multiplication by env_mask preserves gradients")


if __name__ == "__main__":
    test_gradient_flow()
    test_torch_where_gradient()
