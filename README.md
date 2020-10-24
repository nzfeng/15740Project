# 15740Project

We focus on two operations: convolution and matrix-vector multiplication.

## The General Architecture
All the components below are independent to each other and can work in parallel:
- **DSP:** DSP is the function unit that performs floating point computation. The chip has `D` DSPs which can work in parallel. Each floating point computation takes `C` cycles.
- **Memory:** Memory is separated into on-chip and off-chip memory. Data must be read onto the chip before it can be used by DSPs. Reading `k` bytes from off-chip to on-chip needs `P + B * k` cycles, where `P` and `B` are constants. The size of the on-chip memory is `M` bytes. The off-chip memory is infinitely large.