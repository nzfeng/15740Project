# 15740Project

We focus on two operations: convolution and matrix-vector multiplication.

## The General Architecture
All the components below are independent to each other and can work in parallel:
- **DSP:** DSP is the function unit that performs floating point computation. The chip has `D` DSPs which can work in parallel. Each floating point computation takes `C` cycles.
- **Memory:** Memory is separated into on-chip and off-chip memory. Data must be read onto the chip before it can be used by DSPs. Reading `k` bytes from off-chip to on-chip needs `A + B * k` cycles, where `A` and `B` are constants. The size of the on-chip memory is `M` bytes. The off-chip memory is infinitely large. **On-chip memory is not initialized to zero!**
- **Arrays:** To use the on-chip memory, construct on-chip arrays. DSPs can directly access on-chip arrays taking no cycle (it's already included in the `C` cycles).
- **Register:** You can also use as many registers as you want without spending on-chip memory.
- **Ports:** There is only one read-write port between on-chip and off-chip. For simplicity, assume that there are an infinite number of read-write ports between the on-chip memory and DSPs (in practice we need to partition the array into different banks, each of which has its own read-write port).


## Chip Operations
- `chip = Chip(...)`: Construct a chip
- `onchip_array = chip.array(num, name)`: Allocate an on-chip float array of `num` items. Use `onchip_array[i]` to access the items in the array.
- `r = chip.register(init_value)`: Construct a register with an initial value. Use `r[0]` to access the value of the register.
- `chip.read(onchip_array, onchip_offset, offchip_array, offchip_offset, num)`: Copy a chunk of float data from off-chip to on-chip. Takes `A + B * (4 * num)` cycles.
- `chip.write(onchip_array, onchip_offset, offchip_array, offchip_offset, num)`: Copy a chunk of float data from on-chip to off-chip. Takes `A + B * (4 * num)` cycles.
- `chip.compute(dst, value)`: Use a DSP to compute the value and copy the value to a register.
- `chip.compute_array(dst_array, dst_offset, value)`: Use a DSP to compute the value and copy the value to an on-chip array.

## Modules
A module consists of an array of operations which run *sequentially*. Multiple modules run *in parallel* until they meet the `chip.join()` function.
- `chip.module(use_port, array_list, name)`: Construct a module. The module can access the port between on-chip and off-chip if `use_port = True`. Only one module can access the port at any given time. `array_list` is the list of on-chip arrays the module can access. Any on-chip array can only be accessed by one module at any given time.
- `chip.join()`: Join all active modules and update the number of cycles. This method separates the code into blocks. Modules in the same block run in parallel. Modules in different blocks never run at the same time. *Remember to call this method when all operations are done.*

The best approach is to **put all operations into modules**. If an operation does not belong to any module, an auto module will be created for this single operation.