# 15740Project

We focus on two operations: convolution and matrix-vector multiplication.

## The General Architecture
All the components below are independent to each other and can work in parallel:
- **DSP:** DSP is the function unit that performs floating point computation. The chip has `D` DSPs which can work in parallel. Each floating point computation takes `C` cycles.
- **Memory:** Memory is separated into on-chip and off-chip memory. Data must be read onto the chip before it can be used by DSPs. Reading `k` bytes from off-chip to on-chip needs `A + B * k` cycles, where `A` and `B` are constants. The size of the on-chip memory is `M` bytes. The off-chip memory is infinitely large. **On-chip memory is not initialized to zero!**
- **Arrays:** To use the on-chip memory, construct on-chip arrays. DSPs can directly access on-chip arrays taking no cycle (it's already included in the `C` cycles), but *array indexing also takes `X` cycles*.
- **Ports:** There is only one read-write port between on-chip and off-chip. Each array only has one read-write port, so only one module can access it at a time.


## Chip Operations
- `chip = Chip(...)`: Constructs a chip
- `onchip_array = chip.array(num, name)`: Allocates an on-chip float array of `num` items. Use `get_item()` to access the items in the array.
- `chip.read(onchip_array, onchip_offset, offchip_array, offchip_offset, num)`: Copies a chunk of float data from off-chip to on-chip. Takes `A + B * (4 * num)` cycles.
- `chip.write(onchip_array, onchip_offset, offchip_array, offchip_offset, num)`: Copies a chunk of float data from on-chip to off-chip. Takes `A + B * (4 * num)` cycles.
- `chip.get_item(onchip_array, index)`: Returns `onchip_array[index]`. Takes `X` cycles for indexing.
- `chip.compute(value)`: Returns the same value, but uses one DSP to compute and takes `C` cycles.
- `chip.array_write(dst_array, dst_offset, value)`: Writes the value back to an on-chip array. Takes `X` cycles for indexing.

## Modules
A module consists of an array of operations which run *sequentially*. Multiple modules run *in parallel* until they meet the `chip.join()` function.
- `module = chip.module(func, name)`: Construct a module that runs the function `func`. Call this module with `ans = module(...)`.
- `chip.join()`: Join all active modules and update the number of cycles. This method separates the code into blocks. Modules in the same block run in parallel. Modules in different blocks never run at the same time. In the same block, only one module can access the port, and each array can only be accessed by one array.

The best approach is to **put all operations into modules**. If an operation does not belong to any module, an auto module will be created for this single operation.

## Example
`main.py` contains an example of SPMV. Run this example with the following command:
```
python main.py -A 1 -B 0.1 -C 1 -X 1 -D 1 -M 100000
```
