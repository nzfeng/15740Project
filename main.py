import numpy as np
import argparse

from simulator import Chip


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dsp_num', '-D', type=int,
                      help='Number of DSPs')
  parser.add_argument('--dsp_cycle', '-C', type=int,
                      help='Cycles a DSP takes for one flop')
  parser.add_argument('--index_cycle', '-X', type=int,
                      help='Cycles for array indexing')
  parser.add_argument('--mem_size', '-M', type=int,
                      help='Size of on-chip memory (in bytes)')
  parser.add_argument('--mem_overhead', '-A', type=int,
                      help='Overhead of read-write port')
  parser.add_argument('--mem_unit', '-B', type=float,
                      help='Cycles for reading 1 byte from the port')

  args = parser.parse_args()

  chip = Chip(args.dsp_num, args.dsp_cycle, args.index_cycle,
              args.mem_size, args.mem_overhead, args.mem_unit)

  # SPMV
  num_rows = 10
  num_nonzero = 100
  n = 128

  row_ptr = chip.array(num_rows + 1, 'row_ptr')
  column_index = chip.array(num_nonzero, 'column_index')
  values = chip.array(num_nonzero, 'values')
  x = chip.array(n, 'x')
  y = chip.array(num_rows, 'y')

  # Random input data
  offchip_row_ptr = np.linspace(0, 100, 11)
  offchip_column_index = np.linspace(0, 99, 100)
  offchip_values = np.random.randn(100)
  offchip_x = np.random.randn(n)

  # Read inputs onto chip
  with chip.module(True, [row_ptr, column_index, values, x], 'read_inputs'):
    chip.read(row_ptr, 0, offchip_row_ptr, 0, 11)
    chip.read(column_index, 0, offchip_column_index, 0, 100)
    chip.read(values, 0, offchip_values, 0, 100)
    chip.read(x, 0, offchip_x, 0, n)

  chip.join()

  # Computation
  with chip.module(False, [row_ptr, column_index, values, x, y], 'compute'):
    for i in range(num_rows):
      y0 = 0
      up = chip.get_item(row_ptr, i + 1)
      down = chip.get_item(row_ptr, i)
      for j in range(int(down), int(up)):
        vj = chip.get_item(values, j)
        idx = chip.get_item(column_index, j)
        xi = chip.get_item(x, int(idx))
        y1 = chip.compute(vj * xi)
        y0 = chip.compute(y0 + y1)
      chip.array_write(y, i, y0)

  chip.join()

  # Write the result
  offchip_y = np.zeros((num_rows,), dtype=float)
  with chip.module(True, [y], 'write_out'):
    chip.write(y, 0, offchip_y, 0, num_rows)

  chip.join()  # Call at the end to update cycles

  # Check result
  correct_y = np.zeros((num_rows,), dtype=float)
  for i in range(num_rows):
    y0 = 0
    for j in range(int(offchip_row_ptr[i]), int(offchip_row_ptr[i + 1])):
      y0 += offchip_values[j] * offchip_x[int(offchip_column_index[j])]
    correct_y[i] = y0

  for i in range(num_rows):
    if offchip_y[i] != correct_y[i]:
      raise RuntimeError('Result error in index {}'.format(i))

  print('Pass')


if __name__ == '__main__':
  main()