# Simulator for EIE, an accelerator which performs sparse matrix-vector
# multiplication.

import numpy as np
from scipy.sparse import random
import argparse
import math

from simulator import Chip


def isCorrect(W, a, b):
  '''
  Use naive multiplication to confirm if the results of EIE are indeed
  correct, up to floating-point error.

  Args:
      W = weight matrix
      a = input vector
      b = output vector computed by EIE

  Returns:
      True if multiplication checks out, False otherwise
  '''
  epsilon = 1e-5
  [rows, cols] = np.shape(W)
  valid = np.matmul(W, a)
  for i in range(rows):
    if abs(valid[i] - b[i]) > epsilon:
      return False
  return True


def matrixToCSC(M: np.ndarray):
  '''
  Convert a matrix M into the compressed sparse column (CSC) format described
  in the paper. The paper uses 4 bits as the size of each element, but in our
  simulator we use floats throughout, so we won't worry about extra 0's in
  our encoding.

  Args:
    Matrix to be compressed. Could also be just a single column vector.

  Returns:
    v = array of nonzero elements
    z = array representing relative row index, i.e. # of zeros before the
      corresponding element in v (in that column)
    p = array of column pointers, i.e. index in v where each new column
      begins. Last element points one beyond the last element.
  '''
  (rows, cols) = np.shape(M)
  v = []
  z = []  # changed this to be row index instead
  p = []
  for j in range(cols):
    num_zeros = 0
    flag = False  # if first nonzero element in a column has been located
    for i in range(rows):
      elem = M[i, j]
      if elem != 0:
        if not flag: p.append(len(v))
        v.append(elem)
        z.append(i)
        # z.append(num_zeros)
        num_zeros = 0
        flag = True
      else:
        num_zeros += 1
    if not flag: p.append(len(v))

  p.append(len(v))
  # sanity checking
  if len(v) != len(z): print("len(v) != len(z)")
  if len(p) != cols + 1: print("len(p) incorrect")

  # Convert to numpy arrays.
  v = np.asarray(v)
  z = np.asarray(z)
  p = np.asarray(p)

  return [v, z, p]


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
  parser.add_argument('--matrix_size', '-n', type=int,
                      help='Maximum dimension of matrix/vector')
  parser.add_argument('--sparsity', '-s', type=float,
                      help='Density of the weight matrix W and initial input vector')
  parser.add_argument('--PE_num', '-P', type=int, help='Number of PEs')

  args = parser.parse_args()
  n_row = args.matrix_size
  n_col = args.matrix_size
  sparsity = args.sparsity
  num_PE = args.PE_num

  chip = Chip(args.dsp_num, args.dsp_cycle, args.index_cycle,
              args.mem_size, args.mem_overhead, args.mem_unit)

  # Hyper-parameters
  max_vsize_per_PE = (1 << 10)  # Size of v and z in each PE (num of items)
  max_asize_per_PE = (1 << 10)  # Size of a (there are 2 a in each PE)
  max_psize_per_PE = (1 << 10)  # Size of p

  # Create random sparse matrix W, and sparse vector a.
  # Portions of W are put into CSC format by CPU.
  W = random(n_row, n_col, density=sparsity, dtype=float)
  W = np.array(W.A)
  a = random(n_col, 1, density=sparsity, dtype=float)
  a = np.array(a.A).flatten()

  # A list of v, z, p vectors (CSC representations) for each portion of W in
  # each PE.
  VZP = []  # = [[v1, z1, p1], [v2, z2, p2],..., [vN, zN, pN]]
  num_rows = math.ceil(n_row / num_PE)
  offchip_dim_x = np.array([n_col], dtype=np.float)

  for k in range(num_PE):
    # Extract rows of W for which (i mod num_PE) = k.
    W_section = np.zeros((num_rows, n_col))
    for i in range(num_rows):
      if k + i * num_PE >= n_row: break
      W_section[i:i + 1, :] = W[k + i * num_PE:k + i * num_PE + 1, :]
    VZP.append(matrixToCSC(W_section))

  # VZP_len - [VZP n_row num_rows]
  VZP_len = np.zeros((num_PE + 2,))
  for k in range(num_PE):
    VZP_len[k] = len(VZP[k][0])
  VZP_len[num_PE] = n_row
  VZP_len[num_PE + 1] = num_rows

  # Allocate arrays
  V = []
  for k in range(num_PE):
    V.append(chip.array(max_vsize_per_PE, 'V_{}'.format(k)))
  Z = []
  for k in range(num_PE):
    Z.append(chip.array(max_vsize_per_PE, 'Z_{}'.format(k)))
  P = []
  for k in range(num_PE):
    P.append(chip.array(max_psize_per_PE, 'P_{}'.format(k)))
  A = []
  for k in range(num_PE):
    A.append(chip.array(max_asize_per_PE, 'A_{}'.format(k)))
  B = []
  for k in range(num_PE):
    B.append(chip.array(max_asize_per_PE, 'B_{}'.format(k)))
  VL = chip.array(num_PE + 2, 'VL')
  dim_x = chip.array(1, 'dim_x')

  # len_V = 0
  # for k in range(num_PE):
  #   len_V += len(VZP[k][0])
  # V = chip.array(len_V, "V")
  # Z = chip.array(len_V, "Z")
  # P = chip.array((MV_dim + 1) * num_PE, "P")
  # offsets = chip.array(num_PE, "offsets")  # offsets delimiting where each PE begins
  # A = chip.array(MV_dim, "A")
  # B = chip.array(MV_dim, "B")
  # AJ = chip.array(1, "AJ")  # activation index
  # AV = chip.array(1, "AV")  # activation value

  # Control bits
  ctrl_r = 0

  # Module read_w
  read_w = chip.module(name='read_w')
  read_w.n_row = chip.flip_flop()
  read_w.n_col = chip.flip_flop()
  read_w.num_rows = chip.flip_flop() # number of rows of W sent to each PE

  def f_read_w():
    chip.read(VL, 0, VZP_len, 0, num_PE + 2)
    w_row = chip.get_item(VL, num_PE)
    w_col = read_w.n_row.read() # n_row of the previous layer
    read_w.n_row.write(w_row)
    w_num_rows = chip.get_item(VL, num_PE + 1)
    read_w.num_rows.write(w_num_rows)
    for k in range(num_PE):
      num = int(chip.get_item(VL, k))
      chip.read(V[k], 0, VZP[k][0], 0, num)
      chip.read(Z[k], 0, VZP[k][1], 0, num)
      chip.read(P[k], 0, VZP[k][2], 0, w_col + 1)

  read_w.set_func(f_read_w)

  # Module read_x (only need once)
  def f_read_x():
    chip.read(dim_x, 0, offchip_dim_x, 0, 1) # read dim_x
    n = int(chip.get_item(dim_x, 0))
    read_w.n_col.write(n)                    # Set the initial n_col
    read_w.n_row.write(n)                    # For the first layer
    chip.read_and_partition([(arr, 0) for arr in A], a, 0, n, 'cycle')

  read_x = chip.module(f_read_x, 'read_x')

  # def f_readInput():
  #   '''
  #   Read arrays onto chip.
  #   '''
  #   offset_vec = np.zeros(num_PE)
  #   onchip_offset = 0
  #   for k in range(num_PE):
  #     num = len(VZP[k][0])
  #     chip.read(V, onchip_offset, VZP[k][0], 0, num)
  #     chip.read(Z, onchip_offset, VZP[k][1], 0, num)
  #     offset_vec[k] = onchip_offset
  #     onchip_offset += num
  #     chip.read(P, k * (MV_dim + 1), VZP[k][2], 0, MV_dim + 1)
  #     # Read input vector. Only have to do this for the 1st layer.
  #     # (consider breaking this out into another function)
  #     chip.read(A, num_rows * k, [a[i] for i in range(k, MV_dim, num_PE)], 0, num_rows)
  #     # Initialize output array into which results will be accumulated.
  #     chip.read(B, 0, np.zeros(MV_dim), 0, MV_dim)
  #   chip.read(offsets, 0, offset_vec, 0, num_PE)

  # Module LNZD
  LNZD = chip.module(name='LNZD')
  # Flip-flops inside the LNZD (not necessary to implement as flip-flops)
  LNZD.X = np.zeros((num_PE,), dtype=np.int)   # the location of nonzero elements in a
  LNZD.V = np.zeros((num_PE,), dtype=np.float) # non-zero value
  LNZD.S = False                            # True if LNZD reads nothing
  LNZD.J = -1                               # Current J
  # Flip-flops between LNZD and PEs
  LNZD_PE_V = chip.flip_flop()
  LNZD_PE_J = chip.flip_flop()

  def f_LNZD():
    # Check X for non-zero values
    # Done by a large combinatorial logic
    if LNZD.S:
      return
    while True:
      if np.any(LNZD.X):
        t = np.nonzero(LNZD.X)[0][0]
        LNZD.X[t] = 0
        v = LNZD.V[t]
        # Broadcast
        LNZD_PE_V.write(v)
        LNZD_PE_J.write(LNZD.J * num_PE + t)
        break

      # No remaining non-zero values, read the next J
      LNZD.J += 1
      if LNZD.J == num_rows:
        # No non-zero values left
        LNZD.S = True
        break
      def single_LNZD(idx):
        v = chip.get_item(A[idx], LNZD.J)
        LNZD.V[idx] = v                  # Store the value in a flip-flop
        LNZD.X[idx] = 1 if v != 0 else 0 # Set the control bit
      chip.unrolled_loop(num_PE, single_LNZD)

  LNZD.set_func(f_LNZD)

  def r_LNZD():
    LNZD.X.fill(0)
    LNZD.V.fill(0)
    LNZD.S = False
    LNZD.J = -1

  LNZD.set_reset(r_LNZD)

  # Module PE
  def f_processingElement():
    '''
    Simulate an array of PEs running in parallel.
    '''

    # Get the next element of a from the activation queue.
    # The signal is broadcast to all PEs by hardware
    a_j = LNZD_PE_V.read()
    j = LNZD_PE_J.read()

    def single_PE(idx):
      col_start = int(chip.get_item(P[idx], j))
      col_end = int(chip.get_item(P[idx], j + 1))
      for i in range(col_start, col_end):
        w_ij = chip.get_item(V[idx], i)
        z = int(chip.get_item(Z[idx], i))
        b_i = chip.get_item(B[idx], z)
        add = chip.compute(w_ij * a_j, 'single_PE_{}_add'.format(idx))
        res = chip.compute(b_i + add, 'single_PE_{}_res'.format(idx))
        chip.array_write(B[idx], z, res)

    chip.unrolled_loop(num_PE, single_PE)
    # n_row becomes the new n_col
    n_row = read_w.n_row.read()
    read_w.n_col.write(n_row)

  PE = chip.module(f_processingElement, 'PE')

  # Module write_out (only need once)
  b = np.zeros((num_PE, n_row), dtype=float)  # will hold final computed result

  def f_write_out():
    n_col = int(read_w.n_col.read())
    for k in range(num_PE):
      chip.write(B[k], 0, b[k], 0, n_col)

  write_out = chip.module(f_write_out, 'write_out')

  # SPMV
  read_x()
  chip.join()
  # print(a)
  # print('---')
  # for k in range(num_PE):
  #   print(A[k].data)
  read_w()
  chip.join()
  # print(VZP)
  # for k in range(num_PE):
  #   print(V[k].data)
  #   print(Z[k].data)
  #   print(P[k].data)
  # for i in range(5):
  for k in range(num_PE):
    B[k].clear()
  LNZD()
  chip.join()
  # a_j = LNZD_PE_V.read()
  # j = LNZD_PE_J.read()
  # print(a_j)
  # print(j)
  # print(LNZD.S)
  while not LNZD.S:
    LNZD()
    PE()
    chip.join()
  write_out()
  chip.join()
  b = np.transpose(b).flatten()
  b = b[:n_row]
  # print(np.matmul(W, a))
  # print(b)
  # for k in range(num_PE):
  #   print(B[k].data)

  # Check the result
  eps = 1e-6
  assert(max(abs(np.matmul(W, a) - b)) < eps)
  print('Pass')
  chip.print_summary()

  # readInput()
  # chip.join()
  # for m in range(num_rows):
  #   # r = True
  #   LNZD(m)
  #   chip.join()
  #   while (broadcast(m)):
  #     chip.join()
  #     processingElements()
  #     chip.join()
  # writeOutput()
  # chip.join()

  # Make sure EIE performed SPMV correctly.
  # print(b)
  # print(np.matmul(W,a))
  # print("SPMV %s" % isCorrect(W, a, b))


if __name__ == '__main__':
  main()
