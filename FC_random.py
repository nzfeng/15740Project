# Simulator for EIE, an accelerator which performs sparse matrix-vector
# multiplication.
# runfile('FC.py',['-A', '1' ,'-B' ,'0.01', '-C', '1', '-D', '100',
# '-M', '16777216', '-X', '1', '-P', '4', '-n', '784,512,10', '-s', '0.5'])
# 16777216 B = 16 MB

import numpy as np
from scipy.sparse import random
import argparse
import math
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

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
  parser.add_argument('--dims', '-n', type=str, required=True,
                      help='input_dim,layer1_out,...,layerk_out(output_dim)')
  parser.add_argument('--sparsity', '-s', type=float,
                      help='Density of the weight matrix W and initial input vector')
  parser.add_argument('--PE_num', '-P', type=int, help='Number of PEs')

  args = parser.parse_args()
  sparsity = args.sparsity
  num_PE = args.PE_num

  chip = Chip(args.dsp_num, args.dsp_cycle, args.index_cycle,
              args.mem_size, args.mem_overhead, args.mem_unit)

  # Hyper-parameters
  max_vsize_per_PE = (1 << 16)  # Size of v and z in each PE (num of items)
  max_asize_per_PE = (1 << 10)  # Size of a (there are 2 a in each PE)
  max_psize_per_PE = (1 << 10)  # Size of p

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
  dim_x = chip.array(2, 'dim_x')

  # Module read_w
  read_w = chip.module(name='read_w')
  read_w.n_row = chip.flip_flop()
  read_w.n_col = chip.flip_flop()
  read_w.num_rows = chip.flip_flop() # number of rows of W sent to each PE
  read_w.num_cols = chip.flip_flop() # number of cols of a in each LNZD

  def f_read_w(VZP_len, VZP):
    chip.read(VL, 0, VZP_len, 0, num_PE + 2)
    w_row = chip.get_item(VL, num_PE)
    w_col = int(read_w.n_row.read()) # n_row of the previous layer
    read_w.n_row.write(w_row)
    w_num_rows = chip.get_item(VL, num_PE + 1)
    num_rows1 = read_w.num_rows.read()
    read_w.num_rows.write(w_num_rows)
    read_w.num_cols.write(num_rows1)
    for k in range(num_PE):
      num = int(chip.get_item(VL, k))
      chip.read(V[k], 0, VZP[k][0], 0, num)
      chip.read(Z[k], 0, VZP[k][1], 0, num)
      chip.read(P[k], 0, VZP[k][2], 0, w_col + 1)

  read_w.set_func(f_read_w)

  # Module read_x (only need once)
  def f_read_x(dx, a):
    chip.read(dim_x, 0, dx, 0, 2) # read dim_x
    n = int(chip.get_item(dim_x, 0))
    num_rows1 = chip.get_item(dim_x, 1)
    read_w.n_col.write(n)                    # Set the initial n_col
    read_w.n_row.write(n)                    # For the first layer
    read_w.num_rows.write(num_rows1)
    chip.read_and_partition([(arr, 0) for arr in A], a, 0, n, 'cycle')

  read_x = chip.module(f_read_x, 'read_x')

  # Module read_b
  BI = []
  for k in range(num_PE):
    BI.append(chip.array(max_asize_per_PE, 'BI_{}'.format(k)))
  def f_read_b(b):
    n = int(read_w.n_row.read())
    chip.read_and_partition([(arr, 0) for arr in BI], b, 0, n, 'cycle')

  read_b = chip.module(f_read_b, 'read_b')

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

  def f_LNZD(A):
    # Check X for non-zero values
    # Done by a large combinatorial logic
    nc = int(read_w.num_cols.read())
    # print('num_cols = {}'.format(nc))
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
      if LNZD.J == nc:
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
  def f_processingElement(B):
    '''
    Simulate an array of PEs running in parallel.
    '''

    # Get the next element of a from the activation queue.
    # The signal is broadcast to all PEs by hardware
    a_j = LNZD_PE_V.read()
    j = int(LNZD_PE_J.read())

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
    n_row1 = read_w.n_row.read()
    read_w.n_col.write(n_row1)

  PE = chip.module(f_processingElement, 'PE')

  # Module relu
  def f_relu(B, need_relu):
    num_rows = int(read_w.num_rows.read())

    def single_relu(idx):
      for i in range(num_rows):
        b_i = chip.get_item(B[idx], i)
        bi_r = chip.get_item(BI[idx], i)
        b_i_bi = chip.compute(b_i + bi_r, 'single_relu_{}_b_i_bi'.format(idx))
        b_i_relu = chip.compute(max(b_i_bi, 0), 'single_relu_{}_b_i_relu'.format(idx))
        chip.array_write(B[idx], i, b_i_relu if need_relu else b_i_bi)

    chip.unrolled_loop(num_PE, single_relu)

  ReLU = chip.module(f_relu, 'ReLU')

  # Module write_out (only need once)
  def f_write_out(B, b):
    n_col = int(read_w.n_col.read())
    for k in range(num_PE):
      chip.write(B[k], 0, b[k], 0, n_col)

  write_out = chip.module(f_write_out, 'write_out')

  ########################
  # Random inputs
  dims = args.dims.split(',')
  dims = [int(s) for s in dims]
  assert len(dims) >= 2

  # Input data a
  n_col = dims[0]
  num_cols = math.ceil(n_col / num_PE)
  offchip_dim_x = np.array([n_col, num_cols], dtype=np.float)

  # Weights
  VZP_list = []
  VZP_len_list = []
  bi_list = []  # biases
  for layer_i in range(len(dims) - 1):
    n_col = dims[layer_i]
    n_row = dims[layer_i + 1]

    # Create random sparse matrix W, and sparse vector a.
    # Portions of W are put into CSC format by CPU.
    W = random(n_row, n_col, density=sparsity, dtype=float)
    W = np.array(W.A)
    bi = random(n_row, 1, density=sparsity, dtype=float)
    bi = np.array(bi.A).flatten()
    bi_list.append(bi)

    # A list of v, z, p vectors (CSC representations) for each portion of W in
    # each PE.
    VZP = []  # = [[v1, z1, p1], [v2, z2, p2],..., [vN, zN, pN]]
    num_rows = math.ceil(n_row / num_PE)

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

    VZP_list.append(VZP)
    VZP_len_list.append(VZP_len)

  # Output: will hold final result
  n_row = dims[-1]  # output dim
  b = np.zeros((num_PE, n_row), dtype=float)

  # Prepare inputs
  testset = MNIST(root='data', train=False, download=True,
                  transform=transforms.ToTensor())

  ########################
  # Chip code starts here
  # FC ReLU NN
  # Modules:
  # - read_w(VZP_len, VZP)
  # - read_x(dx, a)
  # - read_b(b)
  # - LNZD(A)
  # - PE(B)
  # - ReLU(B)
  # - write_out(B, b)
  n = 0
  c = 0
  for (x, y) in testset:
    n += 1
    a = x.view(-1).numpy()
    # Control bit
    ctrl_r = 0

    # Prologue
    # 1. Read input data
    read_x(offchip_dim_x, a)
    chip.join()


    # 2. Read first layer weights
    read_w(VZP_len_list[0], VZP_list[0])
    chip.join()

    # Main loop
    for layer_i in range(len(dims) - 1):
      # 1. read_b and compute
      read_b(bi_list[layer_i])
      A_now = A if ctrl_r == 0 else B
      B_now = B if ctrl_r == 0 else A
      for k in range(num_PE):
        B_now[k].clear()
      with chip.module_group('compute'):
        LNZD(A_now)
        chip.join()
        while not LNZD.S:  # LNZD.S is a global signal
          # LNZD and PE work in parallel
          LNZD(A_now)
          PE(B_now)
          chip.join()
        LNZD.reset()
      chip.join()

      # 2. read_w and relu
      if layer_i == len(dims) - 2:  # no w and relu in the final layer
        break
      read_w(VZP_len_list[layer_i + 1], VZP_list[layer_i + 1])
      ReLU(B_now, True)
      chip.join()
      ctrl_r = 1 - ctrl_r

      # for debug
      # n_row = dims[layer_i + 1]
      # print(n_row)
      # b_tmp = np.zeros((num_PE, n_row), dtype=float)
      # write_out(B_now, b_tmp)
      # chip.join()
      # b_tmp = np.transpose(b_tmp).flatten()
      # b_tmp = b_tmp[:n_row]
      #
      # eps = 1e-6
      # assert (max(abs(y_list[layer_i] - b_tmp)) < eps)
      # print('check')

    # Epilogue
    # 1. Add the bias of the final layer
    B_now = B if ctrl_r == 0 else A
    ReLU(B_now, False)
    chip.join()

    # 2. write out
    n_row = dims[-1]
    write_out(B_now, b)
    chip.join()
    b0 = np.transpose(b).flatten()
    b0 = b0[:n_row]

    c += chip.read_cycles()
    chip.reset()
    if n == 500:
      break

  print('Avg cycles: {}'.format(c / n))

if __name__ == '__main__':
  main()
