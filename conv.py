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

  # Conv