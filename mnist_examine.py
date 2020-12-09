""" Load in a trained MNIST network to examine its sparsity patterns. """
import numpy as np
import math
import argparse
import pickle

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dsp_num', '-D', type=int, help='Number of DSPs')
	parser.add_argument('--PE_num', '-P', type=int, help='Number of PEs')
	parser.add_argument('--dims', '-n', type=str, required=True,
                      	help='input_dim,layer1_out,...,layerk_out(output_dim)')
	parser.add_argument('--pkl_file', '-f', type=str, help='Model weights')

	args = parser.parse_args()
	num_PE = args.PE_num
	dims = args.dims.split(',')
	dims = [int(s) for s in dims]

	with open(args.pkl_file, 'rb') as f:
		dat = pickle.load(f)

	# For each layer, output matrix densities per PE.
	for layer_i in range(len(dims) - 1):
		X = "" # datapoints
		W = dat[layer_i][0] # weight matrix
		n_col = dims[layer_i]
		n_row = dims[layer_i + 1]
		num_rows = math.ceil(n_row / num_PE) # number of rows per PE
		for k in range(num_PE):
			# Extract rows of W for which (i mod num_PE) = k.
			W_section = np.zeros((num_rows, n_col))
			for i in range(num_rows):
				if k + i * num_PE >= n_row: break
				W_section[i:i + 1, :] = W[k + i * num_PE:k + i * num_PE + 1, :]
			X += "%f," % ((W_section != 0).sum() / len(W_section.flatten()))

		print(X)

if __name__ == '__main__':
	main()