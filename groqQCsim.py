import groq.api as g
import groq.api.nn as nn
import groq.runner.tsp as tsp;
import numpy as np

import os
import shutil
from typing import List

try:
    import groq.runtime.driver as runtime
except ImportError:
    # raise ModuleNotFoundError("groq.runtime")
    print('Error: ModuleNotFoundError("groq.runtime")')

import time
print("Python packages imported successfully")


qbit_num = 7

# determine the soze of the unitary to be decomposed
matrix_size = int(2**qbit_num)



class UploadTopLevel(g.Component): 
	"""
	Top level component for initial upload of the state vector to the chip
	"""

	def __init__(self):
		super().__init__()    


	def build(self, state_real_mt, state_imag_mt, time=0):   #Provide input matrices and a default time

		# component to perform the int8 matrix multiplication, splitted into chunks.
		with g.ResourceScope(name="uploadscope", is_buffered=True, time=0) as uploadscope :

			# make a copy of the real part somewhere else on the chip (just for testing, will be removed)
			print('uploadscope')   
			state_real_st = state_real_mt.read(streams=g.SG4_W[2], time=0) 

			result_mt = state_real_st.write(name=f"result", layout=f"H1(W), -1, S4(40-43)", program_output=True)
            
            
		return result_mt




def compile() -> List[str]:
	"""Compiles a program package with 2 programs.

	Return: (List[str]): A list of IOP files.
	"""
	output_dir = "./build_iop"
	shutil.rmtree(output_dir, ignore_errors=True)
	pgm_pkg = g.ProgramPackage("QCsim_multi_program", output_dir)



	# Defines a program to upload the inputs
	name = "upload"
	with pgm_pkg.create_program_context(name) as pgm1:

		State_input_real = g.input_tensor(shape=(matrix_size,), dtype=g.float32, name="State_input_real", layout="H1(W), -1, S4(0-3)")
		State_input_imag = g.input_tensor(shape=(matrix_size,), dtype=g.float32, name="State_input_imag", layout="H1(W), -1, S4(4-7)")

		#z = g.add(State_input_real, State_input_imag, time=10).write(name="Z", program_output=True)

		top = UploadTopLevel()    # instantiate the top level component
		top(State_input_real, State_input_imag, time=0)    # call into the instance of the top level, providing your inputs and time

		pgm_pkg.compile_program_context(pgm1)

		#g.compile(base_name="QCsim", result_tensor=result)
		g.write_visualizer_data("QCsim_"+name)


	iops = pgm_pkg.assemble()

	return iops



def invoke(device, iop, pgm_num, ep_num, tensors):
	"""Low level interface to the device driver to access multiple programs. A higher level abstraction
    will be provided in a future release that offers access to multiple programs and entry points."""

	pgm = iop[pgm_num]
	ep = pgm.entry_points[ep_num]
	input_buffer = runtime.BufferArray(ep.input, 1)[0]
	output_buffer = runtime.BufferArray(ep.output, 1)[0]

	if ep.input.tensors:
		for input_tensor in ep.input.tensors:
			if input_tensor.name not in tensors:
				raise ValueError(f"Missing input tensor named {input_tensor.name}")
			input_tensor.from_host(tensors[input_tensor.name], input_buffer)

	device.invoke(input_buffer, output_buffer)
	outs = {}

	if ep.output.tensors:
		for output_tensor in ep.output.tensors:
			result_tensor = output_tensor.allocate_numpy_array()
			output_tensor.to_host(output_buffer, result_tensor)
			outs[output_tensor.name] = result_tensor
	return outs


def run(iop_file, input_real, input_imag):
	"""
    This function interacts with the device driver at a lower level to show
    the control of loading 2 programs and invoking each program through
    program entry points. A higher level abstraction will be made available
    to load and invoke multiple programs in a future release.
	"""

	np.set_printoptions(linewidth=1000, threshold=10000)

	if not os.path.exists(iop_file):
		raise Exception(f"IOP file does not exist: {iop_file}")

	print(f"Running programs from {iop_file}")



	iop = runtime.IOProgram(iop_file)
	device = runtime.devices[0]
	device.open()
	device.load(iop[0], unsafe_keep_entry_points=True)
	#device.load(iop[1], unsafe_keep_entry_points=True)


	# run the first program
	pgm_1_output = invoke(device, iop, 0, 0, {"State_input_real": input_real, "State_input_imag": input_imag})

	return pgm_1_output["result"]



def main( State_orig_real_float32, State_orig_imag_float32 ):
	"""Compiles and runs the example programs."""

	iop_files = compile()
	print(f"Program compiled to IOP file: {iop_files}")



	real_part = run(iop_files[0], State_orig_real_float32, State_orig_imag_float32)

	return real_part




if __name__ == "__main__":
    main()
'''
# compile the program
iop_file = g.compile(base_name="QCsim", result_tensor=result)
g.write_visualizer_data("QCsim")


program = tsp.create_tsp_runner(iop_file)
t0 = time.time()
result = program(State_input_real=State_orig_real_float32, State_input_imag=State_orig_real_float32)
groq_result_mm = result['result']
'''
