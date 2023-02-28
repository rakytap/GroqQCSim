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


qbit_num = 8
target_qbit = 0

# determine the soze of the unitary to be decomposed
matrix_size = int(2**qbit_num)



class UploadTopLevel(g.Component): 
	"""
	Top level component for initial upload of the state vector to the chip   --- currently unused
	"""

	def __init__(self):
		super().__init__()    


	def build(self, state_real_mt, state_imag_mt, time=0):   #Provide input matrices and a default time

		# component to perform the int8 matrix multiplication, splitted into chunks.
		with g.ResourceScope(name="uploadscope", is_buffered=True, time=0) as uploadscope :

			# make a copy of the real part somewhere else on the chip (just for testing, will be removed)
			print('uploadscope')   
			state_real_st = state_real_mt.read(streams=g.SG4[2], time=0) 

			uploaded_real_mt = state_real_st.write(name=f"uploaded_real", layout=f"H1(W), -1, S4(0-3)", program_output=True)
			uploaded_imag_mt = state_real_st.write(name=f"uploaded_imag", layout=f"H1(W), -1, S4(4-7)", program_output=True)
            
            
		return uploaded_real_mt, uploaded_imag_mt
		
		
class GateTopLevel(g.Component): 
	"""
	Top level component for a gate operation
	"""

	def __init__(self, state_real_mt, state_imag_mt):
		super().__init__()    
		
		self.state_real_mt = state_real_mt
		self.state_imag_mt = state_imag_mt


	def build(self, time=0):   #Provide input matrices and a default time

		# component to perform the int8 matrix multiplication, splitted into chunks.
		with g.ResourceScope(name="gatescope", is_buffered=True, time=0) as gatescope :

			# make a copy of the real part somewhere else on the chip (just for testing, will be removed)
			print('gatescope')   
			


			#permute_map_np = np.zeros( (4,256,), dtype=np.int32 )
			permute_map_np = np.zeros( (256,), dtype=np.uint32 )
			#permute_map_np = np.ones( (256), dtype=np.int ) * 17
			target_qbit_pair_diff = 1 << target_qbit
			print('target_qubit_pair_diff', target_qbit_pair_diff)
			
			for idx in range(256):
				if (idx < matrix_size):
					permute_map_np[idx] = idx ^ target_qbit_pair_diff
					#permute_map_np[0,idx] = idx ^ target_qbit_pair_diff
					#permute_map_np[1,idx] = idx ^ target_qbit_pair_diff
					#permute_map_np[2,idx] = idx ^ target_qbit_pair_diff
					#permute_map_np[3,idx] = idx ^ target_qbit_pair_diff
				else:
					permute_map_np[idx] = idx
			
			#permute_map_np[0] = 7


			#permute_map_np = permute_map_np.astype(np.uint8)

			print( permute_map_np )

			from groq.api import instruction as inst
			tmp = permute_map_np
			permute_idx = np.random.permutation(256)
			print( type(permute_idx) )
			print( type(permute_idx[0]) )
			print( permute_idx.shape )
			print( permute_idx.tolist() )
			print(type( tmp ) )
			print(type( tmp[0] ) )
			print( tmp.shape )
			print( tmp.tolist() )
			permute_map = inst.encode_permute_map(  permute_map_np.tolist() )

			permute_map_mt = g.from_data( np.asarray(permute_map, dtype=np.uint8) )

			
			state_real_mt_8 = g.reinterpret(self.state_real_mt, g.uint8 ) # reinterpret float32 as 4 uint8 values for permuter input
			#state_real_mt_8 = self.state_real_mt

			#state_real_st = self.state_real_mt.read(streams=g.SG4_W[2], time=0) 
			state_real_st_8 = g.permute_inner(state_real_mt_8, permute_map_mt, permutor_req=0, input_streams=[g.SG1[0], g.SG1[24]], output_streams=g.SG1[0], time=0 )

			state_real_st = g.reinterpret(state_real_st_8, g.float32 )
			#state_real_st = state_real_st_8

			result_mt = state_real_st.write(name=f"result", layout=f"H1(W), -1, S4(40-43)", program_output=True)
			#result_mt = state_real_st.write(name=f"result", layout=f"H1(W), -1, S1", program_output=True)
            
            
            
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

		State_input_real_mt = g.input_tensor(shape=(matrix_size,), dtype=g.float32, name="State_input_real", layout="H1(W), -1, S4(0-3)")
		State_input_imag_mt = g.input_tensor(shape=(matrix_size,), dtype=g.float32, name="State_input_imag", layout="H1(W), -1, S4(4-7)")

		#State_input_real_mt = g.input_tensor(shape=(matrix_size,), dtype=g.uint8, name="State_input_real", layout="H1(W), -1, S1")
		#State_input_imag_mt = g.input_tensor(shape=(matrix_size,), dtype=g.uint8, name="State_input_imag", layout="H1(W), -1, S1")

		State_input_real_mt.is_static = True
		State_input_imag_mt.is_static = True

		#top = UploadTopLevel()    # instantiate the top level component
		#uploaded_real_mt, uploaded_imag_mt = top(State_input_real, State_input_imag, time=0)    # call into the instance of the top level, providing your inputs and time

		pgm_pkg.compile_program_context(pgm1)

		#g.compile(base_name="QCsim", result_tensor=result)
		g.write_visualizer_data("QCsim_"+name)
		
	name = "gate"
	with pgm_pkg.create_program_context(name) as pgm2:

		#g.reserve_tensor(pgm1, pgm2, State_input_real_mt)


		# Creates a shared tensor to reuse the memory allocation made by program pgm1
		State_input_real_shared = g.shared_memory_tensor(State_input_real_mt, name="State_input_real_shared")
		State_input_imag_shared = g.shared_memory_tensor(State_input_imag_mt, name="State_input_imag_shared")

		top = GateTopLevel(State_input_real_shared, State_input_imag_shared)    # instantiate the top level component
		top(time=0)    # call into the instance of the top level, providing your inputs and time

		pgm_pkg.compile_program_context(pgm2)

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
	device.load(iop[1], unsafe_keep_entry_points=True)


	# run the first program
	pgm_1_output = invoke(device, iop, 0, 0, {"State_input_real": input_real, "State_input_imag": input_imag})

	# run the second program
	pgm_2_output = invoke(device, iop, 1, 0, {})

	return pgm_2_output["result"]



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
