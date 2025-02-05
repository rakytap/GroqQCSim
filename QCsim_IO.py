from memory_layouts import *
import groq.api as g
import numpy as np
from groq.api import instruction as inst

class IO:
	def __init__(self, pgm_pkg, mxm_permutation=True):

		self.pgm_pkg = pgm_pkg
		self.gate_kernels_real_packed_mt = None
		self.gate_kernels_imag_packed_mt = None
		self.State_input_real_mt = None
		self.State_input_imag_mt = None
		self.State_input_real_copy_mt = None
		self.State_input_imag_copy_mt = None
		self.state_permute_map_selector_mt = None
		self.permute_maps_west = None
		self.permute_maps_east = None
		self.state_1_broadcasted_mt = None
		self.state_0_broadcasted_mt = None
		self.control_state_1_broadcasted_mt = None
		self.control_state_0_broadcasted_mt = None
		self.ones_mt = None
		self.zeros_mt = None

		# create reusable memory requests
		self.state_imag_request = g.tensor.create_storage_request( layout=self.ML.layout_State_input_imag )
		self.state_real_request = g.tensor.create_storage_request( layout=self.ML.layout_State_input_real )

		self.state_imag_copy_request = g.tensor.create_storage_request( layout=self.ML.layout_State_input_imag_copy )
		self.state_real_copy_request = g.tensor.create_storage_request( layout=self.ML.layout_State_input_real_copy )

		# for target qubits less than 8 MXM slices or the permutors can be used to generate the pairs for the state vector elements
		# set mxm_permutation to True when MXM is used, or set to False when permutor is used
		self.mxm_permutation = mxm_permutation
		
		# the memory addresses for gathering the mxm permutation matrices
		self.state_permute_map_selector_expanded = None
		
		




	def upload_state_program(self):
		# Defines a program to upload the inputs
		name = "uplod_state_vector"
		with self.pgm_pkg.create_program_context(name) as pgm1:


			self.State_input_real_mt = g.input_tensor(shape=(self.ML.memory_addresses, matrix_size,), 
							dtype=g.float32, 
							name="State_input_real", 
							storage_req=self.state_real_request)
							
			self.State_input_imag_mt = g.input_tensor(shape=(self.ML.memory_addresses, matrix_size,), 
							dtype=g.float32, 
							name="State_input_imag", 
							storage_req=self.state_imag_request)
							
							
			# duplicate the state vector to do linear combinations associated with the unitary operation
			State_input_real_st = self.State_input_real_mt.read(streams=[g.SG4[1]], time = 0)
			State_input_imag_st = self.State_input_imag_mt.read(streams=[g.SG4[2]], time = 0)

			self.State_input_real_copy_mt = State_input_real_st.write( name="State_input_real_copy", storage_req=self.state_real_copy_request)
			self.State_input_imag_copy_mt = State_input_imag_st.write( name="State_input_imag_copy", storage_req=self.state_imag_copy_request)
			
			self.State_input_real_mt.is_static      = True
			self.State_input_imag_mt.is_static	= True
			self.State_input_real_copy_mt.is_static = True
			self.State_input_imag_copy_mt.is_static = True
			
			
	
	def download_state_program(self):
		# Defines a program to upload the inputs
		name = "download_state_vector"
		with self.pgm_pkg.create_program_context(name) as pgm1:


			# Creates a shared tensor to reuse the memory allocation made by program pgm1
			State_input_real_shared = g.shared_memory_tensor(self.State_input_real_mt, name="State_transformed_real")
			State_input_imag_shared = g.shared_memory_tensor(self.State_input_imag_mt, name="State_transformed_imag")


			State_input_real_shared.set_program_output()
			State_input_imag_shared.set_program_output()
	
			self.pgm_pkg.compile_program_context(pgm1)
