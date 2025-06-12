from gate_L_base import Gate_L_base
from memory_layouts import *
import groq.api as g
import numpy as np

	
	

class Gate(Gate_L_base):
	"""
	A class used to implement the gate operation on qubits S with no control qubits


	Attributes
	----------
    
	mxm_permutation : bool
		If true, MXM planes are used to generate permuted state pairs for the linear transfromation. If False, the permutor is used for the same task. 
		Permutor takes twice as much time as the MXM.

	"""

	def __init__(self, pgm_pkg, mxm_permutation=True):
		"""
		Parameters
		----------
		pgm_pkg : ProgramPackage
			The program package in which the gate operation program is generated
		mxm_permutation : bool
			If true, MXM planes are used to generate permuted state pairs for the linear transfromation. If False, the permutor is used for the same task. 
			Permutor takes twice as much time as the MXM.
		"""
        
		super().__init__(pgm_pkg, mxm_permutation = mxm_permutation )
		



	
	def gate_L_program(self):
		"""
		Function implementing the L gate operation on the Groq chip
		"""
        
		#############################################################################################################x		
		# program to perform gate operation in a target qubit less than small_qbit_num_limit
		name = "gate_L"
		
		with self.pgm_pkg.create_program_context(name) as pgm6:


			with g.ResourceScope(name=f"apply_gate_scope", is_buffered=True, predecessors=[], time=0) as apply_gate_scope :


				#index_pairs = [ [x, x^(1<<(self.target_qubit-8))] for x in range(memory_addresses) if x < x^(1<<(self.target_qubit-8)) ]

				read_out_qubit_0_memory_tensor_st, read_out_qubit_1_memory_tensor_st = self.select_target_qbit_read_map_west()
				
				# readout quantum state to be stransformed
				Psi_transformed_real_state_0_st, Psi_transformed_imag_state_0_st, Psi_transformed_real_state_1_st, Psi_transformed_imag_state_1_st  = \
							 self.read_out_state_2_transform( read_out_qubit_0_memory_tensor_st, read_out_qubit_1_memory_tensor_st)

				#Psi_transformed_real_state_0_st.write( name='ooo2', layout='H1(E), -1, S4', program_output=True, time=0)

				# readout gate elements	
				element_num = read_out_qubit_0_memory_tensor_st.shape[0]
						
				gate_element_real_st, gate_element_imag_st, gate_element_real_copy_st, gate_element_imag_copy_st = \
							self.read_out_gate_elements( element_num )	
				'''	
				gate_element_real_st.write( name='ooo0', layout='H1(W), -1, S4')
				gate_element_imag_st.write( name='ooo1', layout='H1(W), -1, S4', time=0)
				gate_element_real_copy_st.write( name='ooo2', layout='H1(W), -1, S4', program_output=True)
				gate_element_imag_copy_st.write( name='ooo3', layout='H1(W), -1, S4', time=2)
				'''
				
				# calculate the VXM round trip                            
				Psi_transformed_real_st, Psi_transformed_imag_st = self.two_complex_dot_product( 
												Psi_transformed_real_state_0_st,#pairs0[0], 
												Psi_transformed_imag_state_0_st,#pairs0[1], 
												Psi_transformed_real_state_1_st,#pairs1[0], 
												Psi_transformed_imag_state_1_st,#pairs1[1], 
												gate_element_real_st, 
												gate_element_imag_st, 
												gate_element_real_copy_st, 
												gate_element_imag_copy_st)
												
				
				
				# export the calculated state into memory and set program output
				target_qubit_L_write_map_st = self.select_target_qbit_write_map_east()
				#target_qubit_L_write_map_st.write( name='ooo2', layout='H1(W), -1, S1', program_output=True)
				self.write_tranformed_states_2_mem( Psi_transformed_real_st, Psi_transformed_imag_st, target_qubit_L_write_map_st )
				
				
				
			with g.ResourceScope(name=f"increase_gate_counters_scope", is_buffered=True, predecessors=[apply_gate_scope], time=None) as increase_gate_counters_scope :
			
				self.increment_gate_counters()	

				
			self.pgm_pkg.compile_program_context(pgm6)
			
			#g.compile(base_name="QCsim", result_tensor=result)
			g.write_visualizer_data("QCsim_"+name)		
		
