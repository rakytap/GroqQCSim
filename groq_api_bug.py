import numpy as np
import groq.api as g
import groq.api.tensor as tensor
from groq.runner import tsp
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

"""

"""


class U3GateComponent(g.Component):
    def __init__(self):
        super().__init__()
        
    def two_complex_dot_product(s_r_0, s_i_0, s_r_1, s_i_1, g_r_0, g_i_0, g_r_1, g_i_1, time):
        s_r_0_st = s_r_0.read(streams=g.SG4[7]) #direction=Direction.EASTWARD
        s_i_0_st = s_i_0.read(streams=g.SG4[5]) #direction=Direction.EASTWARD
        s_r_1_st = s_r_1.read(streams=g.SG4[1]) #direction=Direction.EASTWARD
        s_i_1_st = s_i_1.read(streams=g.SG4[4]) #direction=Direction.EASTWARD
        g_r_0_st = g_r_0.read(streams=g.SG4[7]) #direction=Direction.WESTWARD
        g_i_0_st = g_i_0.read(streams=g.SG4[4]) #direction=Direction.WESTWARD
        g_r_1_st = g_r_1.read(streams=g.SG4[3]) #direction=Direction.WESTWARD
        g_i_1_st = g_i_1.read(streams=g.SG4[6]) #direction=Direction.WESTWARD
        rr0_st = g.mul(s_r_0_st, g_r_0_st, alus=[14], output_streams=[g.SG4[0]], time=0) #input_streams=[g.SG4_E[7], g.SG4_W[7]], direction=Direction.WESTWARD
        ii0_st = g.mul(s_i_0_st, g_i_0_st, alus=[5], output_streams=[g.SG4[2]], time=0) #input_streams=[g.SG4_E[5], g.SG4_W[4]], direction=Direction.WESTWARD
        ri0_st = g.mul(s_r_0_st, g_i_0_st, alus=[10], output_streams=[g.SG4[6]]) #input_streams=[g.SG4_E[7], g.SG4_W[4]], direction=Direction.WESTWARD
        ir0_st = g.mul(s_i_0_st, g_r_0_st, alus=[9], output_streams=[g.SG4[5]]) #input_streams=[g.SG4_E[5], g.SG4_W[7]], direction=Direction.WESTWARD
        r0_st = g.sub(rr0_st, ii0_st, alus=[0], output_streams=[g.SG4[0]]) #input_streams=[g.SG4_W[0], g.SG4_W[2]], direction=Direction.EASTWARD
        i0_st = g.add(ri0_st, ir0_st, alus=[8], output_streams=[g.SG4[6]]) #input_streams=[g.SG4_W[6], g.SG4_W[5]], direction=Direction.EASTWARD
        rr1_st = g.mul(s_r_1_st, g_r_1_st, alus=[3], output_streams=[g.SG4[3]], time=0) #input_streams=[g.SG4_E[1], g.SG4_W[3]], direction=Direction.WESTWARD
        ii1_st = g.mul(s_i_1_st, g_i_1_st, alus=[11], output_streams=[g.SG4[5]], time=0) #input_streams=[g.SG4_E[4], g.SG4_W[6]], direction=Direction.WESTWARD
        ri1_st = g.mul(s_r_1_st, g_i_1_st, alus=[15], output_streams=[g.SG4[1]]) #input_streams=[g.SG4_E[1], g.SG4_W[6]], direction=Direction.WESTWARD
        ir1_st = g.mul(s_i_1_st, g_r_1_st, alus=[7], output_streams=[g.SG4[2]]) #input_streams=[g.SG4_E[4], g.SG4_W[3]], direction=Direction.WESTWARD
        r1_st = g.sub(rr1_st, ii1_st, alus=[6], output_streams=[g.SG4[2]]) #input_streams=[g.SG4_W[3], g.SG4_W[5]], direction=Direction.WESTWARD
        i1_st = g.add(ri1_st, ir1_st, alus=[2], output_streams=[g.SG4[1]]) #input_streams=[g.SG4_W[1], g.SG4_W[2]], direction=Direction.WESTWARD
        r_st = g.add(r0_st, i0_st, alus=[13], output_streams=[g.SG4[7]]) #input_streams=[g.SG4_E[0], g.SG4_E[6]], direction=Direction.WESTWARD
        i_st = g.add(r1_st, i1_st, alus=[1], output_streams=[g.SG4[3]]) #input_streams=[g.SG4_W[2], g.SG4_W[1]], direction=Direction.WESTWARD
        r_delay_st = g.vxm_identity(r_st, alus=[12], output_streams=[g.SG4[1]]) #input_streams=[g.SG4_W[7]], direction=Direction.WESTWARD
        i_delay_st = g.vxm_identity(i_st, alus=[4], output_streams=[g.SG4[5]]) #input_streams=[g.SG4_W[3]], direction=Direction.WESTWARD
        return r_st, i_st, r_delay_st, i_delay_st #directions=[Direction.WESTWARD, Direction.WESTWARD, Direction.WESTWARD, Direction.WESTWARD]

    def build(self, s_0_r, s_0_i, s_1_r, s_1_i, g_0_r, g_0_i, g_1_r, g_1_i, time=0):
        _st, i_st, r_delay_st, i_delay_st = U3GateComponent.two_complex_dot_product(
            s_0_r, s_0_i, s_1_r, s_1_i, g_0_r, g_0_i, g_1_r, g_1_i, time=time)
        _st, i_st, r_delay_st, i_delay_st = _st.write(name = "_st"),\
            i_st.write(name = "i_st"), r_delay_st.write(name = "r_delay_st"), i_delay_st.write(name = "i_delay_st")
        return _st, i_st, r_delay_st, i_delay_st
        # return _st, i_st, r_delay_st, i_delay_st

    
num_qubits = 3
pow2qb = 1 << num_qubits

a_layout   = "H1(W), -1, S8(0-7)"
a_i_layout = "H1(W), -1, S8(8-15)"
b_layout   = "H1(W), -1, S8(16-23)"
b_i_layout = "H1(W), -1, S8(24-31)"
c_layout   = "H1(E), -1, S8(0-7)" 
c_i_layout  = "H1(E), -1, S8(16-23)"
d_layout   = "H1(E), -1, S8(0-7)" 
d_i_layout = "H1(E), -1, S8(8-15)"

s0_r_mt = g.input_tensor(shape = (2 , pow2qb), dtype = g.float32, name = "s_0_r",
                       storage_req=tensor.create_storage_request(layout=a_layout))
s0_i_mt = g.input_tensor(shape = (2 , pow2qb), dtype = g.float32, name = "s_0_i",
                       storage_req=tensor.create_storage_request(layout=a_i_layout))

s1_r_mt = g.input_tensor(shape = (2 , pow2qb), dtype = g.float32, name = "s_1_r",
                       storage_req=tensor.create_storage_request(layout=b_layout))
s1_i_mt = g.input_tensor(shape = (2 , pow2qb), dtype = g.float32, name = "s_1_i",
                       storage_req=tensor.create_storage_request(layout=b_i_layout))

g0_r_mt = g.input_tensor(shape = (2 , pow2qb), dtype = g.float32, name = "g_0_r",
                       storage_req=tensor.create_storage_request(layout=c_layout))
g0_i_mt = g.input_tensor(shape = (2 , pow2qb), dtype = g.float32, name = "g_0_i",
                       storage_req=tensor.create_storage_request(layout=c_i_layout))

g1_r_mt = g.input_tensor(shape = (2 , pow2qb), dtype = g.float32, name = "g_1_r",
                       storage_req=tensor.create_storage_request(layout=d_layout))
g1_i_mt = g.input_tensor(shape = (2 , pow2qb), dtype = g.float32, name = "g_1_i",
                       storage_req=tensor.create_storage_request(layout=d_i_layout))


u3_gate = U3GateComponent()
result = u3_gate(s0_r_mt, s0_i_mt, s1_r_mt, s1_i_mt, g0_r_mt, g0_i_mt, g1_r_mt, g1_i_mt, time = 0)
iop_file = g.compile(base_name="u3_gate", result_tensor=result)
print(iop_file)

g.write_visualizer_data("u3_gate_tensors")
#!groqview u3_gate_tensors/visdata.json


u3_gate_program = tsp.create_tsp_runner(iop_file)

# t1_data = np.random.array([[0,1,2,3,4,5]], dtype=np.int8)
random_unitary, _ = np.linalg.qr(
    np.random.uniform(-1, 1, (pow2qb, pow2qb)).astype(np.float32) +
        np.random.uniform(-1, 1, (pow2qb, pow2qb)).astype(np.float32)*1j)

real_parts = []
imaginary_parts = []
np_s0_r_mt, np_s0_i_mt, np_s1_r_mt, \
    np_s1_i_mt, np_g0_r_mt, np_g0_i_mt, np_g1_r_mt, np_g1_i_mt = [],[],[],[],[],[],[],[]

test_data = [np_s0_r_mt, np_s0_i_mt, np_s1_r_mt, np_s1_i_mt, np_g0_r_mt, np_g0_i_mt, np_g1_r_mt, np_g1_i_mt]
# Extracting real and imaginary parts
for i in range(0,7,2):
    u_array = random_unitary[i]
    for number in u_array:
        test_data[i].append(number.real)
        test_data[i+1].append(number.imag)
    test_data[i] = np.ascontiguousarray(test_data[i].view(np.float32).T)
    test_data[i+1] = np.ascontiguousarray(test_data[i+1].view(np.float32).T)

result = u3_gate_program(*test_data)
_st = result["_st"]
i_st = result["i_st"]
r_delay_st = result["r_delay_st"]
i_delay_st = result["i_delay_st"]

print(_st,
i_st,
r_delay_st,
i_delay_st,)
# result = result['input_1'][0]+result['input_1'][1]*1j

# print(random_unitary[0].view(np.float32).shape)
# print(f"Numpy result ---->\n", random_unitary[0])
# print(f"Groq result ---->\n", result)
# print(f"Validated -->", np.array_equal(random_unitary[0], result))