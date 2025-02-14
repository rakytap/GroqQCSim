import os, subprocess
import numpy as np
import groq.api as g

def rintmulinv_float16(ARG0, c_1042479491, c_2139095040, c_1065353216):
        ARG0_st_m7 = ARG0.read(streams=g.SG2[0]) #direction=Direction.EASTWARD
        x_0_st_m6 = g.cast(ARG0_st_m7, dtype=g.float32, alus=[0], output_streams=[g.SG4[2]]) #input_streams=[g.SG4_E[0]], direction=Direction.EASTWARD
        c_1042479491_st_m6 = c_1042479491.read(streams=g.SG4[3]) #direction=Direction.WESTWARD
        x_1_st_m5 = g.mul(x_0_st_m6.reinterpret(g.float32), c_1042479491_st_m6, overflow_mode=g.OverflowMode.MODULAR, alus=[2], output_streams=[g.SG4[2]]) #input_streams=[g.SG4_E[2], g.SG4_W[3]], direction=Direction.EASTWARD
        x_1_id3_st_m4 = g.vxm_identity(x_1_st_m5, alus=[3], output_streams=[g.SG4[2]]) #input_streams=[g.SG4_E[2]], direction=Direction.WESTWARD
        x_1_id6_st_m3 = g.vxm_identity(x_1_id3_st_m4, alus=[6], output_streams=[g.SG4[3]]) #input_streams=[g.SG4_W[2]], direction=Direction.WESTWARD
        x_1_id1_st_m2 = g.vxm_identity(x_1_id6_st_m3, alus=[1], output_streams=[g.SG4[1]]) #input_streams=[g.SG4_W[3]], direction=Direction.EASTWARD
        c_2139095040_st_m5 = c_2139095040.read(streams=g.SG4[4]) #direction=Direction.WESTWARD
        x_2_st_m4 = g.bitwise_and(x_1_st_m5.reinterpret(g.uint32), c_2139095040_st_m5, alus=[7], output_streams=[g.SG4[4]]) #input_streams=[g.SG4_E[2], g.SG4_W[4]], direction=Direction.WESTWARD
        x_3_st_m3 = g.cast(x_2_st_m4.reinterpret(g.float32), dtype=g.uint32, alus=[10], output_streams=[g.SG4[6]]) #input_streams=[g.SG4_W[4]], direction=Direction.EASTWARD
        c_1065353216_st_m3 = c_1065353216.read(streams=g.SG4[1]) #direction=Direction.WESTWARD
        x_4_st_m2 = g.min(x_3_st_m3.reinterpret(g.uint32), c_1065353216_st_m3, alus=[15], output_streams=[g.SG4[6]]) #input_streams=[g.SG4_E[6], g.SG4_W[1]], direction=Direction.WESTWARD
        x_5_st_m1 = g.mul(x_4_st_m2.reinterpret(g.float32), x_1_id1_st_m2, overflow_mode=g.OverflowMode.MODULAR, alus=[14], output_streams=[g.SG4[1]]) #input_streams=[g.SG4_W[6], g.SG4_E[1]], direction=Direction.WESTWARD
        x_6_st_m2 = g.cast(x_1_id6_st_m3, dtype=g.int32, alus=[5], output_streams=[g.SG4[4]]) #input_streams=[g.SG4_W[3]], direction=Direction.WESTWARD
        x_7_st_m1 = g.cast(x_6_st_m2.reinterpret(g.int32), dtype=g.float32, alus=[8], output_streams=[g.SG4[7]]) #input_streams=[g.SG4_W[4]], direction=Direction.EASTWARD
        rintmulinv_float16_st_p0 = g.max(x_5_st_m1.reinterpret(g.uint32), x_7_st_m1.reinterpret(g.uint32), time=0, alus=[13], output_streams=[g.SG4[6]]) #input_streams=[g.SG4_W[1], g.SG4_E[7]], direction=Direction.WESTWARD
        return rintmulinv_float16_st_p0 #directions=[Direction.WESTWARD]

def rintmulinv_float16_apply(ARG0, predecessors=None):
        with g.ResourceScope(name="rintmulinv_float16_scope", is_buffered=True, time=0 if predecessors is None else None, predecessors=predecessors) as rintmulinv_float16_scope:
                c_2139095040 = g.from_data(np.array([2139095040]*320, dtype=np.uint32), name='c_2139095040', layout="-1, H1(E), S4")
                c_1042479491 = g.from_data(np.array([0.15915493667125702]*320, dtype=np.float32), name='c_1042479491', layout="-1, H1(E), S4")
                c_1065353216 = g.from_data(np.array([1065353216]*320, dtype=np.uint32), name='c_1065353216', layout="-1, H1(E), S4")
                rintmulinv_float16_mt = rintmulinv_float16(ARG0, c_1042479491, c_2139095040, c_1065353216)
                rintmulinv_float16_mt = rintmulinv_float16_mt.write(name="rintmulinv_float16", layout=f"-1, H1(W), S{rintmulinv_float16_mt.dtype.size}")
                return (rintmulinv_float16_mt), rintmulinv_float16_scope
                
def remainderhighprec_float16(ARG0, ARG1, c_1086914560, c_989700096, c_883040617):
        ARG0_st_m4 = ARG0.read(streams=g.SG2[2]) #direction=Direction.EASTWARD
        x_0_st_m3 = g.cast(ARG0_st_m4, dtype=g.float32, alus=[0], output_streams=[g.SG4[2]]) #input_streams=[g.SG4_E[1]], direction=Direction.EASTWARD
        ARG1_st_m4 = ARG1.read(streams=g.SG4[5]) #direction=Direction.EASTWARD
        ARG1_id6_st_m3 = g.vxm_identity(ARG1_st_m4, alus=[6], output_streams=[g.SG4[3]]) #input_streams=[g.SG4_E[5]], direction=Direction.EASTWARD
        c_1086914560_st_m4 = c_1086914560.read(streams=g.SG4[6]) #direction=Direction.WESTWARD
        x_1_st_m3 = g.mul(ARG1_st_m4, c_1086914560_st_m4, overflow_mode=g.OverflowMode.MODULAR, alus=[10], output_streams=[g.SG4[5]]) #input_streams=[g.SG4_E[5], g.SG4_W[6]], direction=Direction.WESTWARD
        x_2_st_m2 = g.sub(x_0_st_m3.reinterpret(g.float32), x_1_st_m3, overflow_mode=g.OverflowMode.MODULAR, alus=[5], output_streams=[g.SG4[2]]) #input_streams=[g.SG4_E[2], g.SG4_W[5]], direction=Direction.EASTWARD
        c_989700096_st_m3 = c_989700096.read(streams=g.SG4[3]) #direction=Direction.WESTWARD
        x_3_st_m2 = g.mul(ARG1_id6_st_m3.reinterpret(g.float32), c_989700096_st_m3, overflow_mode=g.OverflowMode.MODULAR, alus=[7], output_streams=[g.SG4[3]]) #input_streams=[g.SG4_E[3], g.SG4_W[3]], direction=Direction.WESTWARD
        x_4_st_m1 = g.sub(x_2_st_m2, x_3_st_m2, overflow_mode=g.OverflowMode.MODULAR, alus=[2], output_streams=[g.SG4[0]]) #input_streams=[g.SG4_E[2], g.SG4_W[3]], direction=Direction.WESTWARD
        c_883040617_st_m3 = c_883040617.read(streams=g.SG4[1]) #direction=Direction.WESTWARD
        x_5_st_m2 = g.mul(ARG1_id6_st_m3.reinterpret(g.float32), c_883040617_st_m3, overflow_mode=g.OverflowMode.MODULAR, alus=[3], output_streams=[g.SG4[1]]) #input_streams=[g.SG4_E[3], g.SG4_W[1]], direction=Direction.WESTWARD
        x_5_id14_st_m1 = g.vxm_identity(x_5_st_m2, alus=[14], output_streams=[g.SG4[1]]) #input_streams=[g.SG4_W[1]], direction=Direction.WESTWARD
        remainderhighprec_float16_st_p0 = g.sub(x_4_st_m1, x_5_id14_st_m1, overflow_mode=g.OverflowMode.MODULAR, time=0, alus=[1], output_streams=[g.SG4[0]]) #input_streams=[g.SG4_W[0], g.SG4_W[1]], direction=Direction.WESTWARD
        return remainderhighprec_float16_st_p0 #directions=[Direction.WESTWARD]

def remainderhighprec_float16_apply(ARG0, ARG1, predecessors=None):
        with g.ResourceScope(name="remainderhighprec_float16_scope", is_buffered=True, time=0 if predecessors is None else None, predecessors=predecessors) as remainderhighprec_float16_scope:
                c_989700096 = g.from_data(np.array([0.0019350051879882812]*320, dtype=np.float32), name='c_989700096', layout="-1, H1(E), S4")
                c_883040617 = g.from_data(np.array([3.019916050561733e-07]*320, dtype=np.float32), name='c_883040617', layout="-1, H1(E), S4")
                c_1086914560 = g.from_data(np.array([6.28125]*320, dtype=np.float32), name='c_1086914560', layout="-1, H1(E), S4")
                remainderhighprec_float16_mt = remainderhighprec_float16(ARG0, ARG1, c_1086914560, c_989700096, c_883040617)
                remainderhighprec_float16_mt = remainderhighprec_float16_mt.write(name="remainderhighprec_float16", layout=f"-1, H1(W), S{remainderhighprec_float16_mt.dtype.size}")
                return (remainderhighprec_float16_mt), remainderhighprec_float16_scope
                
def rintmulinv_float32(ARG0, c_1042479491, c_2139095040, c_1065353216):
        ARG0_st_m6 = ARG0.read(streams=g.SG4[1]) #direction=Direction.EASTWARD
        c_1042479491_st_m6 = c_1042479491.read(streams=g.SG4[1]) #direction=Direction.WESTWARD
        x_0_st_m5 = g.mul(ARG0_st_m6, c_1042479491_st_m6, overflow_mode=g.OverflowMode.MODULAR, alus=[14], output_streams=[g.SG4[1]]) #input_streams=[g.SG4_E[1], g.SG4_W[1]], direction=Direction.EASTWARD
        x_0_id15_st_m4 = g.vxm_identity(x_0_st_m5, alus=[15], output_streams=[g.SG4[7]]) #input_streams=[g.SG4_E[1]], direction=Direction.WESTWARD
        x_0_id10_st_m3 = g.vxm_identity(x_0_id15_st_m4, alus=[10], output_streams=[g.SG4[6]]) #input_streams=[g.SG4_W[7]], direction=Direction.WESTWARD
        x_0_id9_st_m2 = g.vxm_identity(x_0_id10_st_m3, alus=[9], output_streams=[g.SG4[4]]) #input_streams=[g.SG4_W[6]], direction=Direction.EASTWARD
        c_2139095040_st_m5 = c_2139095040.read(streams=g.SG4[0]) #direction=Direction.WESTWARD
        x_1_st_m4 = g.bitwise_and(x_0_st_m5.reinterpret(g.uint32), c_2139095040_st_m5, alus=[3], output_streams=[g.SG4[2]]) #input_streams=[g.SG4_E[1], g.SG4_W[0]], direction=Direction.WESTWARD
        x_2_st_m3 = g.cast(x_1_st_m4.reinterpret(g.float32), dtype=g.uint32, alus=[2], output_streams=[g.SG4[2]]) #input_streams=[g.SG4_W[2]], direction=Direction.EASTWARD
        c_1065353216_st_m3 = c_1065353216.read(streams=g.SG4[5]) #direction=Direction.WESTWARD
        x_3_st_m2 = g.min(x_2_st_m3.reinterpret(g.uint32), c_1065353216_st_m3, alus=[7], output_streams=[g.SG4[5]]) #input_streams=[g.SG4_E[2], g.SG4_W[5]], direction=Direction.WESTWARD
        x_4_st_m1 = g.mul(x_3_st_m2.reinterpret(g.float32), x_0_id9_st_m2, overflow_mode=g.OverflowMode.MODULAR, alus=[6], output_streams=[g.SG4[2]]) #input_streams=[g.SG4_W[5], g.SG4_E[4]], direction=Direction.WESTWARD
        x_5_st_m2 = g.cast(x_0_id10_st_m3, dtype=g.int32, alus=[13], output_streams=[g.SG4[6]]) #input_streams=[g.SG4_W[6]], direction=Direction.WESTWARD
        x_6_st_m1 = g.cast(x_5_st_m2.reinterpret(g.int32), dtype=g.float32, alus=[8], output_streams=[g.SG4[4]]) #input_streams=[g.SG4_W[6]], direction=Direction.EASTWARD
        rintmulinv_float32_st_p0 = g.max(x_4_st_m1.reinterpret(g.uint32), x_6_st_m1.reinterpret(g.uint32), time=0, alus=[5], output_streams=[g.SG4[2]]) #input_streams=[g.SG4_W[2], g.SG4_E[4]], direction=Direction.WESTWARD
        return rintmulinv_float32_st_p0 #directions=[Direction.WESTWARD]

def rintmulinv_float32_apply(ARG0, predecessors=None):
        with g.ResourceScope(name="rintmulinv_float32_scope", is_buffered=True, time=0 if predecessors is None else None, predecessors=predecessors) as rintmulinv_float32_scope:
                c_1065353216 = g.from_data(np.array([1065353216]*320, dtype=np.uint32), name='c_1065353216', layout="-1, H1(E), S4")
                c_2139095040 = g.from_data(np.array([2139095040]*320, dtype=np.uint32), name='c_2139095040', layout="-1, H1(E), S4")
                c_1042479491 = g.from_data(np.array([0.15915493667125702]*320, dtype=np.float32), name='c_1042479491', layout="-1, H1(E), S4")
                rintmulinv_float32_mt = rintmulinv_float32(ARG0, c_1042479491, c_2139095040, c_1065353216)
                rintmulinv_float32_mt = rintmulinv_float32_mt.write(name="rintmulinv_float32", layout=f"-1, H1(W), S{rintmulinv_float32_mt.dtype.size}")
                return (rintmulinv_float32_mt), rintmulinv_float32_scope
                
def remainderhighprec_float32(ARG0, ARG1, c_1086914560, c_989700096, c_883040617):
        ARG0_st_m3 = ARG0.read(streams=g.SG4[5]) #direction=Direction.EASTWARD
        ARG1_st_m4 = ARG1.read(streams=g.SG4[1]) #direction=Direction.EASTWARD
        ARG1_id14_st_m3 = g.vxm_identity(ARG1_st_m4, alus=[14], output_streams=[g.SG4[7]]) #input_streams=[g.SG4_E[1]], direction=Direction.EASTWARD
        c_1086914560_st_m4 = c_1086914560.read(streams=g.SG4[1]) #direction=Direction.WESTWARD
        x_0_st_m3 = g.mul(ARG1_st_m4, c_1086914560_st_m4, overflow_mode=g.OverflowMode.MODULAR, alus=[2], output_streams=[g.SG4[2]]) #input_streams=[g.SG4_E[1], g.SG4_W[1]], direction=Direction.WESTWARD
        x_1_st_m2 = g.sub(ARG0_st_m3, x_0_st_m3, overflow_mode=g.OverflowMode.MODULAR, alus=[5], output_streams=[g.SG4[2]]) #input_streams=[g.SG4_E[5], g.SG4_W[2]], direction=Direction.EASTWARD
        c_989700096_st_m3 = c_989700096.read(streams=g.SG4[4]) #direction=Direction.WESTWARD
        x_2_st_m2 = g.mul(ARG1_id14_st_m3.reinterpret(g.float32), c_989700096_st_m3, overflow_mode=g.OverflowMode.MODULAR, alus=[11], output_streams=[g.SG4[4]]) #input_streams=[g.SG4_E[7], g.SG4_W[4]], direction=Direction.WESTWARD
        x_3_st_m1 = g.sub(x_1_st_m2, x_2_st_m2, overflow_mode=g.OverflowMode.MODULAR, alus=[6], output_streams=[g.SG4[5]]) #input_streams=[g.SG4_E[2], g.SG4_W[4]], direction=Direction.WESTWARD
        c_883040617_st_m3 = c_883040617.read(streams=g.SG4[0]) #direction=Direction.WESTWARD
        x_4_st_m2 = g.mul(ARG1_id14_st_m3.reinterpret(g.float32), c_883040617_st_m3, overflow_mode=g.OverflowMode.MODULAR, alus=[15], output_streams=[g.SG4[6]]) #input_streams=[g.SG4_E[7], g.SG4_W[0]], direction=Direction.WESTWARD
        x_4_id10_st_m1 = g.vxm_identity(x_4_st_m2, alus=[10], output_streams=[g.SG4[4]]) #input_streams=[g.SG4_W[6]], direction=Direction.WESTWARD
        remainderhighprec_float32_st_p0 = g.sub(x_3_st_m1, x_4_id10_st_m1, overflow_mode=g.OverflowMode.MODULAR, time=0, alus=[9], output_streams=[g.SG4[6]]) #input_streams=[g.SG4_W[5], g.SG4_W[4]], direction=Direction.WESTWARD
        return remainderhighprec_float32_st_p0 #directions=[Direction.WESTWARD]

def remainderhighprec_float32_apply(ARG0, ARG1, predecessors=None):
        with g.ResourceScope(name="remainderhighprec_float32_scope", is_buffered=True, time=0 if predecessors is None else None, predecessors=predecessors) as remainderhighprec_float32_scope:
                c_989700096 = g.from_data(np.array([0.0019350051879882812]*320, dtype=np.float32), name='c_989700096', layout="-1, H1(E), S4")
                c_883040617 = g.from_data(np.array([3.019916050561733e-07]*320, dtype=np.float32), name='c_883040617', layout="-1, H1(E), S4")
                c_1086914560 = g.from_data(np.array([6.28125]*320, dtype=np.float32), name='c_1086914560', layout="-1, H1(E), S4")
                remainderhighprec_float32_mt = remainderhighprec_float32(ARG0, ARG1, c_1086914560, c_989700096, c_883040617)
                remainderhighprec_float32_mt = remainderhighprec_float32_mt.write(name="remainderhighprec_float32", layout=f"-1, H1(W), S{remainderhighprec_float32_mt.dtype.size}")
                return (remainderhighprec_float32_mt), remainderhighprec_float32_scope

def cos_quads_float16(ARG0, c_2147483647, c_1078530011, c_1086918619, c_1070141403):
        ARG0_st_m6 = ARG0.read(streams=g.SG2[0]) #direction=Direction.EASTWARD
        x_0_st_m5 = g.cast(ARG0_st_m6, dtype=g.float32, alus=[0], output_streams=[g.SG4[1]]) #input_streams=[g.SG4_E[0]], direction=Direction.EASTWARD
        c_2147483647_st_m5 = c_2147483647.read(streams=g.SG4[0]) #direction=Direction.WESTWARD
        c_2147483647_st_m3 = c_2147483647.read(streams=g.SG4[0]) #direction=Direction.WESTWARD
        x_1_st_m4 = g.bitwise_and(x_0_st_m5.reinterpret(g.uint32), c_2147483647_st_m5.reinterpret(g.uint32), alus=[3], output_streams=[g.SG4[0]]) #input_streams=[g.SG4_E[1], g.SG4_W[0]], direction=Direction.WESTWARD
        x_1_id2_st_m3 = g.vxm_identity(x_1_st_m4, alus=[2], output_streams=[g.SG4[3]]) #input_streams=[g.SG4_W[0]], direction=Direction.EASTWARD
        x_1_id7_st_m2 = g.vxm_identity(x_1_id2_st_m3, alus=[7], output_streams=[g.SG4[5]]) #input_streams=[g.SG4_E[3]], direction=Direction.WESTWARD
        c_1078530011_st_m4 = c_1078530011.read(streams=g.SG4[6]) #direction=Direction.WESTWARD
        x_2_st_m3 = g.sub(c_1078530011_st_m4, x_1_st_m4.reinterpret(g.float32), overflow_mode=g.OverflowMode.MODULAR, alus=[14], output_streams=[g.SG4[7]]) #input_streams=[g.SG4_W[6], g.SG4_W[0]], direction=Direction.EASTWARD
        x_3_st_m2 = g.bitwise_and(x_2_st_m3.reinterpret(g.uint32), c_2147483647_st_m3.reinterpret(g.uint32), alus=[15], output_streams=[g.SG4[7]]) #input_streams=[g.SG4_E[7], g.SG4_W[0]], direction=Direction.WESTWARD
        x_4_st_m1 = g.min(x_1_id7_st_m2.reinterpret(g.float32), x_3_st_m2.reinterpret(g.float32), alus=[10], output_streams=[g.SG4[5]]) #input_streams=[g.SG4_W[5], g.SG4_W[7]], direction=Direction.WESTWARD
        c_1086918619_st_m2 = c_1086918619.read(streams=g.SG4[3]) #direction=Direction.WESTWARD
        x_5_st_m1 = g.sub(c_1086918619_st_m2, x_1_id7_st_m2.reinterpret(g.float32), overflow_mode=g.OverflowMode.MODULAR, alus=[6], output_streams=[g.SG4[2]]) #input_streams=[g.SG4_W[3], g.SG4_W[5]], direction=Direction.WESTWARD
        cosadj_st_p0 = g.min(x_4_st_m1, x_5_st_m1, time=0, alus=[5], output_streams=[g.SG4[5]]) #input_streams=[g.SG4_W[5], g.SG4_W[2]], direction=Direction.WESTWARD
        c_1070141403_st_m2 = c_1070141403.read(streams=g.SG4[1]) #direction=Direction.WESTWARD
        cosneedneg_st_m1 = g.less(x_3_st_m2.reinterpret(g.float32), c_1070141403_st_m2, alus=[13], output_streams=[g.SG4[6]]) #input_streams=[g.SG4_W[7], g.SG4_W[1]], direction=Direction.WESTWARD
        return cosadj_st_p0, cosneedneg_st_m1 #directions=[Direction.WESTWARD, Direction.WESTWARD]

def cos_quads_float16_apply(ARG0, predecessors=None):
        with g.ResourceScope(name="cos_quads_float16_scope", is_buffered=True, time=0 if predecessors is None else None, predecessors=predecessors) as cos_quads_float16_scope:
                c_1086918619 = g.from_data(np.array([6.2831854820251465]*320, dtype=np.float32), name='c_1086918619', layout="-1, H1(E), S4")
                c_2147483647 = g.from_data(np.array([2147483647]*320, dtype=np.uint32).view(np.float32), name='c_2147483647', layout="-1, H1(E), S4")
                c_1078530011 = g.from_data(np.array([3.1415927410125732]*320, dtype=np.float32), name='c_1078530011', layout="-1, H1(E), S4")
                c_1070141403 = g.from_data(np.array([1.5707963705062866]*320, dtype=np.float32), name='c_1070141403', layout="-1, H1(E), S4")
                cosadj_mt, cosneedneg_mt = cos_quads_float16(ARG0, c_2147483647, c_1078530011, c_1086918619, c_1070141403)
                cosadj_mt = cosadj_mt.write(name="cosadj", layout=f"-1, H1(W), S{cosadj_mt.dtype.size}")
                cosneedneg_mt = cosneedneg_mt.write(name="cosneedneg", layout=f"-1, H1(W), S{cosneedneg_mt.dtype.size}")
                return (cosadj_mt, cosneedneg_mt), cos_quads_float16_scope
                
def sin_quads_float16(c_1070141403, ARG0, c_2147483647, c_1083624420, c_2147483648, c_1078530011):
        c_1070141403_st_m3 = c_1070141403.read(streams=g.SG4[4]) #direction=Direction.WESTWARD
        ARG0_st_m7 = ARG0.read(streams=g.SG2[4]) #direction=Direction.EASTWARD
        x_0_st_m6 = g.cast(ARG0_st_m7, dtype=g.float32, alus=[0], output_streams=[g.SG4[3]]) #input_streams=[g.SG4_E[2]], direction=Direction.EASTWARD
        x_0_id1_st_m5 = g.vxm_identity(x_0_st_m6, alus=[1], output_streams=[g.SG4[2]]) #input_streams=[g.SG4_E[3]], direction=Direction.WESTWARD
        x_0_id4_st_m4 = g.vxm_identity(x_0_id1_st_m5, alus=[4], output_streams=[g.SG4[4]]) #input_streams=[g.SG4_W[2]], direction=Direction.EASTWARD
        c_2147483647_st_m4 = c_2147483647.read(streams=g.SG4[6]) #direction=Direction.WESTWARD
        c_2147483647_st_m2 = c_2147483647.read(streams=g.SG4[6]) #direction=Direction.WESTWARD
        x_1_st_m3 = g.bitwise_and(x_0_id4_st_m4.reinterpret(g.uint32), c_2147483647_st_m4.reinterpret(g.uint32), alus=[9], output_streams=[g.SG4[4]]) #input_streams=[g.SG4_E[4], g.SG4_W[6]], direction=Direction.EASTWARD
        x_2_st_m2 = g.sub(c_1070141403_st_m3, x_1_st_m3.reinterpret(g.float32), overflow_mode=g.OverflowMode.MODULAR, alus=[11], output_streams=[g.SG4[7]]) #input_streams=[g.SG4_W[4], g.SG4_E[4]], direction=Direction.WESTWARD
        x_3_st_m1 = g.bitwise_and(x_2_st_m2.reinterpret(g.uint32), c_2147483647_st_m2.reinterpret(g.uint32), alus=[14], output_streams=[g.SG4[0]]) #input_streams=[g.SG4_W[7], g.SG4_W[6]], direction=Direction.WESTWARD
        c_1083624420_st_m3 = c_1083624420.read(streams=g.SG4[5]) #direction=Direction.WESTWARD
        x_4_st_m2 = g.sub(c_1083624420_st_m3, x_1_st_m3.reinterpret(g.float32), overflow_mode=g.OverflowMode.MODULAR, alus=[7], output_streams=[g.SG4[4]]) #input_streams=[g.SG4_W[5], g.SG4_E[4]], direction=Direction.WESTWARD
        x_5_st_m1 = g.bitwise_and(x_4_st_m2.reinterpret(g.uint32), c_2147483647_st_m2.reinterpret(g.uint32), alus=[10], output_streams=[g.SG4[7]]) #input_streams=[g.SG4_W[4], g.SG4_W[6]], direction=Direction.WESTWARD
        sinadj_st_p0 = g.min(x_3_st_m1.reinterpret(g.float32), x_5_st_m1.reinterpret(g.float32), time=0, alus=[13], output_streams=[g.SG4[7]]) #input_streams=[g.SG4_W[0], g.SG4_W[7]], direction=Direction.WESTWARD
        c_2147483648_st_m6 = c_2147483648.read(streams=g.SG4[2]) #direction=Direction.WESTWARD
        x_7_st_m5 = g.bitwise_and(x_0_st_m6.reinterpret(g.uint32), c_2147483648_st_m6, alus=[2], output_streams=[g.SG4[2]]) #input_streams=[g.SG4_E[3], g.SG4_W[2]], direction=Direction.EASTWARD
        c_1078530011_st_m5 = c_1078530011.read(streams=g.SG4[0]) #direction=Direction.WESTWARD
        x_8_st_m4 = g.bitwise_or(x_7_st_m5, c_1078530011_st_m5, alus=[3], output_streams=[g.SG4[3]]) #input_streams=[g.SG4_E[2], g.SG4_W[0]], direction=Direction.WESTWARD
        sinneedneg_st_m3 = g.greater(x_0_id4_st_m4.reinterpret(g.float32), x_8_st_m4.reinterpret(g.float32), alus=[5], output_streams=[g.SG4[5]]) #input_streams=[g.SG4_E[4], g.SG4_W[3]], direction=Direction.WESTWARD
        return sinadj_st_p0, sinneedneg_st_m3 #directions=[Direction.WESTWARD, Direction.WESTWARD]

def sin_quads_float16_apply(ARG0, predecessors=None):
        with g.ResourceScope(name="sin_quads_float16_scope", is_buffered=True, time=0 if predecessors is None else None, predecessors=predecessors) as sin_quads_float16_scope:
                c_2147483647 = g.from_data(np.array([2147483647]*320, dtype=np.uint32).view(np.float32), name='c_2147483647', layout="-1, H1(E), S4")
                c_1083624420 = g.from_data(np.array([4.71238899230957]*320, dtype=np.float32), name='c_1083624420', layout="-1, H1(E), S4")
                c_1070141403 = g.from_data(np.array([1.5707963705062866]*320, dtype=np.float32), name='c_1070141403', layout="-1, H1(E), S4")
                c_1078530011 = g.from_data(np.array([1078530011]*320, dtype=np.uint32), name='c_1078530011', layout="-1, H1(E), S4")
                c_2147483648 = g.from_data(np.array([2147483648]*320, dtype=np.uint32), name='c_2147483648', layout="-1, H1(E), S4")
                sinadj_mt, sinneedneg_mt = sin_quads_float16(c_1070141403, ARG0, c_2147483647, c_1083624420, c_2147483648, c_1078530011)
                sinadj_mt = sinadj_mt.write(name="sinadj", layout=f"-1, H1(W), S{sinadj_mt.dtype.size}")
                sinneedneg_mt = sinneedneg_mt.write(name="sinneedneg", layout=f"-1, H1(W), S{sinneedneg_mt.dtype.size}")
                return (sinadj_mt, sinneedneg_mt), sin_quads_float16_scope

def sincos_quads_float16(ARG0, c_2147483647, c_1078530011, c_1086918619, c_1070141403, c_2147483648):
        ARG0_st_m7 = ARG0.read(streams=g.SG2[2]) #direction=Direction.EASTWARD
        x_0_st_m6 = g.cast(ARG0_st_m7, dtype=g.float32, alus=[0], output_streams=[g.SG4[2]]) #input_streams=[g.SG4_E[1]], direction=Direction.EASTWARD
        x_0_id7_st_m5 = g.vxm_identity(x_0_st_m6, alus=[7], output_streams=[g.SG4[5]]) #input_streams=[g.SG4_E[2]], direction=Direction.WESTWARD
        x_0_id4_st_m4 = g.vxm_identity(x_0_id7_st_m5, alus=[4], output_streams=[g.SG4[4]]) #input_streams=[g.SG4_W[5]], direction=Direction.EASTWARD
        c_2147483647_st_m5 = c_2147483647.read(streams=g.SG4[4]) #direction=Direction.WESTWARD
        c_2147483647_st_m3 = c_2147483647.read(streams=g.SG4[4]) #direction=Direction.WESTWARD
        x_1_st_m4 = g.bitwise_and(x_0_id7_st_m5.reinterpret(g.uint32), c_2147483647_st_m5.reinterpret(g.uint32), alus=[8], output_streams=[g.SG4[5]]) #input_streams=[g.SG4_W[5], g.SG4_W[4]], direction=Direction.EASTWARD
        x_1_id9_st_m3 = g.vxm_identity(x_1_st_m4, alus=[9], output_streams=[g.SG4[6]]) #input_streams=[g.SG4_E[5]], direction=Direction.WESTWARD
        x_1_id12_st_m2 = g.vxm_identity(x_1_id9_st_m3, alus=[12], output_streams=[g.SG4[6]]) #input_streams=[g.SG4_W[6]], direction=Direction.EASTWARD
        c_1078530011_st_m5 = c_1078530011.read(streams=g.SG4[3]) #direction=Direction.WESTWARD
        c_1078530011_st_m4 = c_1078530011.read(streams=g.SG4[3]) #direction=Direction.WESTWARD
        x_2_st_m3 = g.sub(c_1078530011_st_m4.reinterpret(g.float32), x_1_st_m4.reinterpret(g.float32), overflow_mode=g.OverflowMode.MODULAR, alus=[6], output_streams=[g.SG4[5]]) #input_streams=[g.SG4_W[3], g.SG4_E[5]], direction=Direction.EASTWARD
        x_3_st_m2 = g.bitwise_and(x_2_st_m3.reinterpret(g.uint32), c_2147483647_st_m3.reinterpret(g.uint32), alus=[11], output_streams=[g.SG4[6]]) #input_streams=[g.SG4_E[5], g.SG4_W[4]], direction=Direction.WESTWARD
        x_4_st_m1 = g.min(x_1_id12_st_m2.reinterpret(g.float32), x_3_st_m2.reinterpret(g.float32), alus=[10], output_streams=[g.SG4[7]]) #input_streams=[g.SG4_E[6], g.SG4_W[6]], direction=Direction.EASTWARD
        c_1086918619_st_m2 = c_1086918619.read(streams=g.SG4[7]) #direction=Direction.WESTWARD
        x_5_st_m1 = g.sub(c_1086918619_st_m2, x_1_id12_st_m2.reinterpret(g.float32), overflow_mode=g.OverflowMode.MODULAR, alus=[14], output_streams=[g.SG4[1]]) #input_streams=[g.SG4_W[7], g.SG4_E[6]], direction=Direction.EASTWARD
        cosadj_st_p0 = g.min(x_4_st_m1, x_5_st_m1, time=0, alus=[15], output_streams=[g.SG4[0]]) #input_streams=[g.SG4_E[7], g.SG4_E[1]], direction=Direction.WESTWARD
        c_1070141403_st_p0 = c_1070141403.read(streams=g.SG4[1]) #direction=Direction.WESTWARD
        c_1070141403_st_m2 = c_1070141403.read(streams=g.SG4[1]) #direction=Direction.WESTWARD
        cosneedneg_st_m1 = g.less(x_3_st_m2.reinterpret(g.float32), c_1070141403_st_m2, alus=[13], output_streams=[g.SG4[7]]) #input_streams=[g.SG4_W[6], g.SG4_W[1]], direction=Direction.WESTWARD
        sinadj_st_p1 = g.sub(c_1070141403_st_p0, cosadj_st_p0, overflow_mode=g.OverflowMode.MODULAR, alus=[1], output_streams=[g.SG4[3]]) #input_streams=[g.SG4_W[1], g.SG4_W[0]], direction=Direction.WESTWARD
        c_2147483648_st_m6 = c_2147483648.read(streams=g.SG4[0]) #direction=Direction.WESTWARD
        x_9_st_m5 = g.bitwise_and(x_0_st_m6.reinterpret(g.uint32), c_2147483648_st_m6, alus=[3], output_streams=[g.SG4[2]]) #input_streams=[g.SG4_E[2], g.SG4_W[0]], direction=Direction.WESTWARD
        x_10_st_m4 = g.bitwise_or(x_9_st_m5, c_1078530011_st_m5, alus=[2], output_streams=[g.SG4[3]]) #input_streams=[g.SG4_W[2], g.SG4_W[3]], direction=Direction.WESTWARD
        sinneedneg_st_m3 = g.greater(x_0_id4_st_m4.reinterpret(g.float32), x_10_st_m4.reinterpret(g.float32), alus=[5], output_streams=[g.SG4[2]]) #input_streams=[g.SG4_E[4], g.SG4_W[3]], direction=Direction.WESTWARD
        return cosadj_st_p0, cosneedneg_st_m1, sinadj_st_p1, sinneedneg_st_m3 #directions=[Direction.WESTWARD, Direction.WESTWARD, Direction.WESTWARD, Direction.WESTWARD]

def sincos_quads_float16_apply(ARG0, predecessors=None):
        with g.ResourceScope(name="sincos_quads_float16_scope", is_buffered=True, time=0 if predecessors is None else None, predecessors=predecessors) as sincos_quads_float16_scope:
                c_1086918619 = g.from_data(np.array([6.2831854820251465]*320, dtype=np.float32), name='c_1086918619', layout="-1, H1(E), S4")
                c_2147483647 = g.from_data(np.array([2147483647]*320, dtype=np.uint32).view(np.float32), name='c_2147483647', layout="-1, H1(E), S4")
                c_1070141403 = g.from_data(np.array([1.5707963705062866]*320, dtype=np.float32), name='c_1070141403', layout="-1, H1(E), S4")
                c_1078530011 = g.from_data(np.array([1078530011]*320, dtype=np.uint32), name='c_1078530011', layout="-1, H1(E), S4")
                c_2147483648 = g.from_data(np.array([2147483648]*320, dtype=np.uint32), name='c_2147483648', layout="-1, H1(E), S4")
                cosadj_mt, cosneedneg_mt, sinadj_mt, sinneedneg_mt = sincos_quads_float16(ARG0, c_2147483647, c_1078530011, c_1086918619, c_1070141403, c_2147483648)
                cosadj_mt = cosadj_mt.write(name="cosadj", layout=f"-1, H1(W), S{cosadj_mt.dtype.size}")
                cosneedneg_mt = cosneedneg_mt.write(name="cosneedneg", layout=f"-1, H1(W), S{cosneedneg_mt.dtype.size}")
                sinadj_mt = sinadj_mt.write(name="sinadj", layout=f"-1, H1(W), S{sinadj_mt.dtype.size}")
                sinneedneg_mt = sinneedneg_mt.write(name="sinneedneg", layout=f"-1, H1(W), S{sinneedneg_mt.dtype.size}")
                return (cosadj_mt, cosneedneg_mt, sinadj_mt, sinneedneg_mt), sincos_quads_float16_scope

def cos_quads_float32(ARG0, c_2147483647, c_1078530011, c_1086918619, c_1070141403):
        ARG0_st_m5 = ARG0.read(streams=g.SG4[0]) #direction=Direction.EASTWARD
        c_2147483647_st_m5 = c_2147483647.read(streams=g.SG4[6]) #direction=Direction.WESTWARD
        c_2147483647_st_m3 = c_2147483647.read(streams=g.SG4[6]) #direction=Direction.WESTWARD
        x_0_st_m4 = g.bitwise_and(ARG0_st_m5.reinterpret(g.uint32), c_2147483647_st_m5.reinterpret(g.uint32), alus=[15], output_streams=[g.SG4[1]]) #input_streams=[g.SG4_E[0], g.SG4_W[6]], direction=Direction.WESTWARD
        x_0_id2_st_m3 = g.vxm_identity(x_0_st_m4, alus=[2], output_streams=[g.SG4[3]]) #input_streams=[g.SG4_W[1]], direction=Direction.EASTWARD
        x_0_id7_st_m2 = g.vxm_identity(x_0_id2_st_m3, alus=[7], output_streams=[g.SG4[5]]) #input_streams=[g.SG4_E[3]], direction=Direction.WESTWARD
        c_1078530011_st_m4 = c_1078530011.read(streams=g.SG4[7]) #direction=Direction.WESTWARD
        x_1_st_m3 = g.sub(c_1078530011_st_m4, x_0_st_m4.reinterpret(g.float32), overflow_mode=g.OverflowMode.MODULAR, alus=[14], output_streams=[g.SG4[6]]) #input_streams=[g.SG4_W[7], g.SG4_W[1]], direction=Direction.EASTWARD
        x_2_st_m2 = g.bitwise_and(x_1_st_m3.reinterpret(g.uint32), c_2147483647_st_m3.reinterpret(g.uint32), alus=[11], output_streams=[g.SG4[6]]) #input_streams=[g.SG4_E[6], g.SG4_W[6]], direction=Direction.WESTWARD
        x_3_st_m1 = g.min(x_0_id7_st_m2.reinterpret(g.float32), x_2_st_m2.reinterpret(g.float32), alus=[10], output_streams=[g.SG4[5]]) #input_streams=[g.SG4_W[5], g.SG4_W[6]], direction=Direction.WESTWARD
        c_1086918619_st_m2 = c_1086918619.read(streams=g.SG4[3]) #direction=Direction.WESTWARD
        x_4_st_m1 = g.sub(c_1086918619_st_m2, x_0_id7_st_m2.reinterpret(g.float32), overflow_mode=g.OverflowMode.MODULAR, alus=[6], output_streams=[g.SG4[4]]) #input_streams=[g.SG4_W[3], g.SG4_W[5]], direction=Direction.WESTWARD
        cosadj_st_p0 = g.min(x_3_st_m1, x_4_st_m1, time=0, alus=[5], output_streams=[g.SG4[2]]) #input_streams=[g.SG4_W[5], g.SG4_W[4]], direction=Direction.WESTWARD
        c_1070141403_st_m2 = c_1070141403.read(streams=g.SG4[0]) #direction=Direction.WESTWARD
        cosneedneg_st_m1 = g.less(x_2_st_m2.reinterpret(g.float32), c_1070141403_st_m2, alus=[13], output_streams=[g.SG4[7]]) #input_streams=[g.SG4_W[6], g.SG4_W[0]], direction=Direction.WESTWARD
        return cosadj_st_p0, cosneedneg_st_m1 #directions=[Direction.WESTWARD, Direction.WESTWARD]

def cos_quads_float32_apply(ARG0, predecessors=None):
        with g.ResourceScope(name="cos_quads_float32_scope", is_buffered=True, time=0 if predecessors is None else None, predecessors=predecessors) as cos_quads_float32_scope:
                c_2147483647 = g.from_data(np.array([2147483647]*320, dtype=np.uint32).view(np.float32), name='c_2147483647', layout="-1, H1(E), S4")
                c_1086918619 = g.from_data(np.array([6.2831854820251465]*320, dtype=np.float32), name='c_1086918619', layout="-1, H1(E), S4")
                c_1078530011 = g.from_data(np.array([3.1415927410125732]*320, dtype=np.float32), name='c_1078530011', layout="-1, H1(E), S4")
                c_1070141403 = g.from_data(np.array([1.5707963705062866]*320, dtype=np.float32), name='c_1070141403', layout="-1, H1(E), S4")
                cosadj_mt, cosneedneg_mt = cos_quads_float32(ARG0, c_2147483647, c_1078530011, c_1086918619, c_1070141403)
                cosadj_mt = cosadj_mt.write(name="cosadj", layout=f"-1, H1(W), S{cosadj_mt.dtype.size}")
                cosneedneg_mt = cosneedneg_mt.write(name="cosneedneg", layout=f"-1, H1(W), S{cosneedneg_mt.dtype.size}")
                return (cosadj_mt, cosneedneg_mt), cos_quads_float32_scope
                
def sin_quads_float32(c_1070141403, ARG0, c_2147483647, c_1083624420, c_2147483648, c_1078530011):
        c_1070141403_st_m3 = c_1070141403.read(streams=g.SG4[3]) #direction=Direction.WESTWARD
        ARG0_st_m5 = ARG0.read(streams=g.SG4[0]) #direction=Direction.EASTWARD
        ARG0_id0_st_m4 = g.vxm_identity(ARG0_st_m5, alus=[0], output_streams=[g.SG4[1]]) #input_streams=[g.SG4_E[0]], direction=Direction.EASTWARD
        ARG0_id13_st_m3 = g.vxm_identity(ARG0_id0_st_m4, alus=[13], output_streams=[g.SG4[6]]) #input_streams=[g.SG4_E[1]], direction=Direction.EASTWARD
        c_2147483647_st_m4 = c_2147483647.read(streams=g.SG4[2]) #direction=Direction.WESTWARD
        c_2147483647_st_m2 = c_2147483647.read(streams=g.SG4[2]) #direction=Direction.WESTWARD
        x_0_st_m3 = g.bitwise_and(ARG0_id0_st_m4.reinterpret(g.uint32), c_2147483647_st_m4.reinterpret(g.uint32), alus=[1], output_streams=[g.SG4[2]]) #input_streams=[g.SG4_E[1], g.SG4_W[2]], direction=Direction.EASTWARD
        x_1_st_m2 = g.sub(c_1070141403_st_m3, x_0_st_m3.reinterpret(g.float32), overflow_mode=g.OverflowMode.MODULAR, alus=[3], output_streams=[g.SG4[0]]) #input_streams=[g.SG4_W[3], g.SG4_E[2]], direction=Direction.WESTWARD
        x_2_st_m1 = g.bitwise_and(x_1_st_m2.reinterpret(g.uint32), c_2147483647_st_m2.reinterpret(g.uint32), alus=[2], output_streams=[g.SG4[3]]) #input_streams=[g.SG4_W[0], g.SG4_W[2]], direction=Direction.WESTWARD
        c_1083624420_st_m3 = c_1083624420.read(streams=g.SG4[5]) #direction=Direction.WESTWARD
        x_3_st_m2 = g.sub(c_1083624420_st_m3, x_0_st_m3.reinterpret(g.float32), overflow_mode=g.OverflowMode.MODULAR, alus=[7], output_streams=[g.SG4[4]]) #input_streams=[g.SG4_W[5], g.SG4_E[2]], direction=Direction.WESTWARD
        x_4_st_m1 = g.bitwise_and(x_3_st_m2.reinterpret(g.uint32), c_2147483647_st_m2.reinterpret(g.uint32), alus=[6], output_streams=[g.SG4[5]]) #input_streams=[g.SG4_W[4], g.SG4_W[2]], direction=Direction.WESTWARD
        sinadj_st_p0 = g.min(x_2_st_m1.reinterpret(g.float32), x_4_st_m1.reinterpret(g.float32), time=0, alus=[5], output_streams=[g.SG4[2]]) #input_streams=[g.SG4_W[3], g.SG4_W[5]], direction=Direction.WESTWARD
        c_2147483648_st_m5 = c_2147483648.read(streams=g.SG4[6]) #direction=Direction.WESTWARD
        x_6_st_m4 = g.bitwise_and(ARG0_st_m5.reinterpret(g.uint32), c_2147483648_st_m5, alus=[14], output_streams=[g.SG4[1]]) #input_streams=[g.SG4_E[0], g.SG4_W[6]], direction=Direction.EASTWARD
        c_1078530011_st_m4 = c_1078530011.read(streams=g.SG4[0]) #direction=Direction.WESTWARD
        x_7_st_m3 = g.bitwise_or(x_6_st_m4, c_1078530011_st_m4, alus=[15], output_streams=[g.SG4[7]]) #input_streams=[g.SG4_E[1], g.SG4_W[0]], direction=Direction.WESTWARD
        sinneedneg_st_m2 = g.greater(ARG0_id13_st_m3.reinterpret(g.float32), x_7_st_m3.reinterpret(g.float32), alus=[10], output_streams=[g.SG4[4]]) #input_streams=[g.SG4_E[6], g.SG4_W[7]], direction=Direction.WESTWARD
        return sinadj_st_p0, sinneedneg_st_m2 #directions=[Direction.WESTWARD, Direction.WESTWARD]

def sin_quads_float32_apply(ARG0, predecessors=None):
        with g.ResourceScope(name="sin_quads_float32_scope", is_buffered=True, time=0 if predecessors is None else None, predecessors=predecessors) as sin_quads_float32_scope:
                c_1078530011 = g.from_data(np.array([1078530011]*320, dtype=np.uint32), name='c_1078530011', layout="-1, H1(E), S4")
                c_2147483647 = g.from_data(np.array([2147483647]*320, dtype=np.uint32).view(np.float32), name='c_2147483647', layout="-1, H1(E), S4")
                c_1083624420 = g.from_data(np.array([4.71238899230957]*320, dtype=np.float32), name='c_1083624420', layout="-1, H1(E), S4")
                c_1070141403 = g.from_data(np.array([1.5707963705062866]*320, dtype=np.float32), name='c_1070141403', layout="-1, H1(E), S4")
                c_2147483648 = g.from_data(np.array([2147483648]*320, dtype=np.uint32), name='c_2147483648', layout="-1, H1(E), S4")
                sinadj_mt, sinneedneg_mt = sin_quads_float32(c_1070141403, ARG0, c_2147483647, c_1083624420, c_2147483648, c_1078530011)
                sinadj_mt = sinadj_mt.write(name="sinadj", layout=f"-1, H1(W), S{sinadj_mt.dtype.size}")
                sinneedneg_mt = sinneedneg_mt.write(name="sinneedneg", layout=f"-1, H1(W), S{sinneedneg_mt.dtype.size}")
                return (sinadj_mt, sinneedneg_mt), sin_quads_float32_scope

def sincos_quads_float32(ARG0, c_2147483647, c_1078530011, c_1086918619, c_1070141403, c_2147483648):
        ARG0_st_m5 = ARG0.read(streams=g.SG4[2]) #direction=Direction.EASTWARD
        ARG0_id1_st_m4 = g.vxm_identity(ARG0_st_m5, alus=[1], output_streams=[g.SG4[3]]) #input_streams=[g.SG4_E[2]], direction=Direction.WESTWARD
        ARG0_id4_st_m3 = g.vxm_identity(ARG0_id1_st_m4, alus=[4], output_streams=[g.SG4[3]]) #input_streams=[g.SG4_W[3]], direction=Direction.EASTWARD
        c_2147483647_st_m5 = c_2147483647.read(streams=g.SG4[1]) #direction=Direction.WESTWARD
        c_2147483647_st_m3 = c_2147483647.read(streams=g.SG4[1]) #direction=Direction.WESTWARD
        x_0_st_m4 = g.bitwise_and(ARG0_st_m5.reinterpret(g.uint32), c_2147483647_st_m5.reinterpret(g.uint32), alus=[0], output_streams=[g.SG4[1]]) #input_streams=[g.SG4_E[2], g.SG4_W[1]], direction=Direction.EASTWARD
        x_0_id3_st_m3 = g.vxm_identity(x_0_st_m4, alus=[3], output_streams=[g.SG4[0]]) #input_streams=[g.SG4_E[1]], direction=Direction.WESTWARD
        x_0_id14_st_m2 = g.vxm_identity(x_0_id3_st_m3, alus=[14], output_streams=[g.SG4[7]]) #input_streams=[g.SG4_W[0]], direction=Direction.EASTWARD
        c_1078530011_st_m4 = c_1078530011.read(streams=g.SG4[2]) #direction=Direction.WESTWARD
        x_1_st_m3 = g.sub(c_1078530011_st_m4.reinterpret(g.float32), x_0_st_m4.reinterpret(g.float32), overflow_mode=g.OverflowMode.MODULAR, alus=[2], output_streams=[g.SG4[0]]) #input_streams=[g.SG4_W[2], g.SG4_E[1]], direction=Direction.WESTWARD
        x_2_st_m2 = g.bitwise_and(x_1_st_m3.reinterpret(g.uint32), c_2147483647_st_m3.reinterpret(g.uint32), alus=[12], output_streams=[g.SG4[0]]) #input_streams=[g.SG4_W[0], g.SG4_W[1]], direction=Direction.EASTWARD
        x_3_st_m1 = g.min(x_0_id14_st_m2.reinterpret(g.float32), x_2_st_m2.reinterpret(g.float32), alus=[15], output_streams=[g.SG4[6]]) #input_streams=[g.SG4_E[7], g.SG4_E[0]], direction=Direction.WESTWARD
        c_1086918619_st_m2 = c_1086918619.read(streams=g.SG4[6]) #direction=Direction.WESTWARD
        x_4_st_m1 = g.sub(c_1086918619_st_m2, x_0_id14_st_m2.reinterpret(g.float32), overflow_mode=g.OverflowMode.MODULAR, alus=[11], output_streams=[g.SG4[5]]) #input_streams=[g.SG4_W[6], g.SG4_E[7]], direction=Direction.WESTWARD
        cosadj_st_p0 = g.min(x_3_st_m1, x_4_st_m1, time=0, alus=[10], output_streams=[g.SG4[5]]) #input_streams=[g.SG4_W[6], g.SG4_W[5]], direction=Direction.WESTWARD
        c_1070141403_st_p0 = c_1070141403.read(streams=g.SG4[7]) #direction=Direction.WESTWARD
        c_1070141403_st_m2 = c_1070141403.read(streams=g.SG4[7]) #direction=Direction.WESTWARD
        cosneedneg_st_m1 = g.less(x_2_st_m2.reinterpret(g.float32), c_1070141403_st_m2, alus=[13], output_streams=[g.SG4[6]]) #input_streams=[g.SG4_E[0], g.SG4_W[7]], direction=Direction.WESTWARD
        sinadj_st_p1 = g.sub(c_1070141403_st_p0, cosadj_st_p0, overflow_mode=g.OverflowMode.MODULAR, alus=[9], output_streams=[g.SG4[7]]) #input_streams=[g.SG4_W[7], g.SG4_W[5]], direction=Direction.WESTWARD
        c_2147483648_st_m5 = c_2147483648.read(streams=g.SG4[4]) #direction=Direction.WESTWARD
        x_8_st_m4 = g.bitwise_and(ARG0_st_m5.reinterpret(g.uint32), c_2147483648_st_m5, alus=[7], output_streams=[g.SG4[3]]) #input_streams=[g.SG4_E[2], g.SG4_W[4]], direction=Direction.WESTWARD
        x_9_st_m3 = g.bitwise_or(x_8_st_m4, c_1078530011_st_m4, alus=[6], output_streams=[g.SG4[4]]) #input_streams=[g.SG4_W[3], g.SG4_W[2]], direction=Direction.WESTWARD
        sinneedneg_st_m2 = g.greater(ARG0_id4_st_m3.reinterpret(g.float32), x_9_st_m3.reinterpret(g.float32), alus=[5], output_streams=[g.SG4[2]]) #input_streams=[g.SG4_E[3], g.SG4_W[4]], direction=Direction.WESTWARD
        return cosadj_st_p0, cosneedneg_st_m1, sinadj_st_p1, sinneedneg_st_m2 #directions=[Direction.WESTWARD, Direction.WESTWARD, Direction.WESTWARD, Direction.WESTWARD]

def sincos_quads_float32_apply(ARG0, predecessors=None):
        with g.ResourceScope(name="sincos_quads_float32_scope", is_buffered=True, time=0 if predecessors is None else None, predecessors=predecessors) as sincos_quads_float32_scope:
                c_2147483648 = g.from_data(np.array([2147483648]*320, dtype=np.uint32), name='c_2147483648', layout="-1, H1(E), S4")
                c_1078530011 = g.from_data(np.array([1078530011]*320, dtype=np.uint32), name='c_1078530011', layout="-1, H1(E), S4")
                c_1070141403 = g.from_data(np.array([1.5707963705062866]*320, dtype=np.float32), name='c_1070141403', layout="-1, H1(E), S4")
                c_2147483647 = g.from_data(np.array([2147483647]*320, dtype=np.uint32).view(np.float32), name='c_2147483647', layout="-1, H1(E), S4")
                c_1086918619 = g.from_data(np.array([6.2831854820251465]*320, dtype=np.float32), name='c_1086918619', layout="-1, H1(E), S4")
                cosadj_mt, cosneedneg_mt, sinadj_mt, sinneedneg_mt = sincos_quads_float32(ARG0, c_2147483647, c_1078530011, c_1086918619, c_1070141403, c_2147483648)
                cosadj_mt = cosadj_mt.write(name="cosadj", layout=f"-1, H1(W), S{cosadj_mt.dtype.size}")
                cosneedneg_mt = cosneedneg_mt.write(name="cosneedneg", layout=f"-1, H1(W), S{cosneedneg_mt.dtype.size}")
                sinadj_mt = sinadj_mt.write(name="sinadj", layout=f"-1, H1(W), S{sinadj_mt.dtype.size}")
                sinneedneg_mt = sinneedneg_mt.write(name="sinneedneg", layout=f"-1, H1(W), S{sinneedneg_mt.dtype.size}")
                return (cosadj_mt, cosneedneg_mt, sinadj_mt, sinneedneg_mt), sincos_quads_float32_scope

def cos_poly_float32(c_1065353215, c_3204448223, ARG0, c_1026205562, c_3132463055, c_935488695, ARG1, c_31):
        c_1065353215_st_m5 = c_1065353215.read(streams=g.SG4[7]) #direction=Direction.WESTWARD
        c_3204448223_st_m6 = c_3204448223.read(streams=g.SG4[6]) #direction=Direction.WESTWARD
        ARG0_st_m7 = ARG0.read(streams=g.SG4[0]) #direction=Direction.EASTWARD
        x_0_st_m6 = g.mul(ARG0_st_m7, ARG0_st_m7, overflow_mode=g.OverflowMode.MODULAR, alus=[12], output_streams=[g.SG4[7]]) #input_streams=[g.SG4_E[0], g.SG4_E[0]], direction=Direction.EASTWARD
        x_0_id13_st_m5 = g.vxm_identity(x_0_st_m6, alus=[13], output_streams=[g.SG4[6]]) #input_streams=[g.SG4_E[7]], direction=Direction.WESTWARD
        x_1_st_m5 = g.mul(c_3204448223_st_m6, x_0_st_m6, overflow_mode=g.OverflowMode.MODULAR, alus=[15], output_streams=[g.SG4[0]]) #input_streams=[g.SG4_W[6], g.SG4_E[7]], direction=Direction.WESTWARD
        x_2_st_m4 = g.add(c_1065353215_st_m5, x_1_st_m5, overflow_mode=g.OverflowMode.MODULAR, alus=[14], output_streams=[g.SG4[7]]) #input_streams=[g.SG4_W[7], g.SG4_W[0]], direction=Direction.WESTWARD
        c_1026205562_st_m5 = c_1026205562.read(streams=g.SG4[5]) #direction=Direction.WESTWARD
        x_3_st_m5 = g.mul(x_0_st_m6, x_0_st_m6, overflow_mode=g.OverflowMode.MODULAR, alus=[11], output_streams=[g.SG4[4]]) #input_streams=[g.SG4_E[7], g.SG4_E[7]], direction=Direction.WESTWARD
        x_4_st_m4 = g.mul(c_1026205562_st_m5, x_3_st_m5, overflow_mode=g.OverflowMode.MODULAR, alus=[10], output_streams=[g.SG4[6]]) #input_streams=[g.SG4_W[5], g.SG4_W[4]], direction=Direction.WESTWARD
        x_5_st_m3 = g.add(x_2_st_m4, x_4_st_m4, overflow_mode=g.OverflowMode.MODULAR, alus=[9], output_streams=[g.SG4[4]]) #input_streams=[g.SG4_W[7], g.SG4_W[6]], direction=Direction.EASTWARD
        c_3132463055_st_m4 = c_3132463055.read(streams=g.SG4[4]) #direction=Direction.WESTWARD
        x_6_st_m4 = g.mul(x_3_st_m5, x_0_id13_st_m5, overflow_mode=g.OverflowMode.MODULAR, alus=[8], output_streams=[g.SG4[5]]) #input_streams=[g.SG4_W[4], g.SG4_W[6]], direction=Direction.EASTWARD
        x_7_st_m3 = g.mul(c_3132463055_st_m4, x_6_st_m4, overflow_mode=g.OverflowMode.MODULAR, alus=[7], output_streams=[g.SG4[2]]) #input_streams=[g.SG4_W[4], g.SG4_E[5]], direction=Direction.WESTWARD
        x_8_st_m2 = g.add(x_5_st_m3, x_7_st_m3, overflow_mode=g.OverflowMode.MODULAR, alus=[6], output_streams=[g.SG4[3]]) #input_streams=[g.SG4_E[4], g.SG4_W[2]], direction=Direction.WESTWARD
        c_935488695_st_m3 = c_935488695.read(streams=g.SG4[1]) #direction=Direction.WESTWARD
        x_9_st_m4 = g.mul(x_3_st_m5, x_3_st_m5, overflow_mode=g.OverflowMode.MODULAR, alus=[5], output_streams=[g.SG4[2]]) #input_streams=[g.SG4_W[4], g.SG4_W[4]], direction=Direction.WESTWARD
        x_9_id4_st_m3 = g.vxm_identity(x_9_st_m4, alus=[4], output_streams=[g.SG4[3]]) #input_streams=[g.SG4_W[2]], direction=Direction.EASTWARD
        x_10_st_m2 = g.mul(c_935488695_st_m3, x_9_id4_st_m3, overflow_mode=g.OverflowMode.MODULAR, alus=[3], output_streams=[g.SG4[1]]) #input_streams=[g.SG4_W[1], g.SG4_E[3]], direction=Direction.WESTWARD
        x_11_st_m1 = g.add(x_8_st_m2, x_10_st_m2, overflow_mode=g.OverflowMode.MODULAR, alus=[0], output_streams=[g.SG4[2]]) #input_streams=[g.SG4_W[3], g.SG4_W[1]], direction=Direction.EASTWARD
        ARG1_st_m2 = ARG1.read(streams=g.SG4[1]) #direction=Direction.EASTWARD
        c_31_st_m2 = c_31.read(streams=g.SG4[3]) #direction=Direction.WESTWARD
        x_12_st_m1 = g.left_shift(ARG1_st_m2.reinterpret(g.uint32), c_31_st_m2, alus=[2], output_streams=[g.SG4[0]]) #input_streams=[g.SG4_E[1], g.SG4_W[3]], direction=Direction.WESTWARD
        cos_st_p0 = g.bitwise_or(x_11_st_m1, x_12_st_m1.reinterpret(g.float32), time=0, alus=[1], output_streams=[g.SG4[0]]) #input_streams=[g.SG4_E[2], g.SG4_W[0]], direction=Direction.WESTWARD
        return cos_st_p0 #directions=[Direction.WESTWARD]

def cos_poly_float32_apply(ARG0, ARG1, predecessors=None):
        with g.ResourceScope(name="cos_poly_float32_scope", is_buffered=True, time=0 if predecessors is None else None, predecessors=predecessors) as cos_poly_float32_scope:
                c_3204448223 = g.from_data(np.array([-0.4999990165233612]*320, dtype=np.float32), name='c_3204448223', layout="-1, H1(E), S4")
                c_935488695 = g.from_data(np.array([2.3177226466941647e-05]*320, dtype=np.float32), name='c_935488695', layout="-1, H1(E), S4")
                c_1026205562 = g.from_data(np.array([0.041663624346256256]*320, dtype=np.float32), name='c_1026205562', layout="-1, H1(E), S4")
                c_3132463055 = g.from_data(np.array([-0.0013854446588084102]*320, dtype=np.float32), name='c_3132463055', layout="-1, H1(E), S4")
                c_31 = g.from_data(np.array([31]*320, dtype=np.uint32), name='c_31', layout="-1, H1(E), S4")
                c_1065353215 = g.from_data(np.array([0.9999999403953552]*320, dtype=np.float32), name='c_1065353215', layout="-1, H1(E), S4")
                cos_mt = cos_poly_float32(c_1065353215, c_3204448223, ARG0, c_1026205562, c_3132463055, c_935488695, ARG1, c_31)
                cos_mt = cos_mt.write(name="cos", layout=f"-1, H1(W), S{cos_mt.dtype.size}")
                return (cos_mt), cos_poly_float32_scope

def cos_poly_float16(c_1065353216, c_3204431872, ARG0, c_1026031616, c_3130286080, ARG1, c_31):
        c_1065353216_st_m5 = c_1065353216.read(streams=g.SG4[0]) #direction=Direction.WESTWARD
        c_3204431872_st_m6 = c_3204431872.read(streams=g.SG4[6]) #direction=Direction.WESTWARD
        ARG0_st_m7 = ARG0.read(streams=g.SG4[0]) #direction=Direction.EASTWARD
        x_0_st_m6 = g.mul(ARG0_st_m7, ARG0_st_m7, overflow_mode=g.OverflowMode.MODULAR, alus=[12], output_streams=[g.SG4[7]]) #input_streams=[g.SG4_E[0], g.SG4_E[0]], direction=Direction.EASTWARD
        x_0_id9_st_m5 = g.vxm_identity(x_0_st_m6, alus=[9], output_streams=[g.SG4[6]]) #input_streams=[g.SG4_E[7]], direction=Direction.EASTWARD
        x_1_st_m5 = g.mul(c_3204431872_st_m6, x_0_st_m6, overflow_mode=g.OverflowMode.MODULAR, alus=[14], output_streams=[g.SG4[1]]) #input_streams=[g.SG4_W[6], g.SG4_E[7]], direction=Direction.EASTWARD
        x_2_st_m4 = g.add(c_1065353216_st_m5, x_1_st_m5, overflow_mode=g.OverflowMode.MODULAR, alus=[3], output_streams=[g.SG4[2]]) #input_streams=[g.SG4_W[0], g.SG4_E[1]], direction=Direction.WESTWARD
        c_1026031616_st_m5 = c_1026031616.read(streams=g.SG4[4]) #direction=Direction.WESTWARD
        x_3_st_m5 = g.mul(x_0_st_m6, x_0_st_m6, overflow_mode=g.OverflowMode.MODULAR, alus=[10], output_streams=[g.SG4[7]]) #input_streams=[g.SG4_E[7], g.SG4_E[7]], direction=Direction.EASTWARD
        x_4_st_m4 = g.mul(c_1026031616_st_m5, x_3_st_m5, overflow_mode=g.OverflowMode.MODULAR, alus=[11], output_streams=[g.SG4[4]]) #input_streams=[g.SG4_W[4], g.SG4_E[7]], direction=Direction.WESTWARD
        x_5_st_m3 = g.add(x_2_st_m4, x_4_st_m4, overflow_mode=g.OverflowMode.MODULAR, alus=[6], output_streams=[g.SG4[3]]) #input_streams=[g.SG4_W[2], g.SG4_W[4]], direction=Direction.WESTWARD
        c_3130286080_st_m4 = c_3130286080.read(streams=g.SG4[3]) #direction=Direction.WESTWARD
        x_6_st_m4 = g.mul(x_3_st_m5, x_0_id9_st_m5, overflow_mode=g.OverflowMode.MODULAR, alus=[15], output_streams=[g.SG4[0]]) #input_streams=[g.SG4_E[7], g.SG4_E[6]], direction=Direction.WESTWARD
        x_7_st_m3 = g.mul(c_3130286080_st_m4, x_6_st_m4, overflow_mode=g.OverflowMode.MODULAR, alus=[2], output_streams=[g.SG4[2]]) #input_streams=[g.SG4_W[3], g.SG4_W[0]], direction=Direction.WESTWARD
        x_8_st_m2 = g.add(x_5_st_m3, x_7_st_m3, overflow_mode=g.OverflowMode.MODULAR, alus=[5], output_streams=[g.SG4[2]]) #input_streams=[g.SG4_W[3], g.SG4_W[2]], direction=Direction.WESTWARD
        ARG1_st_m3 = ARG1.read(streams=g.SG4[2]) #direction=Direction.EASTWARD
        c_31_st_m3 = c_31.read(streams=g.SG4[1]) #direction=Direction.WESTWARD
        x_9_st_m2 = g.left_shift(ARG1_st_m3.reinterpret(g.uint32), c_31_st_m3, alus=[1], output_streams=[g.SG4[1]]) #input_streams=[g.SG4_E[2], g.SG4_W[1]], direction=Direction.WESTWARD
        x_10_st_m1 = g.bitwise_or(x_8_st_m2, x_9_st_m2.reinterpret(g.float32), alus=[0], output_streams=[g.SG4[0]]) #input_streams=[g.SG4_W[2], g.SG4_W[1]], direction=Direction.EASTWARD
        cos_st_p0 = g.cast(x_10_st_m1, dtype=g.float16, time=0, alus=[13], output_streams=[g.SG4[0]]) #input_streams=[g.SG4_E[0]], direction=Direction.WESTWARD
        return cos_st_p0 #directions=[Direction.WESTWARD]

def cos_poly_float16_apply(ARG0, ARG1, predecessors=None):
        with g.ResourceScope(name="cos_poly_float16_scope", is_buffered=True, time=0 if predecessors is None else None, predecessors=predecessors) as cos_poly_float16_scope:
                c_3130286080 = g.from_data(np.array([-0.0011320114135742188]*320, dtype=np.float32), name='c_3130286080', layout="-1, H1(E), S4")
                c_31 = g.from_data(np.array([31]*320, dtype=np.uint32), name='c_31', layout="-1, H1(E), S4")
                c_1026031616 = g.from_data(np.array([0.041015625]*320, dtype=np.float32), name='c_1026031616', layout="-1, H1(E), S4")
                c_1065353216 = g.from_data(np.array([1.0]*320, dtype=np.float32), name='c_1065353216', layout="-1, H1(E), S4")
                c_3204431872 = g.from_data(np.array([-0.49951171875]*320, dtype=np.float32), name='c_3204431872', layout="-1, H1(E), S4")
                cos_mt = cos_poly_float16(c_1065353216, c_3204431872, ARG0, c_1026031616, c_3130286080, ARG1, c_31)
                cos_mt = cos_mt.write(name="cos", layout=f"-1, H1(W), S{cos_mt.dtype.size}")
                return (cos_mt), cos_poly_float16_scope
        
def sincos(x, skip_fmod=False, sinonly=False, cosonly=False):
  assert x.dtype in (g.float16, g.float32)
  if not skip_fmod:
    rint, rint_scope = rintmulinv_float32_apply(x) if x.dtype == g.float32 else rintmulinv_float16_apply(x)
    f = remainderhighprec_float32_apply if x.dtype == g.float32 else remainderhighprec_float16_apply 
    angle, remainder_scope = f(x, rint.reinterpret(g.float32), predecessors=[rint_scope])
  else: angle = x
  if sinonly:
      f = sin_quads_float32_apply if angle.dtype==g.float32 else sin_quads_float16_apply
      (angle, needneg), quads_scope = f(angle, predecessors=None if skip_fmod else [remainder_scope])
  elif cosonly:
      f = cos_quads_float32_apply if angle.dtype==g.float32 else cos_quads_float16_apply
      (angle, needneg), quads_scope = f(angle, predecessors=None if skip_fmod else [remainder_scope])
  else:
      f = sincos_quads_float32_apply if angle.dtype==g.float32 else sincos_quads_float16_apply
      (angle, needneg, anglesin, neednegsin), quads_scope = f(angle, predecessors=None if skip_fmod else [remainder_scope])
  if not sinonly and not cosonly: angle, needneg = g.stack([angle, anglesin], dim=0), g.stack([needneg, neednegsin], dim=0)
  f = cos_poly_float32_apply if x.dtype==g.float32 else cos_poly_float16_apply
  c, _ = f(angle, needneg, predecessors=[quads_scope])
  if not sinonly and not cosonly: c, s = [z.reshape(*x.shape) for z in g.split(c, dim=0, num_splits=2)]
  c.name = "cos" if cosonly or not sinonly else "sin"
  if not sinonly and not cosonly: s.name = "sin"
  return c if sinonly or cosonly else (s, c)

def sin(x, skip_fmod=False): return sincos(x, skip_fmod=skip_fmod, sinonly=True)

def cos(x, skip_fmod=False): return sincos(x, skip_fmod=skip_fmod, cosonly=True)

def is_zerosubnormal_fp32(x):
  return (x.view(np.uint32) & 0x7F800000) == 0
def subnormal_to_zero_fp32(x): return np.where(is_zerosubnormal_fp32(x), np.copysign(np.float32(0.0), x), x)
def ulp(x):
    emin = np.finfo(x.dtype).minexp
    p = 23 if x.dtype==np.float32 else 10
    #Note: np.frexp(x)[1]-1 == np.floor(np.log2(np.abs(x.astype(np.float64)))).astype(np.int32)
    return np.where(np.isfinite(x) & (x!=0.0), np.ldexp(x.dtype.type(0.5), np.maximum(np.frexp(x)[1]-1, emin)-p+1), np.where(x==0.0, np.finfo(x.dtype).smallest_subnormal, x))
    #np.where(x<0, x-np.nextafter(x, -np.inf), np.nextafter(x, np.inf)-x) #does not handle np.finfo(x).max, np.finfo(x).min
def ufp(x):
    #Note: np.frexp(x)[1]-1 == np.floor(np.log2(np.abs(x.astype(np.float64)))).astype(np.int32)
    return np.where(np.isfinite(x) & (x!=0.0), np.ldexp(x.dtype.type(0.5), np.frexp(x)[1]-1), x)
def getrelabserr(x, r, o):
    p = 23 if x.dtype==np.float32 else 10
    n = np.arange(np.finfo(o.dtype).minexp, np.finfo(o.dtype).maxexp+1)
    return n[np.where(np.all(np.maximum(np.ldexp(0.5, n[:,None,None]+np.frexp(x)[1]-1-p), np.ldexp(0.5, n[:,None,None]-p)) >= np.abs(r - o)[None,...], axis=tuple(range(1, len(x.shape)+1))))[0][0]]
def test_sincos(dtype, func, skip_fmod=False):
    shape = (256, 320)
    npdtype = dtype.to_nptype()
    pgm_pkg = g.ProgramPackage(
            name="math",
            output_dir=os.path.join(os.path.dirname(__file__), "sincos"),
            gen_vis_data=False,
            check_stream_conflicts=True,
        )
    with pgm_pkg.create_program_context("compute"):
        t = g.input_tensor(shape=shape, dtype=dtype, name="x")
        with g.ResourceScope(name="compute_scope", is_buffered=True, time=0, predecessors=[]) as scope:
            res = func(t, skip_fmod=skip_fmod)
            if isinstance(res, tuple):
              for r in res: r.set_program_output()
            else:
              res.set_program_output()
    iop_file = pgm_pkg.assemble()[0]
    p = subprocess.run(["iop-utils", "stats", iop_file], capture_output=True, text=True)
    progcycles = [int(x.replace("Program is ","").replace(" cycles.","")) for x in p.stdout.splitlines() if x.startswith("Program is ")]
    print("Program cycles:", progcycles, "Compute time:", f"{progcycles[0]/900/1000000:.20f}")
    from groq.runner import tsp
    runner = tsp.create_tsp_runner(iop_file)
    if skip_fmod:
        r = (np.random.rand(*shape) * 4 * np.pi - 2 * np.pi).astype(npdtype)
    else:
        r = (np.random.rand(*shape) * 2**11 * 2 * np.pi - 2**11 * 2 * np.pi).astype(npdtype)
    results = {}
    import timeit
    def timerunner():
      results.update(runner(x=r))
    if dtype==g.float32: r = subnormal_to_zero_fp32(r) #Groq chip behavior is to zero input/output subnormals so we compare to same CPU behavior 
    time_lpu = timeit.timeit(timerunner, number=1)
    oracle = {}
    meps = np.finfo(npdtype).eps
    def oraclerunner():
        if "sin" in results:
            oracle["sin"] = np.sin(r)
            #oracle["sin"] = np.sin(r.astype(np.float64)).astype(npdtype)
        if "cos" in results:    
            oracle["cos"] = np.cos(r)
            #oracle["cos"] = np.cos(r.astype(np.float64)).astype(npdtype)
    time_oracle = timeit.timeit(oraclerunner, number=1)
    print("Groq time", time_lpu, "CPU time", time_oracle)
    for x in results:
        if dtype==g.float32: oracle[x] = subnormal_to_zero_fp32(oracle[x])
        #assert np.allclose(oracle[x], results[x]), (oracle[x], results[x])
        #print(getrelabserr(r, results[x], oracle[x]))        
        absdiff = np.abs(oracle[x] - results[x])
        relerr = absdiff/np.abs(oracle[x])
        ulperr = np.minimum(absdiff/meps, absdiff/ulp(r))
        print(f"{x}\n", "Max Abs. Acc", np.max(absdiff), "Avg. Abs. ULP", np.average(absdiff),
              "Max Rel. Acc", np.max(relerr), "Avg. Rel. Acc", np.average(relerr),
              "Max ULP", np.max(ulperr), "Av. ULP", np.average(ulperr))
for dtype in (g.float32, g.float16):
    for func, skip_fmod in ((sin, True), (sin, False), (cos, True), (cos, False), (sincos, True), (sincos, False)):
        test_sincos(dtype, func, skip_fmod=skip_fmod)

