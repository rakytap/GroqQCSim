import os, subprocess
import numpy as np
import groq.api as g

#float16/float32 compatible
def remainder_float16(x, yinv, y, expmask, sgnmask, mantbias):
        x_st = x.read(streams=g.SG4[5]) #direction=Direction.EASTWARD
        yinv_st = yinv.read(streams=g.SG4[7]) #direction=Direction.EASTWARD
        y_st = y.read(streams=g.SG4[2]) #direction=Direction.EASTWARD
        expmask_st = expmask.read(streams=g.SG4[0]).reinterpret(x.dtype) #direction=Direction.EASTWARD
        save = g.memory_tensor(shape=x_st.shape, dtype=x_st.dtype, name="x_st_store", layout=f"-1, H1(E), B2, S{x.dtype.size}(0-{x.dtype.size-1})", is_static=True)
        x_st.write(name="x_st", storage_req=save.storage_request)
        xd_st = save.read(streams=g.SG4[4]) #direction=Direction.WESTWARD
        xyinv_st = g.mul(x_st, yinv_st, alus=[8], output_streams=[g.SG4[6]]) #input_streams=[g.SG4_E[5], g.SG4_E[7]], direction=Direction.EASTWARD
        xyinv_id10_st = g.vxm_identity(xyinv_st, alus=[10], output_streams=[g.SG4[7]]) #input_streams=[g.SG4_E[6]], direction=Direction.EASTWARD
        xyinv_id11_st = g.vxm_identity(xyinv_id10_st, alus=[11], output_streams=[g.SG4[6]]) #input_streams=[g.SG4_E[7]], direction=Direction.WESTWARD
        xint_st = g.cast(xyinv_st, dtype=g.int16, alus=[15], output_streams=[g.SG4[0]]) #input_streams=[g.SG4_E[6]], direction=Direction.WESTWARD
        rintx_st = g.cast(xint_st, dtype=g.float16, alus=[2], output_streams=[g.SG4[0]]) #input_streams=[g.SG4_W[0]], direction=Direction.WESTWARD
        exp_st = g.bitwise_and(xyinv_st, expmask_st, alus=[13], output_streams=[g.SG4[1]]) #input_streams=[g.SG4_E[6], g.SG4_E[0]], direction=Direction.WESTWARD
        sgnmask_st = sgnmask.read(streams=g.SG4[6]) #direction=Direction.EASTWARD
        mantbias_st = mantbias.read(streams=g.SG4[1]) #direction=Direction.EASTWARD
        isint_st = g.greater_equal(exp_st, mantbias_st, alus=[0], output_streams=[g.SG4[1]]) #input_streams=[g.SG4_W[1], g.SG4_E[1]], direction=Direction.EASTWARD
        intx_st = g.mask(isint_st, xyinv_id11_st, alus=[14], output_streams=[g.SG4[7]]) #input_streams=[g.SG4_E[1], g.SG4_W[6]], direction=Direction.WESTWARD
        sgn_st = g.bitwise_and(xyinv_id11_st, sgnmask_st, alus=[12], output_streams=[g.SG4[7]]) #input_streams=[g.SG4_W[6], g.SG4_E[6]], direction=Direction.EASTWARD
        intxsgn_st = g.bitwise_or(intx_st, sgn_st, alus=[9], output_streams=[g.SG4[5]]) #input_streams=[g.SG4_W[7], g.SG4_E[7]], direction=Direction.WESTWARD
        smallx_st = g.mask_bar(isint_st, rintx_st, alus=[1], output_streams=[g.SG4[3]]) #input_streams=[g.SG4_E[1], g.SG4_W[0]], direction=Direction.EASTWARD
        smallx_id6_st = g.vxm_identity(smallx_st, alus=[6], output_streams=[g.SG4[2]]) #input_streams=[g.SG4_E[3]], direction=Direction.WESTWARD
        rfloatadj_st = g.bitwise_or(intxsgn_st, smallx_id6_st, alus=[4], output_streams=[g.SG4[4]]) #input_streams=[g.SG4_W[5], g.SG4_W[2]], direction=Direction.EASTWARD
        xfrac_st = g.mul(rfloatadj_st, y_st, alus=[5], output_streams=[g.SG4[2]]) #input_streams=[g.SG4_E[4], g.SG4_E[2]], direction=Direction.EASTWARD
        r_st = g.sub(xd_st, xfrac_st, alus=[7], output_streams=[g.SG4[3]]) #input_streams=[g.SG4_W[4], g.SG4_E[2]], direction=Direction.WESTWARD
        return x_st, r_st #directions=[Direction.EASTWARD, Direction.WESTWARD]

def remainder_float32(x, yinv, y, y2, expmask, one):
        x_st = x.read(streams=g.SG4[3]) #direction=Direction.EASTWARD
        yinv_st = yinv.read(streams=g.SG4[1]) #direction=Direction.EASTWARD
        y_st = y.read(streams=g.SG4[6]) #direction=Direction.EASTWARD
        y2_st = y2.read(streams=g.SG4[5]) #direction=Direction.EASTWARD
        expmask_st = expmask.read(streams=g.SG4[0]).reinterpret(x.dtype) #direction=Direction.EASTWARD
        save = g.memory_tensor(shape=x_st.shape, dtype=x_st.dtype, name="x_st_store", layout=f"-1, H1(E), B2, 4, S{x.dtype.size}(0-{x.dtype.size-1})", is_static=True)
        x_st.write(name="x_st", storage_req=save.storage_request)
        xd_st = save.read(streams=g.SG4[0]) #direction=Direction.WESTWARD
        xyinv_st = g.mul(x_st, yinv_st, alus=[1], output_streams=[g.SG4[2]]) #input_streams=[g.SG4_E[3], g.SG4_E[1]], direction=Direction.WESTWARD
        xyinv_id4_st = g.vxm_identity(xyinv_st, alus=[4], output_streams=[g.SG4[4]]) #input_streams=[g.SG4_W[2]], direction=Direction.EASTWARD
        xyinv_id6_st = g.vxm_identity(xyinv_id4_st, alus=[6], output_streams=[g.SG4[5]]) #input_streams=[g.SG4_E[4]], direction=Direction.EASTWARD
        xyinv_id11_st = g.vxm_identity(xyinv_id6_st, alus=[11], output_streams=[g.SG4[6]]) #input_streams=[g.SG4_E[5]], direction=Direction.WESTWARD
        xint_st = g.cast(xyinv_id6_st, dtype=g.int32, alus=[7], output_streams=[g.SG4[3]]) #input_streams=[g.SG4_E[5]], direction=Direction.WESTWARD
        rintx_st = g.cast(xint_st, dtype=g.float32, alus=[2], output_streams=[g.SG4[1]]) #input_streams=[g.SG4_W[3]], direction=Direction.EASTWARD
        exp_st = g.bitwise_and(xyinv_st, expmask_st, alus=[0], output_streams=[g.SG4[2]]) #input_streams=[g.SG4_W[2], g.SG4_E[0]], direction=Direction.EASTWARD
        one_st = one.read(streams=g.SG4[7]).reinterpret(expmask.dtype) #direction=Direction.EASTWARD
        expfloat_st = g.cast(exp_st, dtype=g.uint32, alus=[5], output_streams=[g.SG4[5]]) #input_streams=[g.SG4_E[2]], direction=Direction.WESTWARD
        minexpone_st = g.min(expfloat_st, one_st, alus=[8], output_streams=[g.SG4[7]]).reinterpret(x.dtype) #input_streams=[g.SG4_W[5], g.SG4_E[7]], direction=Direction.EASTWARD
        intx_st = g.mul(minexpone_st, xyinv_id11_st, alus=[10], output_streams=[g.SG4[6]]) #input_streams=[g.SG4_E[7], g.SG4_W[6]], direction=Direction.EASTWARD
        rfloatadj_st = g.max(rintx_st.reinterpret(expmask.dtype), intx_st.reinterpret(expmask.dtype), alus=[15], output_streams=[g.SG4[7]]).reinterpret(x.dtype) #input_streams=[g.SG4_E[1], g.SG4_E[6]], direction=Direction.WESTWARD
        xfrac_st = g.mul(rfloatadj_st, y_st, alus=[14], output_streams=[g.SG4[0]]) #input_streams=[g.SG4_W[7], g.SG4_E[6]], direction=Direction.EASTWARD
        r1_st = g.sub(xd_st, xfrac_st, alus=[3], output_streams=[g.SG4[1]]) #input_streams=[g.SG4_W[0], g.SG4_E[0]], direction=Direction.WESTWARD
        xfrac2_st = g.mul(rfloatadj_st, y2_st, alus=[9], output_streams=[g.SG4[6]]) #input_streams=[g.SG4_W[7], g.SG4_E[5]], direction=Direction.WESTWARD
        xfrac2_id12_st = g.vxm_identity(xfrac2_st, alus=[12], output_streams=[g.SG4[0]]) #input_streams=[g.SG4_W[6]], direction=Direction.EASTWARD
        r_st = g.sub(r1_st, xfrac2_id12_st, alus=[13], output_streams=[g.SG4[1]], time=0) #input_streams=[g.SG4_W[1], g.SG4_E[0]], direction=Direction.WESTWARD
        return r_st #directions=[Direction.WESTWARD]
        
#float16/float32 compatible       
def sin_quads(angle, pi, threehalfpi, halfpi, sgnmask, mantexpmask1, mantexpmask2):
        angle_st = angle.read(streams=g.SG4[3]) #direction=Direction.EASTWARD
        angle_id4_st = g.vxm_identity(angle_st, alus=[4], output_streams=[g.SG4[4]]) #input_streams=[g.SG4_E[3]], direction=Direction.EASTWARD
        angle_id9_st = g.vxm_identity(angle_id4_st, alus=[9], output_streams=[g.SG4[5]]) #input_streams=[g.SG4_E[4]], direction=Direction.EASTWARD
        pi_st = pi.read(streams=g.SG4[4]) #direction=Direction.WESTWARD
        threehalfpi_st = threehalfpi.read(streams=g.SG4[3]) #direction=Direction.WESTWARD
        halfpi_st = halfpi.read(streams=g.SG4[6]) #direction=Direction.WESTWARD
        sgnmask_st = sgnmask.read(streams=g.SG4[5]).reinterpret(angle.dtype) #direction=Direction.WESTWARD
        mantexpmask1_st = mantexpmask1.read(streams=g.SG4[2]).reinterpret(angle.dtype) #direction=Direction.WESTWARD
        mantexpmask2_st = mantexpmask2.read(streams=g.SG4[1]).reinterpret(angle.dtype) #direction=Direction.WESTWARD
        absangle_st = g.bitwise_and(angle_st, mantexpmask1_st, alus=[1], output_streams=[g.SG4[0]]) #input_streams=[g.SG4_E[3], g.SG4_W[2]], direction=Direction.EASTWARD
        threehalfpisubangle_st = g.sub(threehalfpi_st, absangle_st, alus=[2], output_streams=[g.SG4[3]]) #input_streams=[g.SG4_W[3], g.SG4_E[0]], direction=Direction.EASTWARD
        halfpisubangle_st = g.sub(halfpi_st, absangle_st, alus=[15], output_streams=[g.SG4[7]]) #input_streams=[g.SG4_W[6], g.SG4_E[0]], direction=Direction.WESTWARD
        abshalfpisubangle_st = g.bitwise_and(halfpisubangle_st, mantexpmask2_st, alus=[14], output_streams=[g.SG4[7]]) #input_streams=[g.SG4_W[7], g.SG4_W[1]], direction=Direction.WESTWARD
        absthreehalfpisubangle_st = g.bitwise_and(threehalfpisubangle_st, mantexpmask2_st, alus=[3], output_streams=[g.SG4[0]]) #input_streams=[g.SG4_E[3], g.SG4_W[1]], direction=Direction.WESTWARD
        r_st = g.min(abshalfpisubangle_st, absthreehalfpisubangle_st, alus=[13], output_streams=[g.SG4[6]], time=0) #input_streams=[g.SG4_W[7], g.SG4_W[0]], direction=Direction.WESTWARD
        sgnangle_st = g.bitwise_and(angle_st, sgnmask_st, alus=[6], output_streams=[g.SG4[4]]) #input_streams=[g.SG4_E[3], g.SG4_W[5]], direction=Direction.EASTWARD
        signpi_st = g.bitwise_or(sgnangle_st, pi_st, alus=[11], output_streams=[g.SG4[6]]) #input_streams=[g.SG4_E[4], g.SG4_W[4]], direction=Direction.WESTWARD
        needneg_st = g.greater(angle_id9_st, signpi_st, alus=[10], output_streams=[g.SG4[4]]) #input_streams=[g.SG4_E[5], g.SG4_W[6]], direction=Direction.WESTWARD
        return r_st, needneg_st #directions=[Direction.WESTWARD, Direction.WESTWARD]

#float16/float32 compatible        
def cos_quads(angle, pi, twopi, halfpi, mantexpmask1, mantexpmask2):
        angle_st = angle.read(streams=g.SG4[0]) #direction=Direction.EASTWARD
        pi_st = pi.read(streams=g.SG4[2]) #direction=Direction.WESTWARD
        twopi_st = twopi.read(streams=g.SG4[5]) #direction=Direction.WESTWARD
        halfpi_st = halfpi.read(streams=g.SG4[1]) #direction=Direction.WESTWARD
        mantexpmask1_st = mantexpmask1.read(streams=g.SG4[0]).reinterpret(angle.dtype) #direction=Direction.WESTWARD
        mantexpmask2_st = mantexpmask2.read(streams=g.SG4[6]).reinterpret(angle.dtype) #direction=Direction.WESTWARD
        absangle_st = g.bitwise_and(angle_st, mantexpmask1_st, alus=[1], output_streams=[g.SG4[1]]) #input_streams=[g.SG4_E[0], g.SG4_W[0]], direction=Direction.EASTWARD
        absangle_id14_st = g.vxm_identity(absangle_st, alus=[14], output_streams=[g.SG4[6]]) #input_streams=[g.SG4_E[1]], direction=Direction.WESTWARD
        absangle_id9_st = g.vxm_identity(absangle_id14_st, alus=[9], output_streams=[g.SG4[4]]) #input_streams=[g.SG4_W[6]], direction=Direction.EASTWARD
        twopisubangle_st = g.sub(twopi_st, absangle_id9_st, alus=[7], output_streams=[g.SG4[5]]) #input_streams=[g.SG4_W[5], g.SG4_E[4]], direction=Direction.WESTWARD
        pisubangle_st = g.sub(pi_st, absangle_st, alus=[2], output_streams=[g.SG4[0]]) #input_streams=[g.SG4_W[2], g.SG4_E[1]], direction=Direction.EASTWARD
        abspisubangle_st = g.bitwise_and(pisubangle_st, mantexpmask2_st, alus=[15], output_streams=[g.SG4[7]]) #input_streams=[g.SG4_E[0], g.SG4_W[6]], direction=Direction.WESTWARD
        minpi_st = g.min(absangle_id9_st, abspisubangle_st, alus=[10], output_streams=[g.SG4[4]]) #input_streams=[g.SG4_E[4], g.SG4_W[7]], direction=Direction.WESTWARD
        r_st = g.min(minpi_st, twopisubangle_st, alus=[5], output_streams=[g.SG4[4]], time=0) #input_streams=[g.SG4_W[4], g.SG4_W[5]], direction=Direction.WESTWARD
        needneg_st = g.less(abspisubangle_st, halfpi_st, alus=[13], output_streams=[g.SG4[0]]) #input_streams=[g.SG4_W[7], g.SG4_W[1]], direction=Direction.WESTWARD
        return r_st, needneg_st #directions=[Direction.WESTWARD, Direction.WESTWARD]

def sincos_quads_float16(angle, pi, twopi, halfpi, mantexpmask1, mantexpmask2, sgnmask, halfpis):
        angle_st = angle.read(streams=g.SG4[0]) #direction=Direction.EASTWARD
        angle_id1_st = g.vxm_identity(angle_st, alus=[1], output_streams=[g.SG4[0]]) #input_streams=[g.SG4_E[0]], direction=Direction.WESTWARD
        angle_id12_st = g.vxm_identity(angle_id1_st, alus=[12], output_streams=[g.SG4[7]]) #input_streams=[g.SG4_W[0]], direction=Direction.EASTWARD
        pi_st = pi.read(streams=g.SG4[6]) #direction=Direction.WESTWARD
        twopi_st = twopi.read(streams=g.SG4[2]) #direction=Direction.WESTWARD
        halfpi_st = halfpi.read(streams=g.SG4[1]) #direction=Direction.WESTWARD
        mantexpmask1_st = mantexpmask1.read(streams=g.SG4[4]) #direction=Direction.WESTWARD
        mantexpmask2_st = mantexpmask2.read(streams=g.SG4[7]) #direction=Direction.WESTWARD
        absangle_st = g.bitwise_and(angle_id12_st, mantexpmask1_st, alus=[11], output_streams=[g.SG4[4]]) #input_streams=[g.SG4_E[7], g.SG4_W[4]], direction=Direction.WESTWARD
        absangle_id6_st = g.vxm_identity(absangle_st, alus=[6], output_streams=[g.SG4[2]]) #input_streams=[g.SG4_W[4]], direction=Direction.WESTWARD
        absangle_id0_st = g.vxm_identity(absangle_id6_st, alus=[0], output_streams=[g.SG4[3]]) #input_streams=[g.SG4_W[2]], direction=Direction.EASTWARD
        absangle_id4_st = g.vxm_identity(absangle_id6_st, alus=[4], output_streams=[g.SG4[4]]) #input_streams=[g.SG4_W[2]], direction=Direction.EASTWARD
        twopisubangle_st = g.sub(twopi_st, absangle_id0_st, alus=[2], output_streams=[g.SG4[3]]) #input_streams=[g.SG4_W[2], g.SG4_E[3]], direction=Direction.EASTWARD
        pisubangle_st = g.sub(pi_st, absangle_st, alus=[8], output_streams=[g.SG4[6]]) #input_streams=[g.SG4_W[6], g.SG4_W[4]], direction=Direction.EASTWARD
        abspisubangle_st = g.bitwise_and(pisubangle_st, mantexpmask2_st, alus=[9], output_streams=[g.SG4[6]]) #input_streams=[g.SG4_E[6], g.SG4_W[7]], direction=Direction.EASTWARD
        minpi_st = g.min(absangle_id4_st, abspisubangle_st, alus=[10], output_streams=[g.SG4[5]]) #input_streams=[g.SG4_E[4], g.SG4_E[6]], direction=Direction.EASTWARD
        r_st = g.min(minpi_st, twopisubangle_st, alus=[7], output_streams=[g.SG4[5]]) #input_streams=[g.SG4_E[5], g.SG4_E[3]], direction=Direction.WESTWARD
        needneg_st = g.less(abspisubangle_st, halfpi_st, alus=[15], output_streams=[g.SG4[1]]) #input_streams=[g.SG4_E[6], g.SG4_W[1]], direction=Direction.WESTWARD
        sgnmask_st = sgnmask.read(streams=g.SG4[0]) #direction=Direction.WESTWARD
        sgnangle_st = g.bitwise_and(angle_st, sgnmask_st, alus=[3], output_streams=[g.SG4[0]]) #input_streams=[g.SG4_E[0], g.SG4_W[0]], direction=Direction.WESTWARD
        signpi_st = g.bitwise_or(sgnangle_st, pi_st, alus=[14], output_streams=[g.SG4[0]]) #input_streams=[g.SG4_W[0], g.SG4_W[6]], direction=Direction.WESTWARD
        neednegsin_st = g.greater(angle_id12_st, signpi_st, alus=[13], output_streams=[g.SG4[7]]) #input_streams=[g.SG4_E[7], g.SG4_W[0]], direction=Direction.WESTWARD
        halfpis_st = halfpis.read(streams=g.SG4[3]) #direction=Direction.WESTWARD
        rsin_st = g.sub(halfpis_st, r_st, alus=[5], output_streams=[g.SG4[3]]) #input_streams=[g.SG4_W[3], g.SG4_W[5]], direction=Direction.WESTWARD
        return r_st, needneg_st, neednegsin_st, rsin_st #directions=[Direction.WESTWARD, Direction.WESTWARD, Direction.WESTWARD, Direction.WESTWARD]

def sincos_quads(angle, pi, twopi, halfpi, mantexpmask1, mantexpmask2, sgnmask, halfpis):
        angle_st = angle.read(streams=g.SG4[1]) #direction=Direction.EASTWARD
        angle_id2_st = g.vxm_identity(angle_st, alus=[2], output_streams=[g.SG4[2]]) #input_streams=[g.SG4_E[1]], direction=Direction.WESTWARD
        angle_id0_st = g.vxm_identity(angle_id2_st, alus=[0], output_streams=[g.SG4[0]]) #input_streams=[g.SG4_W[2]], direction=Direction.EASTWARD
        pi_st = pi.read(streams=g.SG4[7]) #direction=Direction.WESTWARD
        twopi_st = twopi.read(streams=g.SG4[2]) #direction=Direction.WESTWARD
        halfpi_st = halfpi.read(streams=g.SG4[5]) #direction=Direction.WESTWARD
        mantexpmask1_st = mantexpmask1.read(streams=g.SG4[6]).reinterpret(angle.dtype) #direction=Direction.WESTWARD
        mantexpmask2_st = mantexpmask2.read(streams=g.SG4[4]).reinterpret(angle.dtype) #direction=Direction.WESTWARD
        absangle_st = g.bitwise_and(angle_st, mantexpmask1_st, alus=[15], output_streams=[g.SG4[6]]) #input_streams=[g.SG4_E[1], g.SG4_W[6]], direction=Direction.WESTWARD
        absangle_id9_st = g.vxm_identity(absangle_st, alus=[9], output_streams=[g.SG4[4]]) #input_streams=[g.SG4_W[6]], direction=Direction.WESTWARD
        absangle_id8_st = g.vxm_identity(absangle_id9_st, alus=[8], output_streams=[g.SG4[4]]) #input_streams=[g.SG4_W[4]], direction=Direction.EASTWARD
        twopisubangle_st = g.sub(twopi_st, absangle_id8_st, alus=[6], output_streams=[g.SG4[5]]) #input_streams=[g.SG4_W[2], g.SG4_E[4]], direction=Direction.EASTWARD
        pisubangle_st = g.sub(pi_st, absangle_st, alus=[14], output_streams=[g.SG4[7]]) #input_streams=[g.SG4_W[7], g.SG4_W[6]], direction=Direction.EASTWARD
        abspisubangle_st = g.bitwise_and(pisubangle_st, mantexpmask2_st, alus=[11], output_streams=[g.SG4[4]]) #input_streams=[g.SG4_E[7], g.SG4_W[4]], direction=Direction.WESTWARD
        minpi_st = g.min(absangle_id8_st, abspisubangle_st, alus=[5], output_streams=[g.SG4[3]]) #input_streams=[g.SG4_E[4], g.SG4_W[4]], direction=Direction.EASTWARD
        r_st = g.min(minpi_st, twopisubangle_st, alus=[7], output_streams=[g.SG4[3]], time=0) #input_streams=[g.SG4_E[3], g.SG4_E[5]], direction=Direction.WESTWARD
        needneg_st = g.less(abspisubangle_st, halfpi_st, alus=[10], output_streams=[g.SG4[5]]) #input_streams=[g.SG4_W[4], g.SG4_W[5]], direction=Direction.WESTWARD
        sgnmask_st = sgnmask.read(streams=g.SG4[3]).reinterpret(angle.dtype) #direction=Direction.WESTWARD
        sgnangle_st = g.bitwise_and(angle_st, sgnmask_st, alus=[3], output_streams=[g.SG4[1]]) #input_streams=[g.SG4_E[1], g.SG4_W[3]], direction=Direction.WESTWARD
        signpi_st = g.bitwise_or(sgnangle_st, pi_st, alus=[12], output_streams=[g.SG4[7]]) #input_streams=[g.SG4_W[1], g.SG4_W[7]], direction=Direction.EASTWARD
        neednegsin_st = g.greater(angle_id0_st, signpi_st, alus=[13], output_streams=[g.SG4[6]]) #input_streams=[g.SG4_E[0], g.SG4_E[7]], direction=Direction.WESTWARD
        halfpis_st = halfpis.read(streams=g.SG4[0]) #direction=Direction.WESTWARD
        rsin_st = g.sub(halfpis_st, r_st, alus=[1], output_streams=[g.SG4[0]]) #input_streams=[g.SG4_W[0], g.SG4_W[3]], direction=Direction.WESTWARD
        return r_st, needneg_st, neednegsin_st, rsin_st #directions=[Direction.WESTWARD, Direction.WESTWARD, Direction.WESTWARD, Direction.WESTWARD]
        
def sincos_approx_float16(x, needneg, c1, c2, c3, c4, sgnshift):
        x_st = x.read(streams=g.SG4[4]) #direction=Direction.EASTWARD
        needneg_st = needneg.read(streams=g.SG4[5]) #direction=Direction.EASTWARD
        c1_st = c1.read(streams=g.SG4[3]) #direction=Direction.WESTWARD
        c2_st = c2.read(streams=g.SG4[0]) #direction=Direction.WESTWARD
        c3_st = c3.read(streams=g.SG4[1]) #direction=Direction.WESTWARD
        c4_st = c4.read(streams=g.SG4[6]) #direction=Direction.WESTWARD
        sgnshift_st = sgnshift.read(streams=g.SG4[7]) #direction=Direction.WESTWARD
        x2_st = g.mul(x_st, x_st, alus=[8], output_streams=[g.SG4[6]]) #input_streams=[g.SG4_E[4], g.SG4_E[4]], direction=Direction.EASTWARD
        x2_id9_st = g.vxm_identity(x2_st, alus=[9], output_streams=[g.SG4[7]]) #input_streams=[g.SG4_E[6]], direction=Direction.WESTWARD
        x2_id12_st = g.vxm_identity(x2_id9_st, alus=[12], output_streams=[g.SG4[0]]) #input_streams=[g.SG4_W[7]], direction=Direction.EASTWARD
        x2_id1_st = g.vxm_identity(x2_id12_st, alus=[1], output_streams=[g.SG4[1]]) #input_streams=[g.SG4_E[0]], direction=Direction.WESTWARD
        x2_id0_st = g.vxm_identity(x2_id1_st, alus=[0], output_streams=[g.SG4[3]]) #input_streams=[g.SG4_W[1]], direction=Direction.EASTWARD
        c4x2_st = g.mul(c4_st, x2_st, alus=[10], output_streams=[g.SG4[6]]) #input_streams=[g.SG4_W[6], g.SG4_E[6]], direction=Direction.EASTWARD
        c3_c4x2_st = g.add(c3_st, c4x2_st, alus=[15], output_streams=[g.SG4[7]]) #input_streams=[g.SG4_W[1], g.SG4_E[6]], direction=Direction.WESTWARD
        c3x2_c4x4_st = g.mul(x2_id12_st, c3_c4x2_st, alus=[13], output_streams=[g.SG4[1]]) #input_streams=[g.SG4_E[0], g.SG4_W[7]], direction=Direction.EASTWARD
        c2_c3x2_c4x4_st = g.add(c2_st, c3x2_c4x4_st, alus=[2], output_streams=[g.SG4[2]]) #input_streams=[g.SG4_W[0], g.SG4_E[1]], direction=Direction.WESTWARD
        c2x2_c3x4_c4x6_st = g.mul(x2_id0_st, c2_c3x2_c4x4_st, alus=[5], output_streams=[g.SG4[2]]) #input_streams=[g.SG4_E[3], g.SG4_W[2]], direction=Direction.EASTWARD
        r_st = g.add(c1_st, c2x2_c3x4_c4x6_st, alus=[7], output_streams=[g.SG4[3]]).reinterpret(needneg.dtype) #input_streams=[g.SG4_W[3], g.SG4_E[2]], direction=Direction.WESTWARD
        negsign_st = g.left_shift(needneg_st, sgnshift_st, alus=[11], output_streams=[g.SG4[5]]) #input_streams=[g.SG4_E[5], g.SG4_W[7]], direction=Direction.WESTWARD
        rneg_st = g.bitwise_xor(r_st, negsign_st, alus=[6], output_streams=[g.SG4[5]], time=0) #input_streams=[g.SG4_W[3], g.SG4_W[5]], direction=Direction.WESTWARD
        return rneg_st #directions=[Direction.WESTWARD]

def sincos_approx_float32(x, needneg, c1, c2, c3, c4, c5, sgnshift):
        x_st = x.read(streams=g.SG4[1]) #direction=Direction.EASTWARD
        needneg_st = needneg.read(streams=g.SG4[5]) #direction=Direction.EASTWARD
        c1_st = c1.read(streams=g.SG4[2]) #direction=Direction.WESTWARD
        c2_st = c2.read(streams=g.SG4[1]) #direction=Direction.WESTWARD
        c3_st = c3.read(streams=g.SG4[4]) #direction=Direction.WESTWARD
        c4_st = c4.read(streams=g.SG4[3]) #direction=Direction.WESTWARD
        c5_st = c5.read(streams=g.SG4[0]) #direction=Direction.WESTWARD
        sgnshift_st = sgnshift.read(streams=g.SG4[5]) #direction=Direction.WESTWARD
        x2_st = g.mul(x_st, x_st, alus=[12], output_streams=[g.SG4[0]]) #input_streams=[g.SG4_E[1], g.SG4_E[1]], direction=Direction.EASTWARD
        x2_id15_st = g.vxm_identity(x2_st, alus=[15], output_streams=[g.SG4[7]]) #input_streams=[g.SG4_E[0]], direction=Direction.WESTWARD
        c2x2_st = g.mul(x2_st, c2_st, alus=[2], output_streams=[g.SG4[3]]) #input_streams=[g.SG4_E[0], g.SG4_W[1]], direction=Direction.EASTWARD
        c1_c2x2_st = g.add(c1_st, c2x2_st, alus=[3], output_streams=[g.SG4[2]]) #input_streams=[g.SG4_W[2], g.SG4_E[3]], direction=Direction.WESTWARD
        x4_st = g.mul(x2_st, x2_st, alus=[13], output_streams=[g.SG4[7]]) #input_streams=[g.SG4_E[0], g.SG4_E[0]], direction=Direction.EASTWARD
        c3x4_st = g.mul(x4_st, c3_st, alus=[11], output_streams=[g.SG4[4]]) #input_streams=[g.SG4_E[7], g.SG4_W[4]], direction=Direction.WESTWARD
        x6_st = g.mul(x4_st, x2_id15_st, alus=[10], output_streams=[g.SG4[5]]) #input_streams=[g.SG4_E[7], g.SG4_W[7]], direction=Direction.EASTWARD
        c1_c2x2_c3x4_st = g.add(c1_c2x2_st, c3x4_st, alus=[6], output_streams=[g.SG4[2]]) #input_streams=[g.SG4_W[2], g.SG4_W[4]], direction=Direction.WESTWARD
        c4x6_st = g.mul(x6_st, c4_st, alus=[7], output_streams=[g.SG4[3]]) #input_streams=[g.SG4_E[5], g.SG4_W[3]], direction=Direction.WESTWARD
        x8_st = g.mul(x4_st, x4_st, alus=[14], output_streams=[g.SG4[1]]) #input_streams=[g.SG4_E[7], g.SG4_E[7]], direction=Direction.WESTWARD
        x8_id0_st = g.vxm_identity(x8_st, alus=[0], output_streams=[g.SG4[2]]) #input_streams=[g.SG4_W[1]], direction=Direction.EASTWARD
        c1_c2x2_c3x4_c4x6_st = g.add(c1_c2x2_c3x4_st, c4x6_st, alus=[5], output_streams=[g.SG4[4]]) #input_streams=[g.SG4_W[2], g.SG4_W[3]], direction=Direction.WESTWARD
        c5x8_st = g.mul(x8_id0_st, c5_st, alus=[1], output_streams=[g.SG4[2]]) #input_streams=[g.SG4_E[2], g.SG4_W[0]], direction=Direction.WESTWARD
        r_st = g.add(c1_c2x2_c3x4_c4x6_st, c5x8_st, alus=[4], output_streams=[g.SG4[5]]).reinterpret(needneg.dtype) #input_streams=[g.SG4_W[4], g.SG4_W[2]], direction=Direction.EASTWARD
        negsign_st = g.left_shift(needneg_st, sgnshift_st, alus=[8], output_streams=[g.SG4[4]]) #input_streams=[g.SG4_E[5], g.SG4_W[5]], direction=Direction.EASTWARD
        rneg_st = g.bitwise_xor(r_st, negsign_st, alus=[9], output_streams=[g.SG4[6]], time=0) #input_streams=[g.SG4_E[5], g.SG4_E[4]], direction=Direction.WESTWARD
        return rneg_st #directions=[Direction.WESTWARD]

def remainder(x, y):
  assert x.dtype in (g.float16, g.float32)
  if x.dtype == g.float16: mant, exp = 10, 5
  elif x.dtype == g.float32: mant, exp = 23, 8
  uintdtype = g.uint32 if x.dtype == g.float32 else g.uint16
  intdtype = np.int32 if x.dtype == g.float32 else np.int16
  yp = (np.array([y]*320, dtype=np.float32 if x.dtype==g.float32 else np.float16).view(uintdtype.to_nptype()) & intdtype(-1<<(mant+1)//2).view(uintdtype.to_nptype())).view(np.float32 if x.dtype==g.float32 else np.float16)
  yp2 = (y - yp.astype(np.float64)).astype(np.float32 if x.dtype==g.float32 else np.float16)
  yinv = g.from_data(np.array([1/y]*320, dtype=np.float32 if x.dtype==g.float32 else np.float16), name='yinv')
  y = g.from_data(yp, name='y')
  y2 = g.from_data(yp2, name='y2')
  npuintdtype = np.uint32 if x.dtype == g.float32 else np.uint16
  expbias = (1<<(exp-1))-1
  expmask = g.from_data(np.array([((1<<exp)-1)<<mant]*320, dtype=npuintdtype), name='expbias')
  one = g.from_data(np.array([1.0]*320, dtype=np.float32 if x.dtype==g.float32 else np.float16), name="one")
  if x.dtype==g.float16: res = remainder_float16(x, yinv, y, expmask, sgnmask, mantbias)
  else: res = remainder_float32(x, yinv, y, y2, expmask, one)
  res = res.write(name='remainder')
  return res

def sincos(x, skip_fmod=False, sinonly=False, cosonly=False):
  assert x.dtype in (g.float16, g.float32)
  npdtype = np.float32 if x.dtype==g.float32 else np.float16
  degree = 4 if x.dtype == g.float32 else 3
  if not skip_fmod:
    with g.ResourceScope(name="cos_fmod", is_buffered=True, time=0) as cos_fmod:
      x = remainder(x, 2*np.pi)
  halfpi = g.from_data(np.array([np.pi/2]*320, dtype=npdtype), name='halfpi', layout=f'H1(E), -1, S{x.dtype.size}')
  pi = g.from_data(np.array([np.pi]*320, dtype=npdtype), name='pi', layout=f'H1(E), -1, S{x.dtype.size}')
  threehalfpi = g.from_data(np.array([3*np.pi/2]*320, dtype=npdtype), name='threehalfpi', layout=f'H1(E), -1, S{x.dtype.size}')
  twopi = g.from_data(np.array([2*np.pi]*320, dtype=npdtype), name='twopi', layout=f'H1(E), -1, S{x.dtype.size}')
  if x.dtype == g.float16: mant, exp = 10, 5
  elif x.dtype == g.float32: mant, exp = 23, 8
  mantexpmask = g.from_data(np.array([(1<<(mant+exp))-1]*320, dtype=np.uint32 if x.dtype==g.float32 else np.uint16), name='mantexpmask', layout=f'H1(E), -1, S{x.dtype.size}')
  sgnmask = g.from_data(np.array([1<<(mant+exp)]*320, dtype=np.uint32 if x.dtype==g.float32 else np.uint16), name='sgnmask', layout=f'H1(E), -1, S{x.dtype.size}')
  sgnshift = g.from_data(np.array([mant+exp]*320, dtype=np.uint32 if x.dtype==g.float32 else np.uint16), name='sgnshift', layout=f'H1(E), -1, S{x.dtype.size}')
  with g.ResourceScope(name="quads", is_buffered=True, time=0 if skip_fmod else None, predecessors=[] if skip_fmod else [cos_fmod]) as quads:
    if sinonly: angle, needneg = sin_quads(x, pi, threehalfpi, halfpi, sgnmask, mantexpmask, mantexpmask)
    elif cosonly: angle, needneg = cos_quads(x, pi, twopi, halfpi, mantexpmask, mantexpmask)
    else:
        angle, needneg, neednegsin, anglesin = sincos_quads(x, pi, twopi, halfpi, mantexpmask, mantexpmask, sgnmask, halfpi)
        anglesin = anglesin.write(name='anglesin', layout=f'H1(W), -1, S{x.dtype.size}')
        neednegsin = neednegsin.reinterpret(sgnshift.dtype).write(name='neednegsin', layout=f'H1(W), -1, S{x.dtype.size}')        
    angle = angle.write(name="angle", layout=f'H1(W), -1, S{x.dtype.size}')
    needneg = needneg.reinterpret(sgnshift.dtype).write(name="needneg", layout=f'H1(W), -1, S{x.dtype.size}')
  if x.dtype==g.float16:
      coeffs = [1.0, -0.4995, 0.04102, -0.001132]
  else:
      coeffs = [0.99999994, -0.49999902, 0.041663624, -0.0013854447, 2.3177226e-05]
  c1 = g.from_data(np.array([coeffs[0]]*320, dtype=npdtype), name='c1', layout=f'H1(E), -1, S{x.dtype.size}')
  c2 = g.from_data(np.array([coeffs[1]]*320, dtype=npdtype), name='c2', layout=f'H1(E), -1, S{x.dtype.size}')
  if degree >= 2: c3 = g.from_data(np.array([coeffs[2]]*320, dtype=npdtype), name='c3', layout=f'H1(E), -1, S{x.dtype.size}')
  if degree >= 3: c4 = g.from_data(np.array([coeffs[3]]*320, dtype=npdtype), name='c4', layout=f'H1(E), -1, S{x.dtype.size}')
  if degree >= 4: c5 = g.from_data(np.array([coeffs[4]]*320, dtype=npdtype), name='c5', layout=f'H1(E), -1, S{x.dtype.size}')
  with g.ResourceScope(name="cos_approx", is_buffered=True, time=None, predecessors=[quads]) as cos_approx:
    if not sinonly and not cosonly: angle, needneg = g.stack([angle, anglesin], dim=0), g.stack([needneg, neednegsin], dim=0)
    if x.dtype==g.float16: c = sincos_approx_float16(angle, needneg, c1, c2, c3, c4, sgnshift)
    else: c = sincos_approx_float32(angle, needneg, c1, c2, c3, c4, c5, sgnshift)
    if not sinonly and not cosonly: c, s = [z.reshape(*x.shape) for z in g.split(c, dim=0, num_splits=2)]
    c = c.reinterpret(x.dtype).write(name='sin' if sinonly else 'cos', layout=f'H1(W), -1, S{x.dtype.size}')
    if not sinonly and not cosonly: s = s.reinterpret(x.dtype).write(name='sin', layout=f'H1(W), -1, S{x.dtype.size}')
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
    r = subnormal_to_zero_fp32(r) #Groq chip behavior is to zero input/output subnormals so we compare to same CPU behavior 
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
        oracle[x] = subnormal_to_zero_fp32(oracle[x])
        #assert np.allclose(oracle[x], results[x]), (oracle[x], results[x])
        #print(getrelabserr(r, results[x], oracle[x]))        
        absdiff = np.abs(oracle[x] - results[x])
        relerr = absdiff/np.abs(oracle[x])
        ulperr = np.minimum(absdiff/meps, absdiff/ulp(r))
        print(f"{x}\n", "Max Abs. Acc", np.max(absdiff), "Avg. Abs. ULP", np.average(absdiff),
              "Max Rel. Acc", np.max(relerr), "Avg. Rel. Acc", np.average(relerr),
              "Max ULP", np.max(ulperr), "Av. ULP", np.average(ulperr))
for dtype in (g.float32,): #(g.float32, g.float16)
    for func, skip_fmod in ((sin, True), (sin, False), (cos, True), (cos, False), (sincos, True), (sincos, False)):
        test_sincos(dtype, func, skip_fmod=skip_fmod)


