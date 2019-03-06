import numpy as np
import os
import sys

def choose_quantization_params(min, max):
    qmin = 0
    qmax = 255
    scale = (max - min) / (qmax - qmin)
    initial_zero_point = qmin - min / scale

    # refine for zero_point
    if (initial_zero_point > qmax):
        nudged_zero_point = qmax
    elif (initial_zero_point < qmin):
        nudged_zero_point = qmin
    else:
        nudged_zero_point = round(initial_zero_point)

    results = {}
    results['scale'] = scale
    results['zero_point'] = nudged_zero_point

    return results

def quantize(q_params, src):
    q_val = q_params['zero_point'] + src / q_params['scale']
    q_val = np.clip(q_val, 0.0, 255.0)
    return np.round(q_val)


def quantize_M(real_multiplier):
    assert(real_multiplier > 0 and real_multiplier < 1)
    s = 0
    # M0 in [0.5, 1] see formular (6) in google'paper
    while(real_multiplier < 0.5):
        real_multiplier *= 2
        s +=1
    q = round(real_multiplier * (1 << 31))
    if (q == (1<<31)):
        q /=2
        s -=1
    assert(s >= 0)
    assert(q < 1<<32)
    quantized_multiplier = int(q)
    right_shift = s
    return quantized_multiplier, right_shift + 32 -1

def dequantize(q_params, src):
    q_val = q_params['scale']*(src - q_params['zero_point'])
    return q_val

def interger_matrix_mul(uint8_lhs, uint8_rhs,
                        lhs_q_params, rhs_q_params, result_q_params,
                        quantized_multiplier, right_shift, real_multiplier = None):
    assert (uint8_lhs.shape[1] == uint8_rhs.shape[0])
    rows = uint8_lhs.shape[0]
    depth = uint8_lhs.shape[1]
    cols = uint8_rhs.shape[1]

    assert (rows >0 and depth >0 and cols >0 )

    actual_uint8_result = np.empty([rows, cols])

    # method 1: formular [4] in google's paper
    # for i in range(rows):
    #     for k in range(cols):
    #         tmp_sum = 0
    #         for j in range(depth):
    #             tmp_sum +=  (uint8_lhs[i,j] - lhs_q_params['zero_point']) * (uint8_rhs[j,k] - rhs_q_params['zero_point'])
    #         actual_uint8_result[i,k] = tmp_sum

    # method 2: formular [7], [8] in google's paper
    for i in range(rows):
        for k in range(cols):
            tmp_sum = 0
            for j in range(depth):
                tmp_sum += (uint8_lhs[i,j] * uint8_rhs[j,k])
            a_1 = np.sum(uint8_lhs[i,:])
            a_2 = np.sum(uint8_rhs[:,k])
            actual_uint8_result[i,k] = depth * lhs_q_params['zero_point'] * rhs_q_params['zero_point'] - \
                                    lhs_q_params['zero_point'] * a_2 - rhs_q_params['zero_point'] * a_1 + tmp_sum

    M = float(quantized_multiplier) / (1<<int(right_shift))
    if real_multiplier is not None:
        print ('diffence of M : ', (M - real_multiplier))
    actual_uint8_result = actual_uint8_result * M  + result_q_params['zero_point']

    return np.round(actual_uint8_result)

def main():

    # q with 8bit, multiplier use 32 bit

    rows = 2
    depth = 4
    cols = 3

    float_lhs = np.random.uniform(-1, 1, rows * depth).reshape([rows, depth])
    float_rhs = np.random.uniform(-1, 1, depth * cols). reshape([depth, cols])
    reference_float_result = np.matmul(float_lhs, float_rhs)

    print ('left matrix:\n ', float_lhs)
    print ('right matrix:\n ', float_rhs)
    print ('gt results:\n ', reference_float_result)

    #######  THIS PART SHOULD BE DONE OFFLINE

    lhs_min, lhs_max = np.min(float_lhs), np.max(float_lhs)
    rhs_min, rhs_max = np.min(float_rhs), np.max(float_rhs)
    result_min, result_max = np.min(reference_float_result), np.max(reference_float_result)

    lhs_q_params = choose_quantization_params(lhs_min, lhs_max)
    rhs_q_params = choose_quantization_params(rhs_min, rhs_max)
    result_q_params = choose_quantization_params(result_min, result_max)

    print('for lhs: \n, min:{},max:{},scale:{},zero_point:{}'.format
                            (lhs_min, lhs_max, lhs_q_params['scale'], lhs_q_params['zero_point']))
    print('for rhs: \n, min:{},max:{},scale:{},zero_point:{}'.format
                            (rhs_min, rhs_max, rhs_q_params['scale'], rhs_q_params['zero_point']))
    print('for results: \n, min:{},max:{},scale:{},zero_point:{}'.format
                            (result_min, result_max, result_q_params['scale'], result_q_params['zero_point']))

    uint8_lhs = quantize(lhs_q_params, float_lhs)
    uint8_rhs = quantize(rhs_q_params, float_rhs)

    print('quantized uint8 lhs: \n', uint8_lhs)
    print('quantized uint8 rhs: \n', uint8_rhs)

    print ('dequantize from uint8_lhs:\n', dequantize(lhs_q_params, uint8_lhs))
    print ('dequantize from uint8_rhs:\n', dequantize(rhs_q_params, uint8_rhs))

    lhs_offset = -lhs_q_params['zero_point']
    rhs_offset = -rhs_q_params['zero_point']
    result_offset = result_q_params['zero_point']

    # formular (5),(6) in goole's paper
    real_multiplier = (lhs_q_params['scale'] * rhs_q_params['scale']) / result_q_params['scale']
    quantized_multiplier, right_shift = quantize_M(real_multiplier)
    print('quantized_multiplier: ', quantized_multiplier)
    print('right_shift: ', right_shift)

    #######  THIS PART SHOULD BE DONE OFFLINE

    ###### ON DEVICE: USE INTER TO COMPUTE FLOAT

    actual_uint8_result = interger_matrix_mul(uint8_lhs, uint8_rhs,
                                            lhs_q_params, rhs_q_params, result_q_params,
                                            quantized_multiplier, right_shift, real_multiplier)

    ###### ON DEVICE: USE INTER TO COMPUTE FLOAT

    print ('actual uint8 result \n', actual_uint8_result)
    actual_float_result = dequantize(result_q_params, actual_uint8_result)
    print ('actual_float_results: \n', actual_float_result)
    print ('diff with gt: \n', actual_float_result - reference_float_result)

if __name__ == '__main__':
    main()
