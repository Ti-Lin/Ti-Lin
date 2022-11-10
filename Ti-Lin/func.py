import numpy as np
def func(i,weights, biases, out_shape,  pads, strides,  LBs, UBs,aus,als,alpha_u,alpha_l):
    A_u = weights[i+2].reshape((1, 1, weights[i+2].shape[0], weights[i+2].shape[1], weights[i+2].shape[2], weights[i+2].shape[3]))*np.ones((out_shape[0], out_shape[1], weights[i+2].shape[0], weights[i+2].shape[1], weights[i+2].shape[2], weights[i+2].shape[3]), dtype=np.float32)
    B_u = biases[i+2]*np.ones((out_shape[0], out_shape[1], out_shape[2]), dtype=np.float32)
    A_l = A_u.copy()
    B_l = B_u.copy()
    pad = pads[i+2]
    stride = strides[i+2]
    beta_u= np.zeros(alpha_u.shape, dtype=np.float32)
    beta_l= np.zeros(alpha_u.shape, dtype=np.float32)
    A_u, B_u = UL_conv_bound(A_u, B_u, np.asarray(pad), np.asarray(stride), np.asarray(UBs[i+1].shape), weights[i+1], biases[i+1], np.asarray(pads[i+1]), np.asarray(strides[i+1]), np.asarray(UBs[i].shape))
    A_l, B_l = UL_conv_bound(A_l, B_l, np.asarray(pad), np.asarray(stride), np.asarray(LBs[i+1].shape), weights[i+1], biases[i+1], np.asarray(pads[i+1]), np.asarray(strides[i+1]), np.asarray(LBs[i].shape))
    pad = (strides[i+1][0]*pad[0]+pads[i+1][0], strides[i+1][0]*pad[1]+pads[i+1][1], strides[i+1][1]*pad[2]+pads[i+1][2], strides[i+1][1]*pad[3]+pads[i+1][3])
    stride = (strides[i+1][0]*stride[0], strides[i+1][1]*stride[1])
    A_u, B_u = UL_relu_bound(A_u, B_u, np.asarray(pad), np.asarray(stride), alpha_u, alpha_l, beta_u, beta_l)
    A_l, B_l = UL_relu_bound(A_l, B_l, np.asarray(pad), np.asarray(stride), alpha_l, alpha_u, beta_l, beta_u)
    A_u, B_u = UL_relu_bound(A_u, B_u, np.asarray(pad), np.asarray(stride), aus[-1],als[-1], 0,0)
    A_l, B_l = UL_relu_bound(A_l, B_l, np.asarray(pad), np.asarray(stride), als[-1],aus[-1], 0,0)

    return A_u,A_l

def dfunc(A_u,A_l,alpha_u,alpha_l):
    dau = np.zeros(A_u.shape, dtype=np.float32)
    dal = np.zeros(A_l.shape, dtype=np.float32)
    learning_rate=0.01
    for t in range(10):#梯度下降十次
        for i in range(alpha_u.shape[0]):
            for j in range(alpha_u.shape[1]):
                for k in range(alpha_u.shape[2]):
                    alpha_ud= np.zeros(alpha_u.shape, dtype=np.float32)
                    alpha_ld= np.zeros(alpha_u.shape, dtype=np.float32)
                    alpha_ud[i,j,k]=1
                    temp_u,temp_l=func(i,weights, biases, out_shape,  pads, strides,  LBs, UBs,aus,als,alpha_ud,alpha_ld)#梯度
                    alpha_u[i,j,k]=alpha_u[i,j,k]-temp_u[i,j,k].sum()*learning_rate-temp_l[i,j,k].sum()*learning_rate
                    alpha_ld[i,j,k]=1
                    alpha_ud[i,j,k]=0
                    temp_u,temp_l=func(i,weights, biases, out_shape,  pads, strides,  LBs, UBs,aus,als,alpha_ud,alpha_ld)
                    alpha_l[i,j,k]=alpha_l[i,j,k]-temp_u[i,j,k].sum()*learning_rate-temp_l[i,j,k].sum()*learning_rate
    return alpha_ud,alpha_ld

def linear_bound_last_2_atan(alpha_u,alpha_l,LB, UB):
    beta_u = np.zeros(LB.shape, dtype=np.float32)
    beta_l = np.zeros(LB.shape, dtype=np.float32)
    for i in range(LB.shape[0]):
        for j in range(LB.shape[1]):
            for k in range(LB.shape[2]):
                act = atan
                actd = atand
                actid = atanid
                actut = atanut
                actlt = atanlt
                ## General (Sigmoid-like functions)
                if UB[i,j,k] < LB[i,j,k]: 
                	temp=LB[i,j,k]
                	LB[i,j,k]=UB[i,j,k]
                	UB[i,j,k]=temp
                elif UB[i,j,k]-LB[i,j,k]<0.001:
                    alpha_u[i,j,k] = actd(UB[i,j,k])
                    alpha_l[i,j,k] = actd(LB[i,j,k])
                    beta_u[i,j,k] = act(UB[i,j,k])-actd(UB[i,j,k])*UB[i,j,k]
                    beta_l[i,j,k] = act(LB[i,j,k])-actd(LB[i,j,k])*LB[i,j,k]
                else:
                    kk = (act(UB[i, j, k]) - act(LB[i, j, k])) / (UB[i, j, k] - LB[i, j, k])
                    ld=actd(LB[i, j, k])
                    ud=actd(UB[i, j, k])
                    if kk>ld and kk<ud:
                        if alpha_u[i,j,k]<=kk:
                            beta_u[i, j, k] = act(UB[i, j, k]) - alpha_u[i, j, k] * UB[i, j, k]
                        else: 
                            beta_u[i, j, k] = act(LB[i, j, k]) - alpha_u[i, j, k] * LB[i, j, k]
                        alpha_l[i, j, k] = kk# 这里是解方程，方程为0的解，记得帮忙转成float的格式
                        d=actlo(LB[i, j, k],UB[i, j, k])
                        beta_l[i, j, k] = act(d) - alpha_l[i, j, k] * d
                    elif ld>kk and ud<kk:
                        if alpha_u[i,j,k]<=kk:
                            beta_u[i, j, k] = act(UB[i, j, k]) - alpha_u[i, j, k] * UB[i, j, k]
                        else: 
                            beta_u[i, j, k] = act(LB[i, j, k]) - alpha_u[i, j, k] * LB[i, j, k]                        
                        dd=actuo(LB[i, j, k],UB[i, j, k])
                        alpha_l[i, j, k] = kk# 这里是解方程，方程为0的解，记得帮忙转成float的格式
                        beta_l[i, j, k] = act(LB[i, j, k]) - alpha_l[i, j, k] * LB[i, j, k]
                    else:
                        d=actut(LB[i, j, k],UB[i, j, k])#q
                        dd=actlt(LB[i, j, k],UB[i, j, k])#qq
                        alpha_u[i, j, k] = actd(d) # 这里是解方程，方程为0的解，记得帮忙转成float的格式
                        beta_u[i, j, k] = act(d) - alpha_u[i, j, k] * d
                        alpha_l[i, j, k] = actd(dd)# 这里是解方程，方程为0的解，记得帮忙转成float的格式
                        beta_l[i, j, k] = act(dd) - alpha_l[i, j, k] * dd

    return alpha_u, alpha_l,beta_u, beta_l    

def func2(i,weights, biases, out_shape,  pads, strides,  LBs, UBs,aus,als,alpha_u,alpha_l,beta_u, beta_l):
    A_u = weights[i+2].reshape((1, 1, weights[i+2].shape[0], weights[i+2].shape[1], weights[i+2].shape[2], weights[i+2].shape[3]))*np.ones((out_shape[0], out_shape[1], weights[i+2].shape[0], weights[i+2].shape[1], weights[i+2].shape[2], weights[i+2].shape[3]), dtype=np.float32)
    B_u = biases[i+2]*np.ones((out_shape[0], out_shape[1], out_shape[2]), dtype=np.float32)
    A_l = A_u.copy()
    B_l = B_u.copy()
    pad = pads[i+2]
    stride = strides[i+2]

    A_u, B_u = UL_conv_bound(A_u, B_u, np.asarray(pad), np.asarray(stride), np.asarray(UBs[i+1].shape), weights[i+1], biases[i+1], np.asarray(pads[i+1]), np.asarray(strides[i+1]), np.asarray(UBs[i].shape))
    A_l, B_l = UL_conv_bound(A_l, B_l, np.asarray(pad), np.asarray(stride), np.asarray(LBs[i+1].shape), weights[i+1], biases[i+1], np.asarray(pads[i+1]), np.asarray(strides[i+1]), np.asarray(LBs[i].shape))
    pad = (strides[i+1][0]*pad[0]+pads[i+1][0], strides[i+1][0]*pad[1]+pads[i+1][1], strides[i+1][1]*pad[2]+pads[i+1][2], strides[i+1][1]*pad[3]+pads[i+1][3])
    stride = (strides[i+1][0]*stride[0], strides[i+1][1]*stride[1])
    A_u, B_u = UL_relu_bound(A_u, B_u, np.asarray(pad), np.asarray(stride), alpha_u, alpha_l, beta_u, beta_l)
    A_l, B_l = UL_relu_bound(A_l, B_l, np.asarray(pad), np.asarray(stride), alpha_l, alpha_u, beta_l, beta_u)
    A_u, B_u = UL_relu_bound(A_u, B_u, np.asarray(pad), np.asarray(stride), aus[-1],als[-1], bus[-1],bls[-1])
    A_l, B_l = UL_relu_bound(A_l, B_l, np.asarray(pad), np.asarray(stride), als[-1],aus[-1], bls[-1],bus[-1])

    return A_u,A_l,B_u,B_l
