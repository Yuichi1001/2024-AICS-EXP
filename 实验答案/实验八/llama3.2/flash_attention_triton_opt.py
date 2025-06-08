"""
Fused Attention
===============
 
This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)
Credits: OpenAI kernel team
 
Extra Credits:
- Original flash attention paper (https://arxiv.org/abs/2205.14135)
- Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)
 
"""
 
import torch
import torch_mlu
 
import triton
import triton.language as tl
#from genesis.Python.Test.Common.utils import reset_tmp_dir
import time
import numpy as np
 
@triton.jit
def _attn_fwd_inner(
        acc,
        l_i,
        m_i,
        q,  #
        K_block_ptr,
        V_block_ptr,  #
        start_m,
        qk_scale,  #
        BLOCK_M: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        BLOCK_N: tl.constexpr,  #
        STAGE: tl.constexpr,
        offs_m: tl.constexpr,
        offs_n: tl.constexpr,  #
        N_CTX: tl.constexpr,
        IS_DIVISIBLE: tl.constexpr):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    K_block_ptr = tl.advance(K_block_ptr, (lo, 0))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        if IS_DIVISIBLE:
            k = tl.load(K_block_ptr)
        else:
            k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        qk = tl.zeros([BLOCK_N, BLOCK_M], dtype=tl.float32)
        qk += tl.dot(k, q)
        if STAGE == 2:
            mask = offs_m[None, :] >= (start_n + offs_n[:, None])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 0))
            qk -= m_ij[None, :]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk * qk_scale, 0))
            qk = qk * qk_scale - m_ij[None, :]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 0)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        if IS_DIVISIBLE:
            v = tl.load(V_block_ptr)
        else:
            v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        qk_wram = p.to(tl.float16)
        
        qkv = tl.dot(tl.trans(v), qk_wram)
        # -- update output accumulator --
        acc = acc * alpha[None, :]
        acc += qkv
        # update m_i and l_i
        m_i = m_ij
        l_i = l_i * alpha + l_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
 
    return acc, l_i, m_i
 
 
@triton.jit
def _attn_eff_fwd_inner(
        acc,
        l_i,
        m_i,
        q,  #
        K_block_ptr,
        V_block_ptr,  #
        start_m,
        qk_scale,  #
        Mask_block_ptr,
        BLOCK_M: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        BLOCK_N: tl.constexpr,  #
        STAGE: tl.constexpr,
        offs_m: tl.constexpr,
        offs_n: tl.constexpr,  #
        N_CTX: tl.constexpr,
        IS_DIVISIBLE: tl.constexpr,):
        
    # causal = True
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    K_block_ptr = tl.advance(K_block_ptr, (lo, 0))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    Mask_block_ptr = tl.advance(Mask_block_ptr, (lo, 0))
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        if IS_DIVISIBLE:
            k = tl.load(K_block_ptr)
            mask = tl.load(Mask_block_ptr)
        else:
            k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
            mask = tl.load(Mask_block_ptr, boundary_check=(0, 1), padding_option="zero")
        qk = tl.zeros([BLOCK_N, BLOCK_M], dtype=tl.float32)
        qk += tl.dot(k, q)
        
        qk = qk * qk_scale + mask*1.44269504
        m_ij = tl.maximum(m_i, tl.max(qk, 0))
        qk -= m_ij[None, :]
            
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 0)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        if IS_DIVISIBLE:
            v = tl.load(V_block_ptr)
        else:
            v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        qk_wram = p.to(tl.float16)
        
        qkv = tl.dot(tl.trans(v), qk_wram)  # [BLOCK_DMODEL, BLOCK_N] × [BLOCK_N, BLOCK_M] = [BLOCK_DMODEL, BLOCK_M]
        
        # -- update output accumulator --
        acc = acc * alpha[None, :]
        acc += qkv
        # update m_i and l_i
        m_i = m_ij
        l_i = l_i * alpha + l_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
        Mask_block_ptr = tl.advance(Mask_block_ptr, (BLOCK_N, 0))
    return acc, l_i, m_i 

 
@triton.jit
def _attn_eff_fwd(
        Q,
        K,
        V,
        sm_scale,
        M,
        Out,  #
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,  #
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,  #
        stride_vz,
        stride_vh,
        stride_vk,
        stride_vn,  #
        stride_oz,
        stride_oh,
        stride_om,
        stride_on,  #
        stride_mm,
        stride_mn,  #
        Z,
        H,  #
        causal_mask,
        #TODO：指定以下类型为编译时常量
        N_CTX: tl.constexpr,  #
        Q_N_CTX: tl.constexpr,
        BLOCK_M: tl.constexpr,  #
        BLOCK_N: tl.constexpr,  #
        BLOCK_DMODEL: tl.constexpr,  #
        STAGE: tl.constexpr,  #
        IS_DIVISIBLE: tl.constexpr):
    
    #TODO: 获取当前核心的唯一标识符，用于区分不同的计算核心
    core_id = tl.program_id(0)
    #TODO: 获取在第一个维度（核心维度）上的核心总数，用于并行计算
    core_dim = tl.num_programs(0)
    #TODO: 获取当前集群的唯一标识符，用于区分不同的计算集群
    cluster_id = tl.program_id(1)
    #TODO: 获取在第二个维度（集群维度）上的核心总数，以便进行任务分配和调度
    cluster_dim = tl.num_programs(1)

    #TODO: 计算每个上下文的数量
    context_num = tl.cdiv(Q_N_CTX, BLOCK_M)
    #TODO: 计算总的注意力头数量
    total_heads = Z * H * context_num
    #TODO: 每个集群分配的头数量
    task_heads = total_heads // cluster_dim
    #TODO: 计算剩余的头数（总头数减去每个集群处理的头数的乘积）
    task_remain_heads = total_heads - task_heads * cluster_dim
    #TODO: 保证每个集群处理的任务数量至少为1
    task_heads += 1 if cluster_id < task_remain_heads else 0
    #TODO: 计算当前集群开始处理的头的索引
    task_head_begin = cluster_id * (task_heads - (1 if cluster_id < task_remain_heads else 0)) + min(cluster_id, task_remain_heads)
    if cluster_id >= task_remain_heads:
        #TODO: 减少当前集群的任务数量
        task_heads -= 1
        #TODO:  更新当前集群开始处理的头的索引，以便正确定位到可处理的头
        task_head_begin = cluster_id * task_heads + task_remain_heads
    if task_heads <= 0:
        return
 
    #TODO: 计算每个核心需要处理的头的数量
    core_heads = task_heads // core_dim
    #TODO: 计算剩余的头数
    core_remain_heads = task_heads - core_heads * core_dim
    #TODO: 保证每个核心处理的任务数量至少为1
    core_heads += 1 if core_id < core_remain_heads else 0
    #TODO: 计算当前核心开始处理的头的索引
    core_head_begin = core_id * (core_heads - (1 if core_id < core_remain_heads else 0)) + min(core_id, core_remain_heads)
    if core_id >= core_remain_heads:
        #TODO: 减少当前核心的任务数量
        core_heads -= 1
        #TODO: 更新当前核心开始处理的头的索引，以便正确定位到可处理的头
        core_head_begin = core_id * core_heads + core_remain_heads
    if core_heads <= 0:
        return
    #TODO: 计算实际处理的头的起始索引
    head_begin = task_head_begin + core_head_begin
    #TODO: 计算实际处理的头的结束索引
    head_end = head_begin + core_heads
 
    for head_idx in range(head_begin, head_end):  # 一个core处理 q上的 (Q_N_CTX/BLOCK_M) 数据
        #TODO: 计算当前头的起始索引在上下文中的位置
        start_m = head_idx % context_num
        #TODO: 计算当前头在上下文中的偏移量
        off_hz = head_idx // context_num
        #TODO: 计算当前头的 z 维度偏移
        off_z = off_hz // H
        #TODO: 计算当前头的 h 维度偏移
        off_h = off_hz % H
        #TODO: 将 off_z 和 off_h 转换为 int64 类型，并分别乘以查询张量的步幅 stride_qz 和 stride_qh，以得到查询张量在内存中的实际位置
        q_offset = (off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh)
        #TODO: 将 off_z 和 off_h 转换为 int64 类型，并分别乘以键值张量的步幅 stride_kz 和 stride_kh，以得到键值张量在内存中的实际位置
        kv_offset = (off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh)
        # block pointers
        Q_block_ptr = tl.make_block_ptr(
            base=Q + q_offset,
            shape=(BLOCK_DMODEL, Q_N_CTX),          
            strides=(stride_qk, stride_qm),
            offsets=(0, start_m * BLOCK_M),
            block_shape=(BLOCK_DMODEL, BLOCK_M),
            order=(0, 1),
        )
        #TODO: 仿照以上Q_block_ptr的创建方法，创建K_block_ptr，指定基地址和形状。
        K_block_ptr = tl.make_block_ptr(
            base=K + kv_offset,
            shape=(N_CTX, BLOCK_DMODEL),
            strides=(stride_kn, stride_kk),
            offsets=(0, 0),
            block_shape=(BLOCK_N, BLOCK_DMODEL),
            order=(1, 0)
        )
        
        #TODO: 仿照以上Q_block_ptr的创建方法，创建V_block_ptr，指定基地址和形状。
        V_block_ptr = tl.make_block_ptr(
            base=V + kv_offset,
            shape=(N_CTX, BLOCK_DMODEL),
            strides=(stride_vk, stride_vn),
            offsets=(0, 0),
            block_shape=(BLOCK_N, BLOCK_DMODEL),
            order=(1, 0)
        )
     
        #TODO: 仿照以上Q_block_ptr的创建方法，创建Mask_block_ptr，指定基地址和形状。
        Mask_block_ptr = tl.make_block_ptr(
            base=causal_mask + off_z.to(tl.int64) * stride_mm * Q_N_CTX * N_CTX,
            shape=(N_CTX, Q_N_CTX),
            strides=(stride_mn, stride_mm),
            offsets=(0, start_m * BLOCK_M),
            block_shape=(BLOCK_N, BLOCK_M),
            order=(1, 0)
        )

        #TODO: 仿照以上Q_block_ptr的创建方法，创建O_block_ptr，指定基地址和形状。
        O_block_ptr = tl.make_block_ptr(
            base=Out + q_offset,
            shape=(Q_N_CTX, BLOCK_DMODEL),
            strides=(stride_om, stride_on),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_DMODEL),
            order=(1, 0)
        )

        # initialize offsets
        #TODO: 计算当前块的行偏移量，offs_m 为从 start_m 开始的连续 BLOCK_M 行的索引
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        #TODO: 创建 offs_n，表示当前块的所有列索引，从 0 到 BLOCK_N - 1。
        offs_n = tl.arange(0, BLOCK_N)
        # initialize pointer to m and l
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_DMODEL, BLOCK_M], dtype=tl.float32)
        # load scales
        qk_scale = sm_scale
        qk_scale *= 1.44269504  # 1/ln(2)
        # load q: it will stay in SRAM throughout
        if IS_DIVISIBLE:
            #TODO:从 Q_block_ptr 加载数据到 q
            q = tl.load(Q_block_ptr)
        else:
            #TODO:从 Q_block_ptr 加载数据到 q
            q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        # stage 1: off-band
        # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
        # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
        if STAGE & 1:
            #TODO:调用_attn_eff_fwd_inner函数处理非对角线的部分计算
            acc, l_i, m_i = _attn_eff_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr, start_m, qk_scale, Mask_block_ptr, BLOCK_M, BLOCK_DMODEL, BLOCK_N, 1, offs_m, offs_n, N_CTX, IS_DIVISIBLE)
        # stage 2: on-band
        # For causal = True, STAGE = 3 and _attn_fwd_inner gets 2 as its STAGE
        if STAGE & 2:
            # barrier makes it easier for compielr to schedule the
            # two loops independently   
            #TODO: 同步线程
            tl.debug_barrier()
            #TODO:调用__attn_eff_fwd_inner函数处理对角线的部分计算
            acc, l_i, m_i = _attn_eff_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr, start_m, qk_scale, Mask_block_ptr, BLOCK_M, BLOCK_DMODEL, BLOCK_N, 2, offs_m, offs_n, N_CTX, IS_DIVISIBLE)
        # epilogue
        #TODO:计算 log2(l_i) 并加到 m_i
        m_i += tl.math.log2(l_i)
        #TODO: 计算 l_i 的倒数
        l_i_recip = 1.0 / l_i
        #TODO: 将 acc 矩阵的每个元素乘以 l_i_recip 的每一列
        acc = acc * l_i_recip[None, :]
        #TODO: 对 acc 矩阵进行转置操作，以便适应后续存储或计算
        acc = tl.trans(acc)
        #TODO: 计算 m_ptrs，作为 M 矩阵中的指针，结合 off_hz 和 offs_m。
        m_ptrs = M + off_hz * Q_N_CTX + offs_m
        if IS_DIVISIBLE:
            #TODO: 将当前的m_i值存储到m_ptrs指向的位置
            tl.store(m_ptrs, m_i)
            #TODO: # 将累加结果acc转换为输出类型并存储到O_block_ptr指向的位置
            tl.store(O_block_ptr, acc.to(Out.type.element_ty))
        else:
            #TODO: 仅在offs_m小于N_CTX的情况下，将m_i存储到m_ptrs，应用掩码
            tl.store(m_ptrs, m_i, mask=offs_m < Q_N_CTX)
            #TODO:  # 将累加结果acc转换为输出类型并存储到O_block_ptr，进行边界检查
            tl.store(O_block_ptr, acc.to(Out.type.element_ty), boundary_check=(0, 1))
 
@triton.jit
def _attn_fwd(
        Q,
        K,
        V,
        sm_scale,
        M,
        Out,  #
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,  #
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,  #
        stride_vz,
        stride_vh,
        stride_vk,
        stride_vn,  #
        stride_oz,
        stride_oh,
        stride_om,
        stride_on,  #
        Z,
        H,  #
        #TODO：指定以下类型为编译时常量
        N_CTX: tl.constexpr,  #
        Q_N_CTX: tl.constexpr,
        BLOCK_M: tl.constexpr,  #
        BLOCK_N: tl.constexpr,  #
        BLOCK_DMODEL: tl.constexpr,  #
        STAGE: tl.constexpr,  #
        IS_DIVISIBLE: tl.constexpr):
    #TODO: 获取当前核心的唯一标识符，用于区分不同的计算核心
    core_id = tl.program_id(0)
    #TODO: 获取在第一个维度（核心维度）上的核心总数，用于并行计算
    core_dim = tl.num_programs(0)
    #TODO: 获取当前集群的唯一标识符，用于区分不同的计算集群
    cluster_id = tl.program_id(1)
    #TODO: 获取在第二个维度（集群维度）上的核心总数，以便进行任务分配和调度
    cluster_dim = tl.num_programs(1)
 
    #TODO:计算Q_N_CTX与BLOCK_M的整除结果，得到上下文的块数量
    context_num = tl.cdiv(Q_N_CTX, BLOCK_M) # 向上取整
    #TODO: 计算总的注意力头数量
    total_heads = Z * H * context_num
    #TODO: 每个集群分配的头数量
    task_heads = total_heads // cluster_dim
    #TODO: 计算剩余的头数（总头数减去每个集群处理的头数的乘积）
    task_remain_heads = total_heads - task_heads * cluster_dim
    #TODO: 保证每个集群处理的任务数量至少为1
    task_heads += 1 if cluster_id < task_remain_heads else 0
    #TODO: 计算当前集群开始处理的头的索引
    task_head_begin = cluster_id * (task_heads - (1 if cluster_id < task_remain_heads else 0)) + min(cluster_id, task_remain_heads)
    if cluster_id >= task_remain_heads:
        #TODO: 减少当前集群的任务数量
        task_heads -= 1
        #TODO:  更新当前集群开始处理的头的索引，以便正确定位到可处理的头
        task_head_begin = cluster_id * task_heads + task_remain_heads
    else:
        pass
    if task_heads <= 0:
        return
 
    #TODO: 计算每个核心需要处理的头的数量
    core_heads = task_heads // core_dim
    #TODO: 计算剩余的头数
    core_remain_heads = task_heads - core_heads * core_dim
    #TODO: 保证每个核心处理的任务数量至少为1
    core_heads += 1 if core_id < core_remain_heads else 0
    #TODO: 计算当前核心开始处理的头的索引
    core_head_begin = core_id * (core_heads - (1 if core_id < core_remain_heads else 0)) + min(core_id, core_remain_heads)
    if core_id >= core_remain_heads:
        #TODO: 减少当前核心的任务数量
        core_heads -= 1
        #TODO: 更新当前核心开始处理的头的索引，以便正确定位到可处理的头
        core_head_begin = core_id * core_heads + core_remain_heads
    if core_heads <= 0:
        return
    #TODO: 计算实际处理的头的起始索引
    head_begin = task_head_begin + core_head_begin
    #TODO: 计算实际处理的头的结束索引
    head_end = head_begin + core_heads
 
    for head_idx in range(head_begin, head_end):
        #TODO: 计算当前头的起始索引在上下文中的位置
        start_m = head_idx % context_num
        #TODO: 计算当前头在上下文中的偏移量
        off_hz = head_idx // context_num

        #TODO: 计算当前头的 z 维度偏移
        off_z = off_hz // H
        #TODO: 计算当前头的 h 维度偏移
        off_h = off_hz % H
        #TODO: 计算查询（Q）的内存偏移量，基于z和h维度的偏移量及其步幅
        q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
        #TODO: 计算KV的内存偏移量，基于z和h维度的偏移量及其步幅
        kv_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
        # block pointers
        Q_block_ptr = tl.make_block_ptr(
            base=Q + q_offset,
            shape=(BLOCK_DMODEL, Q_N_CTX),          
            strides=(stride_qk, stride_qm),
            offsets=(0, start_m * BLOCK_M),
            block_shape=(BLOCK_DMODEL, BLOCK_M),
            order=(0, 1),
        )
        #TODO: #TODO:仿照以上Q_block_ptr的创建方法，创建K_block_ptr
        K_block_ptr = tl.make_block_ptr(
            base=K + kv_offset,
            shape=(N_CTX, BLOCK_DMODEL),
            strides=(stride_kn, stride_kk),
            offsets=(0, 0),
            block_shape=(BLOCK_N, BLOCK_DMODEL),
            order=(1, 0)
        )

        #TODO: #TODO:仿照以上Q_block_ptr的创建方法，创建V_block_ptr
        V_block_ptr = tl.make_block_ptr(
            base=V + kv_offset,
            shape=(N_CTX, BLOCK_DMODEL),
            strides=(stride_vk, stride_vn),
            offsets=(0, 0),
            block_shape=(BLOCK_N, BLOCK_DMODEL),
            order=(1, 0)
        )
        
        #TODO: #TODO:仿照以上Q_block_ptr的创建方法，创建O_block_ptr
        O_block_ptr = tl.make_block_ptr(
            base=Out + q_offset,
            shape=(Q_N_CTX, BLOCK_DMODEL),
            strides=(stride_om, stride_on),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_DMODEL),
            order=(1, 0)
        )

        # initialize offsets
        #TODO: 计算当前块的行偏移量，offs_m 为从 start_m 开始的连续 BLOCK_M 行的索引
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        #TODO: 创建 offs_n，表示当前块的所有列索引，从 0 到 BLOCK_N - 1。
        offs_n = tl.arange(0, BLOCK_N)
        # initialize pointer to m and l
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_DMODEL, BLOCK_M], dtype=tl.float32)
        # load scales
        qk_scale = sm_scale
        qk_scale *= 1.44269504  # 1/log(2)
        # load q: it will stay in SRAM throughout
        if IS_DIVISIBLE:
            #TODO:从 Q_block_ptr 加载数据到 q
            q = tl.load(Q_block_ptr)
        else:
            #TODO:从 Q_block_ptr 加载数据到 q
            q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        # stage 1: off-band
        # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
        # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
        if STAGE & 1:
            #TODO:调用_attn_fwd_inner函数处理非对角线的部分计算
            acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr, start_m, qk_scale, BLOCK_M, BLOCK_DMODEL, BLOCK_N, 1, offs_m, offs_n, N_CTX, IS_DIVISIBLE)
        # stage 2: on-band
        if STAGE & 2:
            # barrier makes it easier for compielr to schedule the
            # two loops independently
            #TODO: 同步线程
            tl.debug_barrier()
            #TODO:调用_attn_fwd_inner函数处理对角线的部分计算
            acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr, start_m, qk_scale, BLOCK_M, BLOCK_DMODEL, BLOCK_N, 2, offs_m, offs_n, N_CTX, IS_DIVISIBLE)
        # epilogue
        #TODO:计算 log2(l_i) 并加到 m_i
        m_i += tl.math.log2(l_i)
        #TODO: 计算 l_i 的倒数
        l_i_recip = 1.0 / l_i
        #TODO: 将 acc 矩阵的每个元素乘以 l_i_recip 的每一列
        acc = acc * l_i_recip[None, :]
        #TODO: 对 acc 矩阵进行转置操作，以便适应后续存储或计算
        acc = tl.trans(acc)
        #TODO: 计算 m_ptrs，作为 M 矩阵中的指针，结合 off_hz 和 offs_m。
        m_ptrs = M + off_hz * Q_N_CTX + offs_m
        if IS_DIVISIBLE:
            #TODO: 将当前的m_i值存储到m_ptrs指向的位置
            tl.store(m_ptrs, m_i)
            #TODO: # 将累加结果acc转换为输出类型并存储到O_block_ptr指向的位置
            tl.store(O_block_ptr, acc.to(Out.type.element_ty))
        else:
            #TODO: 仅在offs_m小于N_CTX的情况下，将m_i存储到m_ptrs，应用掩码
            tl.store(m_ptrs, m_i, mask=offs_m < Q_N_CTX)
            #TODO:  # 将累加结果acc转换为输出类型并存储到O_block_ptr，进行边界检查
            tl.store(O_block_ptr, acc.to(Out.type.element_ty), boundary_check=(0, 1))

@triton.jit
def _attn_bwd_preprocess(
        O,
        DO,  #
        Delta,  #
        Z,
        H,
        N_CTX,  #
        #TODO： 指定以下类型为编译时常量
        BLOCK_M: tl.constexpr,
        D_HEAD: tl.constexpr  #
):
    #TODO: 计算当前程序实例在 M 维度上的偏移量
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    #TODO: 获取当前程序的第二个维度的 ID，用于头的分配
    off_hz = tl.program_id(1)
    #TODO:  计算每个注意力头的维度上的偏移量off_n，即从 0 到 D_HEAD-1 的范围
    off_n = tl.arange(0, D_HEAD)
    # load
    o = tl.load(O + off_hz * D_HEAD * N_CTX + off_m[:, None] * D_HEAD +
                off_n[None, :])
    #TODO:# 同样计算偏移量并将 DO 中的数据加载到do,并将数据类型转换为 float32
    do = tl.load(DO + off_hz * D_HEAD * N_CTX + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    #TODO: 计算 delta，它是 o 和 do的逐元素乘积的和
    delta = tl.sum(o * do, axis=1)
    # write-back
    #TODO: 计算得到的 delta 存储到 Delta 张量中的特定位置
    tl.store(Delta + off_hz * N_CTX + off_m, delta)
 
 
# The main inner-loop logic for computing dK and dV.
@triton.jit
def _attn_bwd_dkdv(
        dk,
        dv,  #
        Q,
        k,
        v,
        sm_scale,  #
        DO,  #
        M,
        D,  #
        stride_tok,
        stride_d,  #
        H,
        N_CTX,
        #TODO： 指定以下类型为编译时常量
        BLOCK_M1: tl.constexpr,  #
        #TODO： 指定以下类型为编译时常量
        BLOCK_N1: tl.constexpr,  #
        #TODO： 指定以下类型为编译时常量
        BLOCK_DMODEL: tl.constexpr,  #
        start_n,
        start_m,
        num_steps,  #
        #TODO： 指定以下类型为编译时常量
        MASK: tl.constexpr):
    #TODO: 计算 offs_m，表示当前块内 m 维度上的偏移量
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    #TODO: 计算 offs_n，表示当前块内 n 维度上的偏移量
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    #TODO: 计算 offs_k，用于在模型维度上进行计算和索引
    offs_k = tl.arange(0, BLOCK_DMODEL)
    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    #TODO: 计算 DO 的指针位置
    do_ptrs = DO + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    #TODO:  静态断言：BLOCK_N1 必须是 BLOCK_M1 的倍数
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    for blk_idx in range(num_steps):
        #TODO:加载qT_ptrs
        qT = tl.load(qT_ptrs)
        # Load m before computing qk to reduce pipeline stall.
        #TODO： 计算当前块 m 维度上的偏移量
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        #  # 从内存中加载 M 中偏移 offs_m 处的数据，用于后续计算
        m = tl.load(M + offs_m)
        #TODO: 计算 k 和 qT 的点积
        qkT = tl.dot(k, qT)
        #TODO:应用指数函数，将 qkT 和 m 的差值转换为以2为底的指数形式
        pT = tl.math.exp2(qkT - m[None, :])
        # Autoregressive masking.
        if MASK:
            mask = (offs_m[None, :] >= offs_n[:, None])
            #TODO: 应用掩码，将未来的时间步的 pT 设置为 0
            pT = tl.where(mask, pT, 0.0)
        #TODO:加载do_ptrs
        do = tl.load(do_ptrs)
        # Compute dV.
        ppT = pT
        #TODO: 将 ppT 转换为 float16 类型
        ppT = ppT.to(tl.float16)
        #TODO: 计算 dV，dV 是 ppT 和 do 的点积，将结果累加到 dv 中
        dv += tl.dot(ppT, do)
        # D (= delta) is pre-divided by ds_scale.
        #TODO: 从内存中加载 D中偏移 offs_m 处的数据
        Di = tl.load(D + offs_m)
        # Compute dP and dS.
        #TODO: 计算值矩阵 v 和 do 的转置的点积，得到对 v 的梯度，结果转换为 float32 类型
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        #TODO: 计算dsT，通过 pT 乘以 dpT 和Di之间的差值
        dsT = pT * (dpT - Di[None, :])
        #TODO: 将 dsT 转换为 float16 类型
        dsT = dsT.to(tl.float16)
        #TODO: 计算 dK，将 dsT 和 qT 的转置的点积结果累加到 dk 中
        dk += tl.dot(dsT, tl.trans(qT))
        # Increment pointers.
        #TODO: 更新当前块的 m 维度上的起始位置
        curr_m += step_m
        #TODO: 更新 qT_ptrs，指向下一个块
        qT_ptrs += BLOCK_M1 * stride_tok
        # 更新 do_ptrs，指向下一个块
        do_ptrs += BLOCK_M1 * stride_tok
    return dk, dv
 
 
# the main inner-loop logic for computing dQ
@triton.jit
def _attn_bwd_dq(
        dq,
        q,
        K,
        V,  #
        do,
        m,
        D,
        stride_tok,
        stride_d,  #
        H,
        N_CTX,  #
        #TODO： 指定以下类型为编译时常量
        BLOCK_M2: tl.constexpr,  #
        BLOCK_N2: tl.constexpr,  #
        BLOCK_DMODEL: tl.constexpr,
        # Filled in by the wrapper.
        start_m,
        start_n,
        num_steps,  #
        #TODO： 指定以下类型为编译时常量
        MASK: tl.constexpr):
    #TODO: 计算当前块在 m 方向的偏移量
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    #TODO: 计算当前块在 n 方向的偏移量
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    #TODO: 计算 offs_k，用于在模型维度上进行计算和索引
    offs_k = tl.arange(0, BLOCK_DMODEL)
    #TODO: 计算键矩阵 K 的指针位置
    kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    #TODO: 计算键矩阵 V 的指针位置
    vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    # D (= delta) is pre-divided by ds_scale.
    #TODO:从内存中加载数据 D 在由 offs_m 确定的偏移位置上的值，
    Di = tl.load(D + offs_m)
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    #TODO: 静态断言：BLOCK_N1 必须是 BLOCK_M1 的倍数
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    for blk_idx in range(num_steps):
        #TODO: 加载kT_ptrs
        kT = tl.load(kT_ptrs)
        #TODO: 加载vT_ptrs
        vT = tl.load(vT_ptrs)
        #TODO: 计算 q 和 kT 的点积
        qk = tl.dot(q, kT)
        #TODO:将 qk 和 m 的差值转换为以2为底的指数形式
        p = tl.math.exp2(qk - m[:, None])
        # Autoregressive masking.
        if MASK:
            #TODO: 计算当前块的 n 方向的偏移量
            offs_n = curr_n + tl.arange(0, BLOCK_N2)
            #TODO: 生成掩码，掩盖不需要的元素
            mask = (offs_m[:, None] >= offs_n[None, :])
            #TODO: 应用掩码，将掩盖的部分置为 0
            p = tl.where(mask, p, 0.0)
        # Compute dP and dS.
        #TODO: 计算值矩阵 do 和 vT 的转置的点积，将结果转换为 float32 类型
        dp = tl.dot(do, tl.trans(vT)).to(tl.float32)
        #TODO: 计算ds，通过 p乘以 dp 和Di之间的差值
        ds = p * (dp - Di[:, None])
        #TODO: 将 ds 转换为 float16 类型
        ds = ds.to(tl.float16)
        # Compute dQ.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        #TODO: 计算 dq，将 ds 和 kT 的转置的点积结果累加到 dq 中
        dq += tl.dot(ds, tl.trans(kT))
        # Increment pointers.
        #TODO: 更新指针，移动到下一个块的位置
        curr_n += step_n
        kT_ptrs += BLOCK_N2 * stride_tok
        vT_ptrs += BLOCK_N2 * stride_tok
    return dq   
 
 
@triton.jit
def _attn_bwd(
        Q,
        K,
        V,
        sm_scale,  #
        DO,  #
        DQ,
        DK,
        DV,  #
        M,
        D,
        # shared by Q/K/V/DO.
        stride_z,
        stride_h,
        stride_tok,
        stride_d,  #
        H,
        N_CTX,  #
    #TODO： 指定以下类型为编译时常量
        BLOCK_M1: tl.constexpr,  #
        BLOCK_N1: tl.constexpr,  #
        BLOCK_M2: tl.constexpr,  #
        BLOCK_N2: tl.constexpr,  #
        BLK_SLICE_FACTOR: tl.constexpr,  #
        BLOCK_DMODEL: tl.constexpr):
    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)
 
    #TODO: 获取当前程序的批次/头部 ID
    bhid = tl.program_id(2)
    #TODO: 计算当前批次或注意力头部 bhid 的起始位置在数据数组 Q, K, V 和其他相关数组中的偏移量,并将其转换为 64 位整数
    off_chz = (bhid * N_CTX).to(tl.int64)
    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    #TODO: 获取当前程序的 ID
    pid = tl.program_id(0)
 
    # offset pointers for batch/head
    Q += adj
    K += adj
    V += adj
    DO += adj
    DQ += adj
    DK += adj
    DV += adj
    M += off_chz
    D += off_chz
 
    # load scales
    #TODO: 生成一个包含从 0 到 BLOCK_DMODEL - 1 的整数序列，用于表示当前计算块中所有的维度索引
    offs_k = tl.arange(0, BLOCK_DMODEL)
 
    #TODO: 计算当前处理的起始位置
    start_n = pid * BLOCK_N1
    start_m = start_n
 
    MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
    #TODO: 从 start_n 开始，通过增加从 0 到 BLOCK_N1 的范围值来计算每个块的起始位置，用于确定在块内的列偏移
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    
    #TODO: 初始化一个全零张量 dv，其形状为 [BLOCK_N1, BLOCK_DMODEL]，数据类型为 float32
    dv = tl.zeros([BLOCK_N1, BLOCK_DMODEL], dtype=tl.float32)
    #TODO: 初始化一个全零张量 dk，其形状为 [BLOCK_N1, BLOCK_DMODEL]，数据类型为 float32
    dk = tl.zeros([BLOCK_N1, BLOCK_DMODEL], dtype=tl.float32)
 
    # load K and V: they stay in SRAM throughout the inner loop.
    #TODO: 根据偏移量 offs_n 和 offs_k 以及步长 stride_tok 和 stride_d，从内存中加载键向量 K，并将其存储到变量 k 中
    k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    #TODO: 根据偏移量 offs_n 和 offs_k 以及步长 stride_tok 和 stride_d，从内存中加载值向量 V，并将其存储到变量 v 中
    v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
 
    #TODO: 通过将总数据块大小除以每次迭代处理的数据块大小来计算处理所有数据块所需的迭代次数
    num_steps = MASK_BLOCK_M1
 
    #TODO: 调用函数计算 dk 和 dv（采用mask)
    dk, dv = _attn_bwd_dkdv(dk, dv, Q, k, v, sm_scale, DO, M, D, stride_tok, stride_d, H, N_CTX, BLOCK_M1, BLOCK_N1, BLOCK_DMODEL, start_n, start_m, num_steps, MASK=True)
 
    #TODO： 更新 start_m 
    start_m += num_steps * BLOCK_M1
    #TODO: 更新num_steps
    num_steps = (N_CTX - start_m) // BLOCK_M1
 
    # Compute dK and dV for non-masked blocks.
    #TODO: 调用函数计算 dk 和 dv（不用采用mask)
    dk, dv = _attn_bwd_dkdv(dk, dv, Q, k, v, sm_scale, DO, M, D, stride_tok, stride_d, H, N_CTX, BLOCK_M1, BLOCK_N1, BLOCK_DMODEL, start_n, start_m, num_steps, MASK=False)
 
    ## 计算 dv 的存储位置指针
    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    # 将 dv 张量的内容存储到 dv_ptrs 所指向的内存位置
    tl.store(dv_ptrs, dv)
    
    #TODO:按比例缩放 dk 张量，将其乘以缩放因子
    dk *= sm_scale
    #TODO:计算 dk_ptrs，将其指向 DK 张量在内存中的对应位置
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    #TODO: 将缩放后的 dk 张量存储到 dk_ptrs 所指向的内存位置
    tl.store(dk_ptrs, dk)
 
    #TODO: 更新 start_m(根据pid和BLOCK_M2 确定) 
    start_m = pid * BLOCK_M2
    #TODO: 计算结束的列索引
    end_n = start_m + BLOCK_M2
 
    MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
    #TODO: 从 start_n 开始，通过增加从 0 到 BLOCK_M2 的范围值来计算每个块的起始位置，用于确定在块内的列偏移
    offs_m = start_m + tl.arange(0, BLOCK_M2)

    #TODO: 从张量 Q 中加载数据，利用 offs_m 和 offs_k 计算内存地址偏移量
    q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    #TODO: 初始化 dq 为全零张量，大小为 [BLOCK_M2, BLOCK_DMODEL]，数据类型为 float32
    dq = tl.zeros([BLOCK_M2, BLOCK_DMODEL], dtype=tl.float32)
    #TODO: 从张量 DO 中加载数据，利用 offs_m 和 offs_k 计算内存地址偏移量
    do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
 
    #TODO: 从张量 M 中加载数据，根据 offs_m 计算出要加载的内存地址
    m = tl.load(M + offs_m)
    m = m[:, None]
 
    # Compute dQ for masked (diagonal) blocks.
    # NOTE: This code scans each row of QK^T backward (from right to left,
    # but inside each call to _attn_bwd_dq, from left to right), but that's
    # not due to anything important.  I just wanted to reuse the loop
    # structure for dK & dV above as much as possible.
    #TODO:  通过将总数据块大小除以每次迭代处理的数据块大小来计算处理所有数据块所需的迭代次数
    num_steps = MASK_BLOCK_N2
    #TODO: 调用 _attn_bwd_dq 函数，计算带掩码的 dQ 值
    dq = _attn_bwd_dq(dq, q, K, V, do, m, D, stride_tok, stride_d, H, N_CTX, BLOCK_M2, BLOCK_N2, BLOCK_DMODEL, start_m, end_n - num_steps * BLOCK_N2, num_steps, MASK=True)
    #TODO: 更新 end_n，减少已经计算过的列数。
    end_n -= num_steps * BLOCK_N2
    # stage 2
    #TODO: 计算无掩码块的步数
    num_steps = end_n // BLOCK_N2
    #TODO: 再次调用 _attn_bwd_dq 函数，计算没有掩码的 dQ 值
    dq = _attn_bwd_dq(dq, q, K, V, do, m, D, stride_tok, stride_d, H, N_CTX, BLOCK_M2, BLOCK_N2, BLOCK_DMODEL, start_m, 0, num_steps, MASK=False)
    # Write back dQ.
    #TODO：计算 DQ 的存储位置指针
    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    #TODO: 利用LN2对计算的 dQ 进行缩放
    dq *= LN2
    #TODO: 将缩放后的 dq存储到指定的位置dq_ptrs中。
    tl.store(dq_ptrs, dq)
 
 
empty = torch.empty(128, device="mlu")
 
 
class _attention(torch.autograd.Function):
 
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale,causal_mask):
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        #TODO: 确保Lq、Lk和Lv张量的长度相同
        assert Lq == Lk and Lk == Lv
        #assert Lk in {16, 32, 64, 128}
        #TODO: 创建一个与查询张量q形状相同的空张量
        o = torch.empty_like(q)
   
        
        #TODO: 获取查询张量 q 的上下文长度（倒数第二维的大小）
        q_ctx_len = q.shape[-2]
        #TODO: 获取键张量 k 的上下文长度（倒数第二维的大小）
        kv_ctx_len = k.shape[-2]

        if q_ctx_len <= 128:
            if kv_ctx_len%128==0:
                BLOCK_M = q_ctx_len
                BLOCK_N = 128
            else:
                BLOCK_M = q_ctx_len
                BLOCK_N = kv_ctx_len
        elif q_ctx_len < 256:
            if kv_ctx_len%64==0:
                BLOCK_M = q_ctx_len
                BLOCK_N = 64
            else:
                BLOCK_M = q_ctx_len
                BLOCK_N = kv_ctx_len
        elif q_ctx_len >= 256:
            if kv_ctx_len%64==0:
                BLOCK_M = 64
                BLOCK_N = 64
            elif kv_ctx_len%32==0:
                BLOCK_M = 32
                BLOCK_N = 32
            else:
                BLOCK_M = 64
                BLOCK_N = kv_ctx_len
        
        


        #print("------------BLOCK_M:",BLOCK_M)
        #print("------------BLOCK_N:",BLOCK_N)


        num_stages = 4 if Lk <= 64 else 3
        num_warps = 1
        stage = 3 if causal else 1
        if torch.mlu.get_device_capability()[0] == 9:
            num_warps = 8
            # num_stages = 7 if Lk >= 64 else 3
            num_stages = 0
        num_stages = 0
        #grid is coredim clusterdim 1
        grid = (4, 8, 1)
        #TODO: 创建一个与 q 张量形状匹配的空张量 M，同时将 M 张量分配到与 q 张量相同的设备上,并将 M 张量的数据类型设置为 float32。
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

        def is_divisible(a, b):
            if b == 0:
                raise ValueError("Divisor cannot be 0")
            return a % b == 0

        #TODO: 获取张量 q 在第三维的大小，并将其赋值给 N_CTX，表示上下文数量。
        N_CTX = q.shape[2]
        IS_DIVISIBLE = False
        if is_divisible(N_CTX, BLOCK_M) and is_divisible(N_CTX, BLOCK_N):
            IS_DIVISIBLE = True


        if(causal_mask is not None):
            _attn_eff_fwd[grid](
                q,
                k,
                v,
                sm_scale,
                M,
                o,  #
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),  #
                k.stride(0),
                k.stride(1),
                k.stride(2),
                k.stride(3),  #
                v.stride(0),
                v.stride(1),
                v.stride(2),
                v.stride(3),  #
                o.stride(0),
                o.stride(1),
                o.stride(2),
                o.stride(3),  #
                causal_mask.stride(2), 
                causal_mask.stride(3),
                q.shape[0],
                q.shape[1],  #
                causal_mask,
                N_CTX=k.shape[2],  #
                Q_N_CTX=q.shape[2],
                BLOCK_M=BLOCK_M,  #
                BLOCK_N=BLOCK_N,  #
                BLOCK_DMODEL=Lk,  # D_HEAD
                STAGE=stage,  #
                IS_DIVISIBLE=IS_DIVISIBLE,  #
                num_warps=num_warps,  #
                num_stages=num_stages  #
            )
        else:
            _attn_fwd[grid](
                q,
                k,
                v,
                sm_scale,
                M,
                o,  #
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),  #
                k.stride(0),
                k.stride(1),
                k.stride(2),
                k.stride(3),  #
                v.stride(0),
                v.stride(1),
                v.stride(2),
                v.stride(3),  #
                o.stride(0),
                o.stride(1),
                o.stride(2),
                o.stride(3),  #
                q.shape[0],
                q.shape[1],  #
                N_CTX=k.shape[2],  #
                Q_N_CTX=q.shape[2],
                BLOCK_M=BLOCK_M,  #
                BLOCK_N=BLOCK_N,  #
                BLOCK_DMODEL=Lk,  # D_HEAD
                STAGE=stage,  #
                IS_DIVISIBLE=IS_DIVISIBLE,  #
                num_warps=num_warps,  #
                num_stages=num_stages  #
            )
 
 
 
        #TODO:保存前向传播中需要反向传播使用的张量q、k、v、o、M以供反向传播时使用
        ctx.save_for_backward(q, k, v, o, M)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        ctx.causal = causal
        return o
 
    @staticmethod
    def backward(ctx, do):
        #TODO:  从 ctx 对象中恢复在前向传播中保存的张量q, k, v, o, M
        q, k, v, o, M = ctx.saved_tensors
        #TODO: 确保梯度张量do是连续的（即在内存中存储为连续块）
        do = do.contiguous()
        #TODO: 检查q、k、v、o和do张量的内存步长是否一致，以确保它们的内存布局相同。
        assert do.stride() == q.stride() == k.stride() == v.stride() == o.stride()
        #TODO: 创建与q张量相同形状的新张量dq
        dq = torch.empty_like(q)
        #TODO: 创建与k张量相同形状的新张量dk
        dk = torch.empty_like(k)
        #TODO: 创建与v张量相同形状的新张量dv
        dv = torch.empty_like(v)
        BATCH, N_HEAD, N_CTX = q.shape[:3]
        PRE_BLOCK = 128
        NUM_WARPS, NUM_STAGES = 4, 0
        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32
        BLK_SLICE_FACTOR = 2
        RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
        arg_k = k
        #TODO:将arg_k 乘以标量 ctx.sm_scale * RCP_LN进行缩放，更新 arg_k 的值。
        arg_k = arg_k * (ctx.sm_scale * RCP_LN2)
        PRE_BLOCK = 128
        #assert N_CTX % PRE_BLOCK == 0
        pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
        #TODO: 创建一个与M张量形状相同的新张量delta
        delta = torch.empty_like(M)
        #TODO: 调用_attn_bwd_preprocess函数，在pre_grid网格上执行反向传播预处理操作
        _attn_bwd_preprocess[pre_grid](o, do, delta, BATCH, N_HEAD, N_CTX, BLOCK_M=PRE_BLOCK, D_HEAD=ctx.BLOCK_DMODEL)
        grid = (N_CTX // BLOCK_N1, 1, BATCH * N_HEAD)
        _attn_bwd[grid](
            q,
            arg_k,
            v,
            ctx.sm_scale,
            do,
            dq,
            dk,
            dv,  #
            M,
            delta,  #
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),  #
            N_HEAD,
            N_CTX,  #
            BLOCK_M1=BLOCK_M1,
            BLOCK_N1=BLOCK_N1,  #
            BLOCK_M2=BLOCK_M2,
            BLOCK_N2=BLOCK_N2,  #
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,  #
            BLOCK_DMODEL=ctx.BLOCK_DMODEL,  #
            num_warps=NUM_WARPS,  #
            num_stages=NUM_STAGES  #
        )
 
        return dq, dk, dv, None, None, None
 
 
attention = _attention.apply
 
 

def test_op():
    torch.manual_seed(20)
    Z, H, N_CTX, D_HEAD=1,128,257,16
    causal=False # 
    causal_mask=[]
    dtype=torch.float16
    use_data_from_file=False
    Q_N_CTX=N_CTX
    
    for n in range(N_CTX,N_CTX+1):
        if (use_data_from_file==False):
            #TODO: 生成随机的q张量，并设置为需要梯度
            q = torch.empty((Z, H, Q_N_CTX, D_HEAD), dtype=dtype, device="mlu").normal_(mean=0.0, std=0.5).requires_grad_()
            #TODO: 生成随机的k张量，并设置为需要梯度
            k = torch.empty((Z, H, n, D_HEAD), dtype=dtype, device="mlu").normal_(mean=0.0, std=0.5).requires_grad_()
            v = (torch.empty((Z, H, n, D_HEAD), dtype=dtype,
                         device="mlu").normal_(mean=0.0, std=0.5).requires_grad_()).contiguous()
            if(causal_mask is not None):
                causal_mask = (torch.empty((Z, 1, Q_N_CTX, n), dtype=dtype,
                             device="mlu").normal_(mean=0.0, std=0.5).requires_grad_()).contiguous()
                # causal_mask = (torch.zeros((Z, 1, Q_N_CTX, n), dtype=dtype,
                             # device="mlu").requires_grad_()).contiguous()
        else:
            q_np = np.fromfile("query_states.npy", dtype=np.float16).reshape(Z, H, N_CTX, D_HEAD)
            k_np = np.fromfile("key_states.npy", dtype=np.float16).reshape(Z, H, N_CTX, D_HEAD)
            v_np = np.fromfile("value_states.npy", dtype=np.float16).reshape(Z, H, N_CTX, D_HEAD)

            #TODO: 将q_np 转换为 PyTorch 张量，并移动到mlu设备上，调整张量的形状为(Z, H, N_CTX, D_HEAD),设置为需要计算梯度
            q = torch.from_numpy(q_np).to("mlu").reshape(Z, H, N_CTX, D_HEAD).requires_grad_()
            #TODO: 将k_np 转换为 PyTorch 张量，并移动到mlu设备上，调整张量的形状为(Z, H, N_CTX, D_HEAD),设置为需要计算梯度
            k = torch.from_numpy(k_np).to("mlu").reshape(Z, H, N_CTX, D_HEAD).requires_grad_()
            #TODO: 将v_np 转换为 PyTorch 张量，并移动到mlu设备上，调整张量的形状为(Z, H, N_CTX, D_HEAD),设置为需要计算梯度
            v = torch.from_numpy(v_np).to("mlu").reshape(Z, H, N_CTX, D_HEAD).requires_grad_()

        sm_scale = 0.5
        #TODO: 生成与q张量相同形状的随机张量
        dout = torch.randn_like(q)
        
        print("q:",q.shape)
        print("k:",k.shape)
        print("v:",v.shape)
        print("causal:",causal)
        
        # triton的实现
        st=time.time()    
        #TODO: 调用自定义的高效注意力机制函数，并将结果转换为半精度浮点数
        tri_out = attention(q, k, v, causal, sm_scale, causal_mask).half()
        ed=time.time()
        print("triton attention cost:",ed-st)
        ##print("tri_out:",tri_out)
        #TODO: 将tri_out展平
        tri_out = tri_out.flatten()
        #TODO: 标识张量tri_out中的每个元素是否为 NaN (Not a Number)
        nan_mask = torch.isnan(tri_out)
        #TODO:  检查nan_mask中是否有任何元素为 True
        has_nan = torch.any(nan_mask)
        #print("tri_out has_nan",has_nan)
        
        
        # sdpa的实现
        st=time.time()
        if(causal_mask is not None): causal=False
        #TODO: 调用 PyTorch 的 scaled_dot_product_attention 函数，计算缩放点积注意力。
        sdpa_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask, is_causal=causal, scale=sm_scale)
        ed=time.time()
        print("scaled_dot_product_attention attention cost:",ed-st)
        ##print("sdpa_output:",sdpa_output)
        #TODO: 将sdpa_output展平
        sdpa_output = sdpa_output.flatten()
        
        pytorch_valible=True
        if(pytorch_valible==True):
            ## pytorch的实现
            st=time.time() 
            #TODO: 创建一个下三角矩阵 M，大小为 (Q_N_CTX, N_CTX)，其元素为 1
            M = torch.tril(torch.ones((Q_N_CTX, n), device="mlu"))
            #TODO: 计算查询张量 q 和键张量 k 的转置的点积
            qk = torch.matmul(q, k.transpose(2, 3))
            #TODO: 将 qk 乘以缩放因子 sm_scale
            p = qk * sm_scale
            if(causal_mask is not None):
                p=p+causal_mask
            elif causal:
                p[:, :, M == 0] = float("-inf")
            
            if(1):
                #TODO: 对 p 应用 softmax，并将结果转换为半精度浮点数
                p = torch.softmax(p.float(), dim=-1).half()
                #TODO:  计算 p 和值张量 v 的点积，得到 pyt_out
                pyt_out = torch.matmul(p, v)
            #TODO:将pyt_out进行展平
            pyt_out = pyt_out.flatten()
            ed=time.time()
            print("pytorch attention cost:",ed-st)
            
            # compare
            #TODO：计算 pyt_out 和 tri_out 之间的绝对误差总和
            abs_tp_error = torch.sum(torch.abs(pyt_out - tri_out))
            #TODO: 计算 pyt_out 和 tri_out 之间的相对误差，总绝对误差除以两者绝对值和的最小值，避免除以零
            rel_tp_error = abs_tp_error / torch.min(torch.sum(torch.abs(pyt_out)) + torch.sum(torch.abs(tri_out)), torch.tensor(1e-8, device="mlu"))
            print("abs_tp_error:",abs_tp_error)
            print("rel_tp_error:",rel_tp_error)
            #TODO:计算 pyt_out 和 sdpa_output 之间的绝对误差总和
            abs_sp_error = torch.sum(torch.abs(pyt_out - sdpa_output))
            #TODO:计算 pyt_out 和 sdpa_output 之间的相对误差
            rel_sp_error = abs_sp_error / torch.min(torch.sum(torch.abs(pyt_out)) + torch.sum(torch.abs(sdpa_output)), torch.tensor(1e-8, device="mlu"))
            print("abs_sp_error:",abs_sp_error)
            print("rel_sp_error:",rel_sp_error)
        #TODO: 计算 sdpa_output 和 tri_out 之间的绝对误差总和
        abs_ts_error = torch.sum(torch.abs(sdpa_output - tri_out))
        #TODO: 计算 sdpa_output 和 tri_out 之间的相对误差
        rel_ts_error = abs_ts_error / torch.min(torch.sum(torch.abs(sdpa_output)) + torch.sum(torch.abs(tri_out)), torch.tensor(1e-8, device="mlu"))
        print("abs_ts_error:",abs_ts_error)
        print("rel_ts_error:",rel_ts_error)

 
if __name__ == '__main__':
    print("====================== Val =======================")
    #TODO:调用测试函数进行性能测试
    test_op()
     
    #print("====================== Benchmark =======================")
    #bench_flash_attention.run(save_path=".", print_data=True)