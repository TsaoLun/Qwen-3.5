use burn::{
    backend::{
        Backend, Dispatch, TensorMetadata, backend_extension,
        fusion::custom::TensorSpec,
        ops::{FloatTensorOps, IntTensorOps},
        tensor::{FloatTensor, IntTensor, QuantizedTensor},
        wgpu::CubeBackend,
    },
    tensor::{DType, FloatDType, IntDType, Shape},
};
use cubecl::prelude::*;

/// This only works for Q8F quantized embeddings
/// We will launch n * D units
#[cube(launch)]
fn q8_embedding_lookup_kernel(
    table: &Tensor<u32>,      // [vocab_size, D / 4] packed Q8F words, flat
    ids: &Tensor<i32>,        // [n]
    scales: &Tensor<f32>,     // [vocab_size, D / 64]
    output: &mut Tensor<f32>, // [n, D]
    #[comptime] dim: usize,   // D
) {
    let pos = ABSOLUTE_POS;
    // guard check to stay less than n*D
    if pos < output.len() {
        let row = pos / dim; // Units 0 to D-1 cover first row, D to 2D-1 cover the second and so on
        let col = pos % dim; // For 1st row, unit 0 covers first embedding idx, unit 1 covers second and so on
        let id = ids[row]; // Which id this unit is working on, each id will have D units
        let words_per_row = dim / 4; // Q8F
        // e.g. id = 1, words_per_row = 512 (dim = 2048), col = 3
        // word_idx = 512 + 3 / 4 = 512, which is what we want. as each word contains 4 values, (col 0, 1, 2 and 3 all will go to 512th word)
        let word_idx = usize::cast_from(id) * words_per_row + col / 4;
        let word = table[word_idx];
        // col 0 -> 0 shift
        // col 1 -> 8 shift
        // col 2 -> 16 shift
        // col 3 -> 24 shift
        // Because [a, b, c, d] are stored as [d, c, b, a] in packed format.
        let shift = u32::cast_from(col % 4) * 8u32;
        let raw = (word >> shift) & 255u32; // (Masking with 255 to get actual value)
        let is_negative = raw >= 128u32; // We have to subtract 256 if we originally had a negative number
        let signed = i32::cast_from(raw) - i32::cast_from(is_negative) * 256;
        output[pos] = f32::cast_from(signed) * scales[usize::cast_from(id) * (dim / 64) + col / 64];
    }
}

/// e.g. in_proj_qkv: [2048, 6144]
/// weights: [2048, 1536] Q8F packed words
/// hidden_states: [2048]
/// scales: [2048, 96]
/// output: [6144]
/// We will launch out_dim units, one for each output
#[cube(launch)]
fn q8_gemv_kernel(
    hidden_states: &Tensor<f32>, // [in_dim]
    weights: &Tensor<u32>,       // [in_dim, out_dim / 4]
    scales: &Tensor<f32>,        // [in_dim, out_dim / 64]
    output: &mut Tensor<f32>,    // [out_dim]
    #[comptime] in_dim: usize,
    #[comptime] out_dim: usize,
) {
    let pos = ABSOLUTE_POS;
    if pos >= output.len() {
        terminate!()
    }
    // pos 0: shift 0
    // pos 1: shift 8
    // pos 2: shift 16
    // pos 3: shift 24
    // pos 4: shift 0 ...
    let shift = u32::cast_from(pos % 4) * 8;
    let col = pos / 4; // 0 to 3 work on 1st col of weights, 4-7 work on 2nd col of weights and so on...
    let words_per_row = out_dim / 4;
    let mut acc = 0.0f32;
    let num_scales = out_dim / 64;
    let scale_col = pos / 64;
    let u = 8usize;
    for blk in 0..(in_dim / u) {
        #[unroll]
        for j in 0..u {
            let row = blk * u + j;
            let word_idx = row * words_per_row + col;
            let word = weights[word_idx];
            let raw = (word >> shift) & 255u32; // Get correct bytes
            let is_negative = raw >= 128u32;
            let signed = i32::cast_from(raw) - i32::cast_from(is_negative) * 256;
            let scale_idx = row * num_scales + scale_col;
            acc += f32::cast_from(signed) * hidden_states[row] * scales[scale_idx];
        }
    }
    output[pos] = acc;
}

/// Will launch out_dim * num_splits units
/// CubeCount: (num_tiles, num_splits, 1)
/// e.g. in_proj_qkv: [2048, 6144]
/// weights: [2048, 1536] Q8F packed words
/// hidden_states: [2048]
/// scales: [2048, 96]
/// num_splits = 4
/// output: [4, 6144]
/// Every thread will now read the chunk which is in_dim / 4 => 512
#[cube(launch)]
fn q8_gemv_partial_sum_kernel(
    hidden_states: &Tensor<f32>, // [in_dim]
    weights: &Tensor<u32>,       // [in_dim, out_dim / 4]
    scales: &Tensor<f32>,        // [in_dim, out_dim / 64]
    output: &mut Tensor<f32>,    // [num_splits, out_dim]
    #[comptime] in_dim: usize,
    #[comptime] out_dim: usize,
    #[comptime] num_splits: usize,
) {
    let pos = usize::cast_from(CUBE_POS_X * 256 + UNIT_POS); // 0..6143, the output idx
    if pos >= out_dim {
        terminate!()
    }
    let split = usize::cast_from(CUBE_POS_Y);

    let chunk_size = in_dim / num_splits; // 512
    let row_start = chunk_size * split;

    let words_per_row = out_dim / 4;
    let mut acc = 0.0f32;
    let num_scales = out_dim / 64;

    let shift = (u32::cast_from(pos) % 4) * 8u32;
    let col = pos / 4; // Q8F, 6144 units should cover 1536 words
    let scale_col = pos / 64;
    let u = 8usize;
    for blk in 0..(chunk_size / u) {
        #[unroll]
        for j in 0..u {
            let i = row_start + blk * u + j;
            let word_idx = words_per_row * i + col;
            let word = weights[word_idx];
            let raw = (word >> shift) & 255u32;
            let is_negative = raw >= 128u32;
            let signed = i32::cast_from(raw) - i32::cast_from(is_negative) * 256;
            let scale_idx = i * num_scales + scale_col;
            acc += f32::cast_from(signed) * hidden_states[i] * scales[scale_idx];
        }
    }

    output[out_dim * split + pos] = acc;
}

/// Will launch dim units
#[cube(launch)]
fn q8_gemv_sum_reduction_kernel(
    input: &Tensor<f32>,      // [num_splits, dim]
    output: &mut Tensor<f32>, // [dim]
    #[comptime] dim: usize,
    #[comptime] num_splits: usize,
) {
    let pos = ABSOLUTE_POS;
    if pos >= dim {
        terminate!()
    }
    let mut acc = 0.0f32;
    #[unroll]
    for i in 0..num_splits {
        acc += input[dim * i + pos];
    }
    output[pos] = acc;
}

// Will launch 2 * num_splits * n_heads units
// And 1 block
// e.g. hidden_states: [2048]
// w_b: [2048, 16]
// w_a: [2048, 16]
// a_log: [16]
// dt_bias: [16]
#[cube(launch)]
fn gate_proj_kernel(
    hidden_states: &Tensor<f32>, // [in_dim]
    w_b: &Tensor<f32>,           // [in_dim, n_heads]
    w_a: &Tensor<f32>,           // [in_dim, n_heads]
    a_log: &Tensor<f32>,         // [n_heads]
    dt_bias: &Tensor<f32>,       // [n_heads]
    out: &mut Tensor<f32>,       // [2 * n_heads] -> half beta, half gate
    #[comptime] in_dim: usize,
    #[comptime] n_heads: usize,
    #[comptime] num_splits: usize,
) {
    // beta then gate
    let mut partial = Shared::<[f32]>::new_slice(num_splits * 2 * n_heads); // (num_splits, 2 * n_heads)
    let n_out = 2 * n_heads;
    let pos = ABSOLUTE_POS;
    let output_idx = pos % n_heads; // 0 to 15
    let chunk_size = in_dim / num_splits;
    let split_idx = pos / n_out;
    let row_start = split_idx * chunk_size;
    let is_beta = (pos / n_heads) % 2 == 0; // Units 0 to 15 for beta, 16 to 31 for gate, 32 to 47 for beta, 48 to 61 for gate and so on
    let mut acc = 0.0f32;
    let u = 8usize;
    if is_beta {
        for blk in 0..(chunk_size / u) {
            #[unroll]
            for j in 0..u {
                let row = row_start + blk * u + j;
                let weight_idx = n_heads * row + output_idx;
                acc += hidden_states[row] * w_b[weight_idx];
            }
        }
        partial[2 * n_heads * split_idx + output_idx] = acc;
    } else {
        for blk in 0..(chunk_size / u) {
            #[unroll]
            for j in 0..u {
                let row = row_start + blk * u + j;
                let weight_idx = n_heads * row + output_idx;
                acc += hidden_states[row] * w_a[weight_idx];
            }
        }
        partial[2 * n_heads * split_idx + output_idx + n_heads] = acc;
    }
    sync_cube(); // barrier so that every partial sum is there
    if split_idx == 0 {
        if is_beta {
            // sigmoid(b) => 1 / (1 + exp(-b))
            let mut sum = 0.0f32; // b
            #[unroll]
            for i in 0..num_splits {
                let partial_idx = n_out * i + output_idx;
                sum += partial[partial_idx];
            }
            let sigmoid = 1.0 / (1.0 + (-sum).exp());
            out[output_idx] = sigmoid;
        } else {
            let mut sum = 0.0f32; // a
            #[unroll]
            for i in 0..num_splits {
                let partial_idx = n_out * i + output_idx + n_heads;
                sum += partial[partial_idx];
            }
            let x_gate = sum + dt_bias[output_idx];
            let pre_softplus = (x_gate.exp() + 1.0).ln();
            let softplus = if pre_softplus > 20.0 {
                x_gate
            } else {
                pre_softplus
            };
            out[output_idx + n_heads] = -a_log[output_idx].exp() * softplus;
        }
    }
}

/// One row of the LM head. Launch with 256 threads per cube: eight rows, 32 lanes each.
/// Each lane owns one packed word. The 32-lane sum is done in shared memory: `plane_sum`
/// spans the whole plane, which is 64 lanes (two rows) on AMD GCN.
#[cube(launch)]
fn q8_lm_head_kernel(
    hidden_states: &Tensor<f32>, // [dim]
    embeddings: &Tensor<u32>,    // [vocab_size, dim]
    scales: &Tensor<f32>,        // [vocab_size, dim / 64]
    output: &mut Tensor<f32>,    // [vocab_size]
    #[comptime] dim: usize,
) {
    let mut smem = Shared::<[f32]>::new_slice(256usize);
    let local = usize::cast_from(UNIT_POS);
    let lane = local % 32;
    let row = usize::cast_from(ABSOLUTE_POS) / 32;
    // No early exit: every thread has to reach the barriers below.
    let active = row < output.len();

    let mut acc = 0.0f32;
    if active {
        let words_per_row = dim / 4;
        let scales_per_row = dim / 64;
        // 32 lanes cover 128 columns per step: each lane owns one packed word.
        for k in 0..(dim / 128) {
            let word_col = k * 32 + lane;
            let col0 = word_col * 4;
            let word = embeddings[words_per_row * row + word_col];
            let scale = scales[row * scales_per_row + col0 / 64];
            let b0 = (word >> 0u32) & 255u32;
            let b1 = (word >> 8u32) & 255u32;
            let b2 = (word >> 16u32) & 255u32;
            let b3 = (word >> 24u32) & 255u32;
            let s0 = i32::cast_from(b0) - i32::cast_from(b0 >= 128u32) * 256;
            let s1 = i32::cast_from(b1) - i32::cast_from(b1 >= 128u32) * 256;
            let s2 = i32::cast_from(b2) - i32::cast_from(b2 >= 128u32) * 256;
            let s3 = i32::cast_from(b3) - i32::cast_from(b3 >= 128u32) * 256;
            acc += (f32::cast_from(s0) * hidden_states[col0]
                + f32::cast_from(s1) * hidden_states[col0 + 1]
                + f32::cast_from(s2) * hidden_states[col0 + 2]
                + f32::cast_from(s3) * hidden_states[col0 + 3])
                * scale;
        }
    }

    smem[local] = acc;
    sync_cube();
    if lane < 16 {
        let sum = smem[local] + smem[local + 16];
        smem[local] = sum;
    }
    sync_cube();
    if lane < 8 {
        let sum = smem[local] + smem[local + 8];
        smem[local] = sum;
    }
    sync_cube();
    if lane < 4 {
        let sum = smem[local] + smem[local + 4];
        smem[local] = sum;
    }
    sync_cube();
    if lane < 2 {
        let sum = smem[local] + smem[local + 2];
        smem[local] = sum;
    }
    sync_cube();
    if lane < 1 {
        let sum = smem[local] + smem[local + 1];
        smem[local] = sum;
    }
    sync_cube();
    if lane == 0 && active {
        output[row] = smem[local];
    }
}

// Will launch vocab_size / num_scans units
// e.g. vocab_size = 248320
// num_scans = 8
// Each unit only compares 8 addresses => 31040 units
// We will launch 31040 units
#[cube(launch)]
fn argmax_partial_kernel(
    logits: &Tensor<f32>,             // [vocab_size]
    output: &mut Tensor<f32>,         // [vocab_size / num_scans]
    output_indices: &mut Tensor<i32>, // [vocab_size / num_scans]
    #[comptime] vocab_size: usize,
    #[comptime] num_scans: usize, // number of addresses one unit has to look up
) {
    let pos = ABSOLUTE_POS;
    if pos >= vocab_size / num_scans {
        terminate!()
    }
    let chunk_size = vocab_size / num_scans; // 31040
    let mut max = logits[pos];
    let mut max_idx = pos;
    for k in 1..num_scans {
        let idx = k * chunk_size + ABSOLUTE_POS; // unit 0 will look at addresses 0, 31040, 62080, ... 217280
        // unit 31039 will look at addresses 31039, 62079, 93119, ... 248319
        let value = logits[idx];
        if value > max {
            max = value;
            max_idx = idx;
        }
    }

    output[pos] = max;
    output_indices[pos] = i32::cast_from(max_idx);
}

// 256 threads, 1 block
// e.g. this is 31040 according to the kernel above with vocab_size = 248320 and num_scans = 8
// dim = 31040
#[cube(launch)]
fn argmax_final(
    partial: &Tensor<f32>,         // [dim]
    partial_indices: &Tensor<i32>, // [dim]
    output: &mut Tensor<i32>,      // [1]
    #[comptime] dim: usize,
) {
    let pos = ABSOLUTE_POS;
    // Need coalescing again
    let num_scans = dim.div_ceil(256); // How many addresses each unit will look at
    let mut shared_partial = Shared::<[f32]>::new_slice(256usize);
    let mut shared_partial_indices = Shared::<[i32]>::new_slice(256usize);
    let mut max = partial[pos];
    let mut max_idx = partial_indices[pos];
    for k in 1..num_scans {
        let idx = k * 256 + pos; // unit 0 will look at address 0, 256, ... 30976
        // unit 1 will look at 1, 257, ... 30976
        // unit 255 will look at 255, 511, ... 31231
        if idx < dim {
            let value = partial[idx];
            if value > max {
                max = value;
                max_idx = partial_indices[idx];
            }
        }
    }
    shared_partial[pos] = max;
    shared_partial_indices[pos] = max_idx;
    sync_cube();
    if pos == 0 {
        let mut max = shared_partial[0];
        let mut max_idx = shared_partial_indices[0];
        for idx in 1..256 {
            let value = shared_partial[idx];
            if value > max {
                max = value;
                max_idx = shared_partial_indices[idx];
            }
        }
        output[0] = max_idx;
    }
}

/// Row-wise RMSNorm. `plus_one` matches `RMSNorm` (`(1 + weight)`); otherwise the weight is the
/// scale itself. `gated` multiplies by `silu(gate)` after the norm.
#[cube(launch)]
fn rmsnorm_kernel(
    x: &Tensor<f32>,      // [rows, dim], contiguous
    gate: &Tensor<f32>,   // same shape, read only when gated
    weight: &Tensor<f32>, // [dim]
    out: &mut Tensor<f32>,
    eps: f32,
    rows: u32,
    row_stride: u32,
    #[comptime] dim: usize,
    #[comptime] gated: bool,
    #[comptime] plus_one: bool,
) {
    let mut smem = Shared::<[f32]>::new_slice(256usize);
    let tid = usize::cast_from(UNIT_POS);
    let mut row = CUBE_POS_X;
    while row < rows {
        let row_us = usize::cast_from(row);
        let mut local = 0.0f32;
        let steps = (dim + 255) / 256;
        for step in 0..steps {
            let idx = tid + step * 256;
            if idx < dim {
                let v = x[row_us * dim + idx];
                local += v * v;
            }
        }
        smem[tid] = local;
        sync_cube();
        if tid < 128 {
            let sum = smem[tid] + smem[tid + 128];
            smem[tid] = sum;
        }
        sync_cube();
        if tid < 64 {
            let sum = smem[tid] + smem[tid + 64];
            smem[tid] = sum;
        }
        sync_cube();
        if tid < 32 {
            let sum = smem[tid] + smem[tid + 32];
            smem[tid] = sum;
        }
        sync_cube();
        if tid < 16 {
            let sum = smem[tid] + smem[tid + 16];
            smem[tid] = sum;
        }
        sync_cube();
        if tid < 8 {
            let sum = smem[tid] + smem[tid + 8];
            smem[tid] = sum;
        }
        sync_cube();
        if tid < 4 {
            let sum = smem[tid] + smem[tid + 4];
            smem[tid] = sum;
        }
        sync_cube();
        if tid < 2 {
            let sum = smem[tid] + smem[tid + 2];
            smem[tid] = sum;
        }
        sync_cube();
        if tid < 1 {
            let sum = smem[tid] + smem[tid + 1];
            smem[tid] = sum;
        }
        sync_cube();
        let inv = (smem[0] / f32::cast_from(dim) + eps).sqrt().recip();
        // The next row overwrites smem[0]; wait until every unit has read it.
        sync_cube();
        for step in 0..steps {
            let idx = tid + step * 256;
            if idx < dim {
                let v = x[row_us * dim + idx];
                let scale = if comptime!(plus_one) {
                    1.0 + weight[idx]
                } else {
                    weight[idx]
                };
                let mut y = v * inv * scale;
                if comptime!(gated) {
                    let g = gate[row_us * dim + idx];
                    y *= g / (1.0 + (-g).exp());
                }
                out[row_us * dim + idx] = y;
            }
        }
        row += row_stride;
    }
}

/// One Gated Delta Rule step for a single token.
///
/// One cube per head, one unit per value column. `k_dim` must equal `v_dim` and be a power of two.
/// `state` is contiguous `[1, heads, k_dim, v_dim]`. `beta_g` is `[beta | g]`.
#[cube(launch)]
fn gdn_decode_kernel(
    qkv: &Tensor<f32>,           // [2 * heads * k_dim + heads * v_dim]
    beta_g: &Tensor<f32>,        // [2 * heads]
    state: &Tensor<f32>,         // [1, heads, k_dim, v_dim]
    out: &mut Tensor<f32>,       // [heads * v_dim]
    new_state: &mut Tensor<f32>, // [1, heads, k_dim, v_dim]
    #[comptime] num_heads: usize,
    #[comptime] k_dim: usize,
    #[comptime] v_dim: usize,
) {
    let head = usize::cast_from(CUBE_POS_X);
    let j = usize::cast_from(UNIT_POS);

    let mut q_s = Shared::<[f32]>::new_slice(k_dim);
    let mut k_s = Shared::<[f32]>::new_slice(k_dim);
    let mut red = Shared::<[f32]>::new_slice(k_dim);

    let key_dim = num_heads * k_dim;
    q_s[j] = qkv[head * k_dim + j];
    k_s[j] = qkv[key_dim + head * k_dim + j];
    sync_cube();

    // 128 adds. Serial on one unit so every unit agrees on the norm.
    if j == 0 {
        let mut sum_q = 0.0f32;
        let mut sum_k = 0.0f32;
        for i in 0..k_dim {
            sum_q += q_s[i] * q_s[i];
            sum_k += k_s[i] * k_s[i];
        }
        red[0] = (sum_q + 1e-6).sqrt().recip();
        red[1] = (sum_k + 1e-6).sqrt().recip();
    }
    sync_cube();
    let inv_q = red[0];
    let inv_k = red[1];
    let scale = f32::cast_from(k_dim).powf(-0.5);

    q_s[j] = q_s[j] * inv_q * scale;
    k_s[j] = k_s[j] * inv_k;
    sync_cube();

    let beta = beta_g[head];
    let decay = beta_g[num_heads + head].exp();
    let v = qkv[key_dim * 2 + head * v_dim + j];

    let mut kv = 0.0f32;
    for i in 0..k_dim {
        let s = state[(head * k_dim + i) * v_dim + j] * decay;
        kv += s * k_s[i];
    }
    let delta = (v - kv) * beta;

    let mut acc = 0.0f32;
    for i in 0..k_dim {
        let s = state[(head * k_dim + i) * v_dim + j] * decay + k_s[i] * delta;
        new_state[(head * k_dim + i) * v_dim + j] = s;
        acc += s * q_s[i];
    }
    out[head * v_dim + j] = acc;
}

#[backend_extension(Cube, Fusion)]
pub trait CustomKernels: Backend {
    #[fusion(meta=|table: &TensorSpec, ids: &TensorSpec| {
        let n = ids.shape.num_elements();
        let [_v, dim] = table.shape.dims::<2>();
        TensorSpec::new(Shape::from([n, dim]), DType::F32)
    })]
    fn q8_embedding_lookup(table: QuantizedTensor<Self>, ids: IntTensor<Self>)
    -> FloatTensor<Self>;
    #[fusion(meta=|_hidden_states: &TensorSpec, weights: &TensorSpec| {
        let [_in_dim, out_dim] = weights.shape.dims::<2>();
        TensorSpec::new(Shape::from([out_dim]), DType::F32)
    })]
    fn q8_gemv(
        hidden_states: FloatTensor<Self>,
        weights: QuantizedTensor<Self>,
    ) -> FloatTensor<Self>;
    #[fusion(meta=|_hidden_states: &TensorSpec, weights: &TensorSpec| {
        let [_in_dim, out_dim] = weights.shape.dims::<2>();
        TensorSpec::new(Shape::from([out_dim]), DType::F32)
    })]
    fn q8_gemv_partial_sum(
        hidden_states: FloatTensor<Self>,
        weights: QuantizedTensor<Self>,
    ) -> FloatTensor<Self>;

    #[fusion(meta=|_hidden_states: &TensorSpec, w_b: &TensorSpec, _w_a: &TensorSpec, _a_log: &TensorSpec, _dt_bias: &TensorSpec| {
        let [_in_dim, n_heads] = w_b.shape.dims::<2>();
        TensorSpec::new(Shape::from([2 * n_heads]), DType::F32)
    })]
    fn gate_proj(
        hidden_states: FloatTensor<Self>,
        w_b: FloatTensor<Self>,
        w_a: FloatTensor<Self>,
        a_log: FloatTensor<Self>,
        dt_bias: FloatTensor<Self>,
    ) -> FloatTensor<Self>;

    #[fusion(meta=|_hidden_states: &TensorSpec, embeddings: &TensorSpec|{
        let [vocab_size, _dim] = embeddings.shape.dims::<2>();
        TensorSpec::new(Shape::from([vocab_size]), DType::F32)
    })]
    fn q8_lm_head(
        hidden_states: FloatTensor<Self>,
        embeddings: QuantizedTensor<Self>,
    ) -> FloatTensor<Self>;

    #[fusion(meta = |logits: &TensorSpec| {
        let [vocab_size] = logits.shape.dims::<1>();
        (TensorSpec::new(Shape::from([vocab_size / 8]), DType::F32), TensorSpec::new(Shape::from([vocab_size / 8]), DType::I32))
    })]
    fn argmax_partial(logits: FloatTensor<Self>) -> (FloatTensor<Self>, IntTensor<Self>);

    #[fusion(meta = |_partial: &TensorSpec, _partial_indices: &TensorSpec| {
        TensorSpec::new(Shape::from([1]), DType::I32)
    })]
    fn argmax_final(
        partial: FloatTensor<Self>,
        partial_indices: IntTensor<Self>,
    ) -> IntTensor<Self>;

    #[fusion(meta=|x: &TensorSpec, _weight: &TensorSpec, _eps: &f32| { x.clone() })]
    fn rms_norm(x: FloatTensor<Self>, weight: FloatTensor<Self>, eps: f32) -> FloatTensor<Self>;

    #[fusion(meta=|x: &TensorSpec, _gate: &TensorSpec, _weight: &TensorSpec, _eps: &f32| { x.clone() })]
    fn rms_norm_gated(
        x: FloatTensor<Self>,
        gate: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        eps: f32,
    ) -> FloatTensor<Self>;

    #[fusion(meta=|_qkv: &TensorSpec, _beta_g: &TensorSpec, state: &TensorSpec| {
        let [_, num_heads, _, v_dim] = state.shape.dims::<4>();
        (
            TensorSpec::new(Shape::from([num_heads * v_dim]), DType::F32),
            state.clone(),
        )
    })]
    fn gdn_decode(
        qkv: FloatTensor<Self>,
        beta_g: FloatTensor<Self>,
        state: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>);
}

/// Keep the upstream split target, then step down until every K row is covered by the
/// 8-wide unroll. `2048 / 12` would otherwise drop the tail of a 2048-wide projection.
fn gemv_splits(in_dim: usize, out_dim: usize) -> usize {
    let mut splits = (25_000 / out_dim).clamp(1, (in_dim / 80).max(1));
    while splits > 1 && (in_dim % splits != 0 || (in_dim / splits) % 8 != 0) {
        splits -= 1;
    }
    splits
}

fn launch_rmsnorm<const GATED: bool, const PLUS_ONE: bool>(
    x: FloatTensor<CubeBackend>,
    gate: Option<FloatTensor<CubeBackend>>,
    weight: FloatTensor<CubeBackend>,
    eps: f32,
) -> FloatTensor<CubeBackend> {
    let client = x.client.clone();
    let device = x.device.clone();
    let [rows, dim] = x.shape().dims::<2>();
    let [weight_dim] = weight.shape().dims::<1>();
    assert_eq!(
        weight_dim, dim,
        "rmsnorm weight length {weight_dim} != {dim}"
    );
    let output = CubeBackend::float_empty(Shape::from([rows, dim]), &device, FloatDType::F32);
    if rows == 0 {
        return output;
    }
    let gate = gate.unwrap_or_else(|| x.clone());
    let cubes = u32::try_from(rows.min(2048)).unwrap();
    rmsnorm_kernel::launch(
        &client,
        CubeCount::Static(cubes, 1, 1),
        CubeDim::new_3d(256, 1, 1),
        x.into_tensor_arg(),
        gate.into_tensor_arg(),
        weight.into_tensor_arg(),
        output.clone().into_tensor_arg(),
        eps,
        u32::try_from(rows).unwrap(),
        cubes,
        dim,
        GATED,
        PLUS_ONE,
    );
    output
}

impl CustomKernels for CubeBackend {
    fn q8_embedding_lookup(
        table: QuantizedTensor<Self>,
        ids: IntTensor<Self>,
    ) -> FloatTensor<Self> {
        let (values, scales) = table
            .quantized_handles()
            .expect("Embedding table must be QFloat");
        let client = values.client.clone();
        let device = values.device.clone();

        let [_vocab, dim] = table.shape().dims::<2>();
        let [_b, n] = ids.shape().dims::<2>();

        // Have to import burn::backend::ops:FloatTensorOps
        let output = Self::float_empty(Shape::from([n, dim]), &device, FloatDType::F32);
        let num_elements = (n * dim) as u32;
        let cube_dim = CubeDim::new_3d(256, 1, 1);
        let cube_count = CubeCount::Static(num_elements.div_ceil(cube_dim.x), 1, 1);

        q8_embedding_lookup_kernel::launch(
            &client,
            cube_count,
            cube_dim,
            values.into_tensor_arg(),
            ids.into_tensor_arg(),
            scales.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            dim,
        );

        output
    }
    fn q8_gemv(
        hidden_states: FloatTensor<Self>,
        weights: QuantizedTensor<Self>,
    ) -> FloatTensor<Self> {
        let (values, scales) = weights.quantized_handles().expect("weights must be QFloat");
        let client = values.client.clone();
        let device = values.device.clone();

        let [in_dim, out_dim] = weights.shape().dims::<2>();
        let output = Self::float_empty(Shape::from([out_dim]), &device, FloatDType::F32);
        let cube_dim = CubeDim::new_3d(256, 1, 1);
        let cube_count = CubeCount::Static((out_dim as u32).div_ceil(cube_dim.x), 1, 1);

        q8_gemv_kernel::launch(
            &client,
            cube_count,
            cube_dim,
            hidden_states.into_tensor_arg(),
            values.into_tensor_arg(),
            scales.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            in_dim,
            out_dim,
        );
        output
    }

    fn q8_gemv_partial_sum(
        hidden_states: FloatTensor<Self>,
        weights: QuantizedTensor<Self>,
    ) -> FloatTensor<Self> {
        let (values, scales) = weights.quantized_handles().expect("weights must be QFloat");
        let client = values.client.clone();
        let device = values.device.clone();

        let [in_dim, out_dim] = weights.shape().dims::<2>();
        assert!(
            in_dim % 8 == 0,
            "Q8 GEMV needs in_dim % 8 == 0, got {in_dim}x{out_dim}"
        );
        let num_splits = gemv_splits(in_dim, out_dim);
        // Partial sum output
        let output =
            Self::float_empty(Shape::from([num_splits, out_dim]), &device, FloatDType::F32);
        let cube_dim = CubeDim::new_3d(256, 1, 1);
        let cube_count =
            CubeCount::Static((out_dim as u32).div_ceil(cube_dim.x), num_splits as u32, 1);

        q8_gemv_partial_sum_kernel::launch(
            &client,
            cube_count,
            cube_dim,
            hidden_states.into_tensor_arg(),
            values.into_tensor_arg(),
            scales.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            in_dim,
            out_dim,
            num_splits,
        );
        let reduced = Self::float_empty(Shape::from([out_dim]), &device, FloatDType::F32);
        let reduce_count = CubeCount::Static((out_dim as u32).div_ceil(cube_dim.x), 1, 1);

        q8_gemv_sum_reduction_kernel::launch(
            &client,
            reduce_count,
            cube_dim,
            output.into_tensor_arg(),
            reduced.clone().into_tensor_arg(),
            out_dim,
            num_splits,
        );
        reduced
    }

    fn gate_proj(
        hidden_states: FloatTensor<Self>,
        w_b: FloatTensor<Self>,
        w_a: FloatTensor<Self>,
        a_log: FloatTensor<Self>,
        dt_bias: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let client = hidden_states.client.clone();
        let device = hidden_states.device.clone();
        let [in_dim, n_heads] = w_b.shape().dims::<2>();
        let output = Self::float_empty(Shape::from([2 * n_heads]), &device, FloatDType::F32);
        let num_splits = 8;

        let num_units = 2 * n_heads * num_splits;
        let cube_dim = CubeDim::new_3d(num_units as u32, 1, 1);
        let cube_count = CubeCount::Static(1, 1, 1);

        gate_proj_kernel::launch(
            &client,
            cube_count,
            cube_dim,
            hidden_states.into_tensor_arg(),
            w_b.into_tensor_arg(),
            w_a.into_tensor_arg(),
            a_log.into_tensor_arg(),
            dt_bias.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            in_dim,
            n_heads,
            num_splits,
        );

        output
    }

    fn q8_lm_head(
        hidden_states: FloatTensor<Self>,
        embeddings: QuantizedTensor<Self>,
    ) -> FloatTensor<Self> {
        let (values, scales) = embeddings
            .quantized_handles()
            .expect("Embeddings must be Q8F quantized!");
        let client = values.client.clone();
        let device = values.device.clone();
        let [vocab_size, dim] = embeddings.shape().dims::<2>();
        assert!(
            dim % 128 == 0,
            "Q8 LM head needs dim % 128 == 0 so each lane owns whole words, got {dim}"
        );
        let output = Self::float_empty(Shape::from([vocab_size]), &device, FloatDType::F32);

        let num_units = vocab_size * 32;
        let cube_dim = CubeDim::new_3d(256, 1, 1);
        let cube_count = CubeCount::Static((num_units as u32).div_ceil(cube_dim.x), 1, 1);

        q8_lm_head_kernel::launch(
            &client,
            cube_count,
            cube_dim,
            hidden_states.into_tensor_arg(),
            values.into_tensor_arg(),
            scales.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            dim,
        );

        output
    }

    fn argmax_partial(logits: FloatTensor<Self>) -> (FloatTensor<Self>, IntTensor<Self>) {
        let device = logits.device.clone();
        let client = logits.client.clone();
        let [vocab_size] = logits.shape().dims::<1>();
        let num_scans = 8;
        let output = Self::float_empty(
            Shape::from([vocab_size / num_scans]),
            &device,
            FloatDType::F32,
        );
        // Have to include burn::backend::ops::IntTensorOps
        let output_indices = Self::int_empty(
            Shape::from([vocab_size / num_scans]),
            &device,
            IntDType::I32,
        );
        let num_units = vocab_size / num_scans;
        let cube_dim = CubeDim::new_3d(256, 1, 1);
        let cube_count = CubeCount::Static((num_units as u32).div_ceil(cube_dim.x), 1, 1);

        argmax_partial_kernel::launch(
            &client,
            cube_count,
            cube_dim,
            logits.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            output_indices.clone().into_tensor_arg(),
            vocab_size,
            num_scans,
        );

        (output, output_indices)
    }

    fn argmax_final(
        partial: FloatTensor<Self>,
        partial_indices: IntTensor<Self>,
    ) -> IntTensor<Self> {
        let device = partial.device.clone();
        let client = partial.client.clone();
        let [dim] = partial.shape().dims::<1>();

        let output = Self::int_empty(Shape::from([1]), &device, IntDType::I32);
        let cube_dim = CubeDim::new_3d(256, 1, 1);
        let cube_count = CubeCount::Static(1, 1, 1);

        argmax_final::launch(
            &client,
            cube_count,
            cube_dim,
            partial.into_tensor_arg(),
            partial_indices.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            dim,
        );
        output
    }

    fn rms_norm(x: FloatTensor<Self>, weight: FloatTensor<Self>, eps: f32) -> FloatTensor<Self> {
        launch_rmsnorm::<false, true>(x, None, weight, eps)
    }

    fn rms_norm_gated(
        x: FloatTensor<Self>,
        gate: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        eps: f32,
    ) -> FloatTensor<Self> {
        launch_rmsnorm::<true, false>(x, Some(gate), weight, eps)
    }

    fn gdn_decode(
        qkv: FloatTensor<Self>,
        beta_g: FloatTensor<Self>,
        state: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        let client = qkv.client.clone();
        let device = qkv.device.clone();
        let [_, num_heads, k_dim, v_dim] = state.shape().dims::<4>();
        let out = Self::float_empty(Shape::from([num_heads * v_dim]), &device, FloatDType::F32);
        let new_state = Self::float_empty(state.shape(), &device, FloatDType::F32);

        gdn_decode_kernel::launch(
            &client,
            CubeCount::Static(num_heads as u32, 1, 1),
            CubeDim::new_3d(v_dim as u32, 1, 1),
            qkv.into_tensor_arg(),
            beta_g.into_tensor_arg(),
            state.into_tensor_arg(),
            out.clone().into_tensor_arg(),
            new_state.clone().into_tensor_arg(),
            num_heads,
            k_dim,
            v_dim,
        );
        (out, new_state)
    }
}

/// embeddings: (vocab_size, emb_dim)
///
/// input_ids: (batch_size, seq_len)
///
/// Currently only works with batch size 1.
/// Make sure embeddings are tensor-scale Q8F quantized
pub fn q8_embedding_lookup(
    embeddings: burn::Tensor<2>,
    input_ids: burn::Tensor<2, burn::tensor::Int>,
) -> burn::Tensor<2> {
    let out = <Dispatch as CustomKernels>::q8_embedding_lookup(
        embeddings.into_dispatch(),
        input_ids.into_dispatch(),
    );
    let out = burn::Tensor::<2>::from_dispatch(out);
    out
}

pub fn q8_gemv(hidden_states: burn::Tensor<3>, weights: burn::Tensor<2>) -> burn::Tensor<3> {
    let out = <Dispatch as CustomKernels>::q8_gemv(
        hidden_states.squeeze_dims::<1>(&[0, 1]).into_dispatch(),
        weights.into_dispatch(),
    );
    let out = burn::Tensor::<1>::from_dispatch(out).unsqueeze_dims::<3>(&[0, 1]);
    out
}

pub fn q8_gemv_partial_sum(
    hidden_states: burn::Tensor<3>,
    weights: burn::Tensor<2>,
) -> burn::Tensor<3> {
    let out = <Dispatch as CustomKernels>::q8_gemv_partial_sum(
        hidden_states.squeeze_dims::<1>(&[0, 1]).into_dispatch(),
        weights.into_dispatch(),
    );
    let out = burn::Tensor::<1>::from_dispatch(out).unsqueeze_dims::<3>(&[0, 1]);
    out
}

/// Kernels index their inputs as row-major; a sliced or permuted view is not.
fn materialize<const D: usize>(tensor: burn::Tensor<D>) -> burn::Tensor<D> {
    let dims = tensor.dims();
    let n = dims.iter().product::<usize>();
    tensor.reshape([n]).reshape(dims)
}

/// Normalize over the last axis. Matches `RMSNorm`: scale is `(1 + weight)`.
pub fn rms_norm<const D: usize>(
    x: burn::Tensor<D>,
    weight: burn::Tensor<1>,
    eps: f32,
) -> burn::Tensor<D> {
    let dims = x.dims();
    let dim = dims[D - 1];
    let rows = dims.iter().product::<usize>() / dim;
    let flat = materialize(x).reshape([rows, dim]);
    let out =
        <Dispatch as CustomKernels>::rms_norm(flat.into_dispatch(), weight.into_dispatch(), eps);
    burn::Tensor::<2>::from_dispatch(out).reshape(dims)
}

/// RMSNorm without the `1 +` on the weight, then `silu(gate)`.
pub fn rms_norm_gated(
    x: burn::Tensor<3>,
    gate: burn::Tensor<3>,
    weight: burn::Tensor<1>,
    eps: f32,
) -> burn::Tensor<3> {
    let dims = x.dims();
    assert_eq!(gate.dims(), dims);
    let dim = dims[2];
    let rows = dims[0] * dims[1];
    let flat_x = materialize(x).reshape([rows, dim]);
    let flat_gate = materialize(gate).reshape([rows, dim]);
    let out = <Dispatch as CustomKernels>::rms_norm_gated(
        flat_x.into_dispatch(),
        flat_gate.into_dispatch(),
        weight.into_dispatch(),
        eps,
    );
    burn::Tensor::<2>::from_dispatch(out).reshape(dims)
}

pub fn gate_proj_raw(
    hidden_states: burn::Tensor<3>,
    w_b: burn::Tensor<2>,
    w_a: burn::Tensor<2>,
    a_log: burn::Tensor<1>,
    dt_bias: burn::Tensor<1>,
) -> burn::Tensor<1> {
    let out = <Dispatch as CustomKernels>::gate_proj(
        hidden_states.squeeze_dims::<1>(&[0, 1]).into_dispatch(),
        w_b.into_dispatch(),
        w_a.into_dispatch(),
        a_log.into_dispatch(),
        dt_bias.into_dispatch(),
    );
    burn::Tensor::<1>::from_dispatch(out)
}

/// One decode step of Gated DeltaNet. `beta_g` is the raw `gate_proj` output
/// (`[beta | g]`, length `2 * heads`). State is `[1, heads, k_dim, v_dim]`.
pub fn gdn_decode(
    qkv: burn::Tensor<3>,
    beta_g: burn::Tensor<1>,
    state: burn::Tensor<4>,
) -> (burn::Tensor<3>, burn::Tensor<4>) {
    let qkv = materialize(qkv);
    let state = materialize(state);
    let (out, new_state) = <Dispatch as CustomKernels>::gdn_decode(
        qkv.squeeze_dims::<1>(&[0, 1]).into_dispatch(),
        beta_g.into_dispatch(),
        state.into_dispatch(),
    );
    (
        burn::Tensor::<1>::from_dispatch(out).unsqueeze_dims::<3>(&[0, 1]),
        burn::Tensor::<4>::from_dispatch(new_state),
    )
}

pub fn gate_proj(
    hidden_states: burn::Tensor<3>,
    w_b: burn::Tensor<2>,
    w_a: burn::Tensor<2>,
    a_log: burn::Tensor<1>,
    dt_bias: burn::Tensor<1>,
) -> (burn::Tensor<3>, burn::Tensor<3>) {
    let out = gate_proj_raw(hidden_states, w_b, w_a, a_log, dt_bias);
    let beta_gate_vec = out.chunk(2, 0);
    (
        beta_gate_vec[0].clone().unsqueeze_dims::<3>(&[0, 1]),
        beta_gate_vec[1].clone().unsqueeze_dims::<3>(&[0, 1]),
    )
}

pub fn q8_lm_head(hidden_states: burn::Tensor<3>, embeddings: burn::Tensor<2>) -> burn::Tensor<3> {
    let out = <Dispatch as CustomKernels>::q8_lm_head(
        hidden_states.squeeze_dims::<1>(&[0, 1]).into_dispatch(),
        embeddings.into_dispatch(),
    );
    let out = burn::Tensor::<1>::from_dispatch(out);
    out.unsqueeze_dims::<3>(&[0, 1])
}

pub fn argmax(logits: burn::Tensor<1>) -> burn::Tensor<1, burn::tensor::Int> {
    let (partial, partial_indices) =
        <Dispatch as CustomKernels>::argmax_partial(logits.into_dispatch());
    let index = <Dispatch as CustomKernels>::argmax_final(partial, partial_indices);
    let out = burn::Tensor::<1, burn::tensor::Int>::from_dispatch(index);
    out
}

#[cfg(test)]
mod tests {
    use burn::tensor::{
        Device, Distribution, Tensor,
        activation::silu,
        quantization::{QuantScheme, ScaleDtype},
        s,
    };
    use cubecl::quant::scheme::QuantValue;

    use super::{q8_gemv, q8_gemv_partial_sum, q8_lm_head, rms_norm, rms_norm_gated};

    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
    }

    fn scheme() -> QuantScheme {
        QuantScheme::default()
            .per_block([64], ScaleDtype::F32)
            .with_value(QuantValue::Q8F)
    }

    #[test]
    fn gemv_matches_dequantized_matmul() {
        let device = Device::wgpu(Default::default());
        let scheme = scheme();
        // Every decode projection shape of Qwen3.5-2B, plus a small one.
        for (in_dim, out_dim) in [
            (128usize, 256usize),
            (2048, 512),
            (2048, 2048),
            (2048, 4096),
            (2048, 6144),
            (6144, 2048),
        ] {
            let weight =
                Tensor::<2>::random([in_dim, out_dim], Distribution::Uniform(-0.5, 0.5), &device)
                    .quantize_dynamic(&scheme);
            let x = Tensor::<3>::random([1, 1, in_dim], Distribution::Uniform(-1.0, 1.0), &device);
            let reference = x
                .clone()
                .matmul(weight.clone().dequantize().unsqueeze_dim::<3>(0));
            let reference = reference.into_data().try_to_vec::<f32>().unwrap();
            for (name, got) in [
                ("gemv", q8_gemv(x.clone(), weight.clone())),
                ("partial", q8_gemv_partial_sum(x, weight)),
            ] {
                let diff = max_abs(&got.into_data().try_to_vec::<f32>().unwrap(), &reference);
                assert!(diff < 1e-2, "{name} {in_dim}x{out_dim} diff {diff}");
            }
        }
    }

    #[test]
    fn lm_head_matches_dequantized_matmul() {
        let device = Device::wgpu(Default::default());
        // Not a multiple of 8 rows, so the last cube has idle rows.
        let vocab = 1001usize;
        let dim = 2048usize;
        let table = Tensor::<2>::random([vocab, dim], Distribution::Uniform(-0.5, 0.5), &device)
            .quantize_dynamic(&scheme());
        let x = Tensor::<3>::random([1, 1, dim], Distribution::Uniform(-1.0, 1.0), &device);
        let reference = x.clone().matmul(
            table
                .clone()
                .dequantize()
                .swap_dims(0, 1)
                .unsqueeze_dim::<3>(0),
        );
        let got = q8_lm_head(x, table);
        let diff = max_abs(
            &got.into_data().try_to_vec::<f32>().unwrap(),
            &reference.into_data().try_to_vec::<f32>().unwrap(),
        );
        assert!(diff < 1e-2, "lm_head diff {diff}");
    }

    #[test]
    fn rmsnorm_matches_burn_formula() {
        let device = Device::wgpu(Default::default());
        // 5000 rows: cubes loop over more than one row.
        for (rows, dim) in [(1usize, 2048usize), (4, 256), (3, 32), (5000, 32)] {
            let x = Tensor::<2>::random([rows, dim], Distribution::Uniform(-1.0, 1.0), &device);
            let weight = Tensor::<1>::random([dim], Distribution::Uniform(-0.25, 0.25), &device);
            let eps = 1e-6;
            let reference = x.clone()
                * (x.clone().square().mean_dim(1) + eps).sqrt().recip()
                * (weight.clone() + 1.0).unsqueeze_dim::<2>(0);
            let got = rms_norm(x, weight, eps);
            let diff = max_abs(
                &got.into_data().try_to_vec::<f32>().unwrap(),
                &reference.into_data().try_to_vec::<f32>().unwrap(),
            );
            assert!(diff < 1e-4, "rmsnorm {rows}x{dim} diff {diff}");
        }

        let wide = Tensor::<2>::random([4, 64], Distribution::Uniform(-1.0, 1.0), &device);
        let x = wide.slice(s![.., 0..32]);
        let weight = Tensor::<1>::random([32], Distribution::Uniform(-0.25, 0.25), &device);
        let eps = 1e-6;
        let reference = x.clone()
            * (x.clone().square().mean_dim(1) + eps).sqrt().recip()
            * (weight.clone() + 1.0).unsqueeze_dim::<2>(0);
        let got = rms_norm(x, weight, eps);
        let diff = max_abs(
            &got.into_data().try_to_vec::<f32>().unwrap(),
            &reference.into_data().try_to_vec::<f32>().unwrap(),
        );
        assert!(diff < 1e-4, "sliced rmsnorm diff {diff}");
    }

    #[test]
    fn gated_rmsnorm_matches_burn_formula() {
        let device = Device::wgpu(Default::default());
        let x = Tensor::<3>::random([4, 2, 128], Distribution::Uniform(-1.0, 1.0), &device);
        let gate = Tensor::<3>::random([4, 2, 128], Distribution::Uniform(-1.0, 1.0), &device);
        let weight = Tensor::<1>::random([128], Distribution::Uniform(0.5, 1.5), &device);
        let eps = 1e-6;
        let normed = x.clone() * (x.clone().square().mean_dim(2) + eps).sqrt().recip();
        let reference = normed * weight.clone().unsqueeze_dims::<3>(&[0, 1]) * silu(gate.clone());
        let got = rms_norm_gated(x, gate, weight, eps);
        let diff = max_abs(
            &got.into_data().try_to_vec::<f32>().unwrap(),
            &reference.into_data().try_to_vec::<f32>().unwrap(),
        );
        assert!(diff < 1e-4, "gated rmsnorm diff {diff}");
    }
}
