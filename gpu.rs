//! GPU Backend Implementation
//!
//! High-performance GPU compute using WebGPU (wgpu).
//! Ported from TharvexalFrameworkDL with clean modular design.

use super::ComputeBackend;
use crate::tensor::{Tensor, TensorError, TensorResult};
use std::sync::{Arc, Mutex, OnceLock};
use std::collections::HashMap;
use wgpu::util::DeviceExt;

// =============================================================================
// WGSL SHADERS
// =============================================================================

mod shaders {
    /// Element-wise addition
    pub const ADD: &str = r#"
        @group(0) @binding(0) var<storage, read> a: array<f32>;
        @group(0) @binding(1) var<storage, read> b: array<f32>;
        @group(0) @binding(2) var<storage, read_write> r: array<f32>;
        @compute @workgroup_size(256) 
        fn main(@builtin(global_invocation_id) g: vec3<u32>) { 
            if g.x < arrayLength(&r) { r[g.x] = a[g.x] + b[g.x]; } 
        }
    "#;
    
    /// Element-wise subtraction
    pub const SUB: &str = r#"
        @group(0) @binding(0) var<storage, read> a: array<f32>;
        @group(0) @binding(1) var<storage, read> b: array<f32>;
        @group(0) @binding(2) var<storage, read_write> r: array<f32>;
        @compute @workgroup_size(256) 
        fn main(@builtin(global_invocation_id) g: vec3<u32>) { 
            if g.x < arrayLength(&r) { r[g.x] = a[g.x] - b[g.x]; } 
        }
    "#;
    
    /// Element-wise multiplication
    pub const MUL: &str = r#"
        @group(0) @binding(0) var<storage, read> a: array<f32>;
        @group(0) @binding(1) var<storage, read> b: array<f32>;
        @group(0) @binding(2) var<storage, read_write> r: array<f32>;
        @compute @workgroup_size(256) 
        fn main(@builtin(global_invocation_id) g: vec3<u32>) { 
            if g.x < arrayLength(&r) { r[g.x] = a[g.x] * b[g.x]; } 
        }
    "#;
    
    /// Element-wise division
    pub const DIV: &str = r#"
        @group(0) @binding(0) var<storage, read> a: array<f32>;
        @group(0) @binding(1) var<storage, read> b: array<f32>;
        @group(0) @binding(2) var<storage, read_write> r: array<f32>;
        @compute @workgroup_size(256) 
        fn main(@builtin(global_invocation_id) g: vec3<u32>) { 
            if g.x < arrayLength(&r) { r[g.x] = a[g.x] / b[g.x]; } 
        }
    "#;
    
    /// ReLU activation
    pub const RELU: &str = r#"
        @group(0) @binding(0) var<storage, read> i: array<f32>;
        @group(0) @binding(1) var<storage, read_write> o: array<f32>;
        @compute @workgroup_size(256) 
        fn main(@builtin(global_invocation_id) g: vec3<u32>) { 
            if g.x < arrayLength(&o) { o[g.x] = max(0.0, i[g.x]); } 
        }
    "#;
    
    /// GELU activation
    pub const GELU: &str = r#"
        @group(0) @binding(0) var<storage, read> i: array<f32>;
        @group(0) @binding(1) var<storage, read_write> o: array<f32>;
        @compute @workgroup_size(256) 
        fn main(@builtin(global_invocation_id) g: vec3<u32>) { 
            if g.x < arrayLength(&o) { 
                let x = i[g.x]; 
                o[g.x] = 0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x))); 
            } 
        }
    "#;
    
    /// Sigmoid activation
    pub const SIGMOID: &str = r#"
        @group(0) @binding(0) var<storage, read> i: array<f32>;
        @group(0) @binding(1) var<storage, read_write> o: array<f32>;
        @compute @workgroup_size(256) 
        fn main(@builtin(global_invocation_id) g: vec3<u32>) { 
            if g.x < arrayLength(&o) { o[g.x] = 1.0 / (1.0 + exp(-i[g.x])); } 
        }
    "#;
    
    /// SiLU (Swish) activation
    pub const SILU: &str = r#"
        @group(0) @binding(0) var<storage, read> i: array<f32>;
        @group(0) @binding(1) var<storage, read_write> o: array<f32>;
        @compute @workgroup_size(256) 
        fn main(@builtin(global_invocation_id) g: vec3<u32>) { 
            if g.x < arrayLength(&o) { let x = i[g.x]; o[g.x] = x / (1.0 + exp(-x)); } 
        }
    "#;
    
    /// Tanh activation
    pub const TANH: &str = r#"
        @group(0) @binding(0) var<storage, read> i: array<f32>;
        @group(0) @binding(1) var<storage, read_write> o: array<f32>;
        @compute @workgroup_size(256) 
        fn main(@builtin(global_invocation_id) g: vec3<u32>) { 
            if g.x < arrayLength(&o) { o[g.x] = tanh(i[g.x]); } 
        }
    "#;
    
    /// Exponential
    pub const EXP: &str = r#"
        @group(0) @binding(0) var<storage, read> i: array<f32>;
        @group(0) @binding(1) var<storage, read_write> o: array<f32>;
        @compute @workgroup_size(256) 
        fn main(@builtin(global_invocation_id) g: vec3<u32>) { 
            if g.x < arrayLength(&o) { o[g.x] = exp(i[g.x]); } 
        }
    "#;
    
    /// Natural log
    pub const LOG: &str = r#"
        @group(0) @binding(0) var<storage, read> i: array<f32>;
        @group(0) @binding(1) var<storage, read_write> o: array<f32>;
        @compute @workgroup_size(256) 
        fn main(@builtin(global_invocation_id) g: vec3<u32>) { 
            if g.x < arrayLength(&o) { o[g.x] = log(i[g.x]); } 
        }
    "#;
    
    /// Square root
    pub const SQRT: &str = r#"
        @group(0) @binding(0) var<storage, read> i: array<f32>;
        @group(0) @binding(1) var<storage, read_write> o: array<f32>;
        @compute @workgroup_size(256) 
        fn main(@builtin(global_invocation_id) g: vec3<u32>) { 
            if g.x < arrayLength(&o) { o[g.x] = sqrt(i[g.x]); } 
        }
    "#;
    
    /// Square (power of 2)
    pub const POW2: &str = r#"
        @group(0) @binding(0) var<storage, read> i: array<f32>;
        @group(0) @binding(1) var<storage, read_write> o: array<f32>;
        @compute @workgroup_size(256) 
        fn main(@builtin(global_invocation_id) g: vec3<u32>) { 
            if g.x < arrayLength(&o) { let x = i[g.x]; o[g.x] = x * x; } 
        }
    "#;
    
    /// Negate
    pub const NEG: &str = r#"
        @group(0) @binding(0) var<storage, read> i: array<f32>;
        @group(0) @binding(1) var<storage, read_write> o: array<f32>;
        @compute @workgroup_size(256) 
        fn main(@builtin(global_invocation_id) g: vec3<u32>) { 
            if g.x < arrayLength(&o) { o[g.x] = -i[g.x]; } 
        }
    "#;
    
    /// Absolute value
    pub const ABS: &str = r#"
        @group(0) @binding(0) var<storage, read> i: array<f32>;
        @group(0) @binding(1) var<storage, read_write> o: array<f32>;
        @compute @workgroup_size(256) 
        fn main(@builtin(global_invocation_id) g: vec3<u32>) { 
            if g.x < arrayLength(&o) { o[g.x] = abs(i[g.x]); } 
        }
    "#;
    
    /// Tiled matrix multiplication (16x16 tiles)
    pub const MATMUL: &str = r#"
        struct Params { M: u32, N: u32, K: u32, }
        @group(0) @binding(0) var<uniform> params: Params;
        @group(0) @binding(1) var<storage, read> A: array<f32>;
        @group(0) @binding(2) var<storage, read> B: array<f32>;
        @group(0) @binding(3) var<storage, read_write> C: array<f32>;
        
        const TILE_SIZE: u32 = 16u;
        var<workgroup> tileA: array<array<f32, 16>, 16>;
        var<workgroup> tileB: array<array<f32, 16>, 16>;
        
        @compute @workgroup_size(16, 16) 
        fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
            let row = gid.y;
            let col = gid.x;
            let localRow = lid.y;
            let localCol = lid.x;
            
            var acc: f32 = 0.0;
            let numTiles = (params.K + TILE_SIZE - 1u) / TILE_SIZE;
            
            for (var t: u32 = 0u; t < numTiles; t = t + 1u) {
                let tiledCol = t * TILE_SIZE + localCol;
                let tiledRow = t * TILE_SIZE + localRow;
                
                if (row < params.M && tiledCol < params.K) {
                    tileA[localRow][localCol] = A[row * params.K + tiledCol];
                } else {
                    tileA[localRow][localCol] = 0.0;
                }
                
                if (tiledRow < params.K && col < params.N) {
                    tileB[localRow][localCol] = B[tiledRow * params.N + col];
                } else {
                    tileB[localRow][localCol] = 0.0;
                }
                
                workgroupBarrier();
                
                for (var k: u32 = 0u; k < TILE_SIZE; k = k + 1u) {
                    acc = acc + tileA[localRow][k] * tileB[k][localCol];
                }
                
                workgroupBarrier();
            }
            
            if (row < params.M && col < params.N) {
                C[row * params.N + col] = acc;
            }
        }
    "#;
    
    /// Transpose
    pub const TRANSPOSE: &str = r#"
        struct Params { rows: u32, cols: u32, }
        @group(0) @binding(0) var<uniform> params: Params;
        @group(0) @binding(1) var<storage, read> input: array<f32>;
        @group(0) @binding(2) var<storage, read_write> output: array<f32>;
        
        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            let idx = gid.x;
            if (idx >= params.rows * params.cols) { return; }
            
            let r = idx / params.cols;
            let c = idx % params.cols;
            let new_idx = c * params.rows + r;
            output[new_idx] = input[idx];
        }
    "#;
    
    /// Row-wise softmax
    pub const SOFTMAX: &str = r#"
        struct Params { rows: u32, cols: u32, }
        @group(0) @binding(0) var<uniform> params: Params;
        @group(0) @binding(1) var<storage, read> input: array<f32>;
        @group(0) @binding(2) var<storage, read_write> output: array<f32>;
        
        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            let row = gid.x;
            if (row >= params.rows) { return; }
            
            let offset = row * params.cols;
            
            // Find max for numerical stability
            var max_val: f32 = input[offset];
            for (var i: u32 = 1u; i < params.cols; i = i + 1u) {
                max_val = max(max_val, input[offset + i]);
            }
            
            // Compute exp and sum
            var sum_exp: f32 = 0.0;
            for (var i: u32 = 0u; i < params.cols; i = i + 1u) {
                let exp_val = exp(input[offset + i] - max_val);
                output[offset + i] = exp_val;
                sum_exp = sum_exp + exp_val;
            }
            
            // Normalize
            let inv_sum = 1.0 / sum_exp;
            for (var i: u32 = 0u; i < params.cols; i = i + 1u) {
                output[offset + i] = output[offset + i] * inv_sum;
            }
        }
    "#;
    
    /// Layer normalization
    pub const LAYERNORM: &str = r#"
        struct Params { rows: u32, cols: u32, eps: f32, }
        @group(0) @binding(0) var<uniform> params: Params;
        @group(0) @binding(1) var<storage, read> input: array<f32>;
        @group(0) @binding(2) var<storage, read> gamma: array<f32>;
        @group(0) @binding(3) var<storage, read> beta: array<f32>;
        @group(0) @binding(4) var<storage, read_write> output: array<f32>;
        
        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            let row = gid.x;
            if (row >= params.rows) { return; }
            
            let offset = row * params.cols;
            
            // Compute mean
            var mean: f32 = 0.0;
            for (var i: u32 = 0u; i < params.cols; i = i + 1u) {
                mean = mean + input[offset + i];
            }
            mean = mean / f32(params.cols);
            
            // Compute variance
            var variance: f32 = 0.0;
            for (var i: u32 = 0u; i < params.cols; i = i + 1u) {
                let diff = input[offset + i] - mean;
                variance = variance + diff * diff;
            }
            variance = variance / f32(params.cols);
            
            // Normalize and scale
            let inv_std = 1.0 / sqrt(variance + params.eps);
            for (var i: u32 = 0u; i < params.cols; i = i + 1u) {
                let norm = (input[offset + i] - mean) * inv_std;
                output[offset + i] = gamma[i] * norm + beta[i];
            }
        }
    "#;
    
    /// RMS normalization
    pub const RMSNORM: &str = r#"
        struct Params { rows: u32, cols: u32, eps: f32, }
        @group(0) @binding(0) var<uniform> params: Params;
        @group(0) @binding(1) var<storage, read> input: array<f32>;
        @group(0) @binding(2) var<storage, read> weight: array<f32>;
        @group(0) @binding(3) var<storage, read_write> output: array<f32>;
        
        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            let row = gid.x;
            if (row >= params.rows) { return; }
            
            let offset = row * params.cols;
            
            // Compute RMS
            var sum_sq: f32 = 0.0;
            for (var i: u32 = 0u; i < params.cols; i = i + 1u) {
                let v = input[offset + i];
                sum_sq = sum_sq + v * v;
            }
            let rms = sqrt(sum_sq / f32(params.cols) + params.eps);
            let inv_rms = 1.0 / rms;
            
            // Normalize and scale
            for (var i: u32 = 0u; i < params.cols; i = i + 1u) {
                output[offset + i] = weight[i] * input[offset + i] * inv_rms;
            }
        }
    "#;
}

// =============================================================================
// GPU CONTEXT
// =============================================================================

/// GPU context holding device, queue, and pipeline cache
struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_info: wgpu::AdapterInfo,
    pipeline_cache: Mutex<HashMap<String, Arc<wgpu::ComputePipeline>>>,
}

impl GpuContext {
    fn new() -> Result<Self, String> {
        pollster::block_on(Self::init_async())
    }
    
    async fn init_async() -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or("GPU adaptörü bulunamadı")?;
        
        let adapter_info = adapter.get_info();
        
        // Request larger limits for big matrices
        let mut limits = wgpu::Limits::default();
        limits.max_storage_buffer_binding_size = 1 << 30; // 1GB
        limits.max_buffer_size = 1 << 30;
        
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("TürkçeKod GPU"),
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .map_err(|e| format!("GPU cihazı alınamadı: {:?}", e))?;
        
        Ok(Self {
            device,
            queue,
            adapter_info,
            pipeline_cache: Mutex::new(HashMap::new()),
        })
    }
    
    fn get_info(&self) -> String {
        format!("{} ({:?})", self.adapter_info.name, self.adapter_info.backend)
    }
    
    fn get_or_create_pipeline(&self, shader_code: &str, label: &str) -> Arc<wgpu::ComputePipeline> {
        let mut cache = self.pipeline_cache.lock().unwrap();
        
        if let Some(pipeline) = cache.get(label) {
            return pipeline.clone();
        }
        
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(shader_code.into()),
        });
        
        let pipeline = Arc::new(self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: None, // Auto layout
            module: &shader,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        }));
        
        cache.insert(label.to_string(), pipeline.clone());
        pipeline
    }
    
    /// Create a storage buffer from data
    fn create_buffer(&self, data: &[f32], usage: wgpu::BufferUsages) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(data),
            usage,
        })
    }
    
    /// Create an empty storage buffer
    fn create_empty_buffer(&self, size: usize, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (size * 4) as u64,
            usage,
            mapped_at_creation: false,
        })
    }
    
    /// Read data from GPU buffer
    fn read_buffer(&self, buffer: &wgpu::Buffer, size: usize) -> Vec<f32> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (size * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, (size * 4) as u64);
        self.queue.submit(Some(encoder.finish()));
        
        let slice = staging.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        self.device.poll(wgpu::Maintain::Wait);
        pollster::block_on(rx).unwrap().unwrap();
        
        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        result
    }
}

// Global GPU context (lazy initialized)
static GPU_CONTEXT: OnceLock<Result<GpuContext, String>> = OnceLock::new();

fn get_gpu() -> Result<&'static GpuContext, String> {
    GPU_CONTEXT
        .get_or_init(|| GpuContext::new())
        .as_ref()
        .map_err(|e| e.clone())
}

// =============================================================================
// GPU BACKEND
// =============================================================================

/// GPU compute backend using WebGPU
pub struct GpuBackend {
    context: &'static GpuContext,
}

impl GpuBackend {
    /// Try to create a GPU backend (returns None if GPU unavailable)
    pub fn try_new() -> Option<Self> {
        get_gpu().ok().map(|ctx| Self { context: ctx })
    }
    
    /// Run a unary operation (1 input, 1 output)
    fn run_unary(&self, input: &Tensor, shader: &str, label: &str) -> Tensor {
        let size = input.numel();
        let pipeline = self.context.get_or_create_pipeline(shader, label);
        
        let input_buffer = self.context.create_buffer(
            &input.data(),
            wgpu::BufferUsages::STORAGE,
        );
        let output_buffer = self.context.create_empty_buffer(
            size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        
        let bind_group = self.context.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: input_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: output_buffer.as_entire_binding() },
            ],
        });
        
        let mut encoder = self.context.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(((size + 255) / 256) as u32, 1, 1);
        }
        self.context.queue.submit(Some(encoder.finish()));
        
        let result = self.context.read_buffer(&output_buffer, size);
        Tensor::new(result, input.shape().clone()).unwrap()
    }
    
    /// Run a binary operation (2 inputs, 1 output)
    fn run_binary(&self, a: &Tensor, b: &Tensor, shader: &str, label: &str) -> TensorResult<Tensor> {
        if *a.shape() != *b.shape() {
            return Err(TensorError::ShapeMismatch {
                expected: a.shape().clone(),
                got: b.shape().clone(),
            });
        }
        
        let size = a.numel();
        let pipeline = self.context.get_or_create_pipeline(shader, label);
        
        let a_buffer = self.context.create_buffer(&a.data(), wgpu::BufferUsages::STORAGE);
        let b_buffer = self.context.create_buffer(&b.data(), wgpu::BufferUsages::STORAGE);
        let out_buffer = self.context.create_empty_buffer(
            size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        
        let bind_group = self.context.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: out_buffer.as_entire_binding() },
            ],
        });
        
        let mut encoder = self.context.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(((size + 255) / 256) as u32, 1, 1);
        }
        self.context.queue.submit(Some(encoder.finish()));
        
        let result = self.context.read_buffer(&out_buffer, size);
        Tensor::new(result, a.shape().clone())
    }
}

impl ComputeBackend for GpuBackend {
    fn name(&self) -> &str { "GPU" }
    
    fn info(&self) -> String {
        self.context.get_info()
    }
    
    fn is_available(&self) -> bool { true }
    
    // Element-wise operations
    fn add(&self, a: &Tensor, b: &Tensor) -> TensorResult<Tensor> { self.run_binary(a, b, shaders::ADD, "add") }
    fn sub(&self, a: &Tensor, b: &Tensor) -> TensorResult<Tensor> { self.run_binary(a, b, shaders::SUB, "sub") }
    fn mul(&self, a: &Tensor, b: &Tensor) -> TensorResult<Tensor> { self.run_binary(a, b, shaders::MUL, "mul") }
    fn div(&self, a: &Tensor, b: &Tensor) -> TensorResult<Tensor> { self.run_binary(a, b, shaders::DIV, "div") }
    
    // Activations
    fn relu(&self, x: &Tensor) -> Tensor { self.run_unary(x, shaders::RELU, "relu") }
    fn gelu(&self, x: &Tensor) -> Tensor { self.run_unary(x, shaders::GELU, "gelu") }
    fn sigmoid(&self, x: &Tensor) -> Tensor { self.run_unary(x, shaders::SIGMOID, "sigmoid") }
    fn silu(&self, x: &Tensor) -> Tensor { self.run_unary(x, shaders::SILU, "silu") }
    fn tanh(&self, x: &Tensor) -> Tensor { self.run_unary(x, shaders::TANH, "tanh") }
    
    // Math
    fn exp(&self, x: &Tensor) -> Tensor { self.run_unary(x, shaders::EXP, "exp") }
    fn log(&self, x: &Tensor) -> Tensor { self.run_unary(x, shaders::LOG, "log") }
    fn sqrt(&self, x: &Tensor) -> Tensor { self.run_unary(x, shaders::SQRT, "sqrt") }
    fn pow2(&self, x: &Tensor) -> Tensor { self.run_unary(x, shaders::POW2, "pow2") }
    fn neg(&self, x: &Tensor) -> Tensor { self.run_unary(x, shaders::NEG, "neg") }
    fn abs(&self, x: &Tensor) -> Tensor { self.run_unary(x, shaders::ABS, "abs") }
    
    // Matrix operations
    fn matmul(&self, a: &Tensor, b: &Tensor) -> TensorResult<Tensor> {
        if a.ndim() != 2 || b.ndim() != 2 {
            return Err(TensorError::DimensionError("Matris çarpımı 2D gerektirir".into()));
        }
        
        let (m, k1) = (a.shape()[0], a.shape()[1]);
        let (k2, n) = (b.shape()[0], b.shape()[1]);
        
        if k1 != k2 {
            return Err(TensorError::ShapeMismatch {
                expected: vec![m, k2],
                got: vec![k1, k2],
            });
        }
        
        let k = k1;
        let pipeline = self.context.get_or_create_pipeline(shaders::MATMUL, "matmul");
        
        // Create uniform buffer for params
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params { m: u32, n: u32, k: u32, _pad: u32 }
        
        let params = Params { m: m as u32, n: n as u32, k: k as u32, _pad: 0 };
        let params_buffer = self.context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        
        let a_buffer = self.context.create_buffer(&a.data(), wgpu::BufferUsages::STORAGE);
        let b_buffer = self.context.create_buffer(&b.data(), wgpu::BufferUsages::STORAGE);
        let c_buffer = self.context.create_empty_buffer(
            m * n,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        
        let bind_group = self.context.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: a_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: c_buffer.as_entire_binding() },
            ],
        });
        
        let mut encoder = self.context.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(
                ((n + 15) / 16) as u32,
                ((m + 15) / 16) as u32,
                1,
            );
        }
        self.context.queue.submit(Some(encoder.finish()));
        
        let result = self.context.read_buffer(&c_buffer, m * n);
        Tensor::new(result, vec![m, n])
    }
    
    fn transpose(&self, x: &Tensor) -> TensorResult<Tensor> {
        if x.ndim() != 2 {
            return Err(TensorError::DimensionError("Transpoz 2D gerektirir".into()));
        }
        
        let (rows, cols) = (x.shape()[0], x.shape()[1]);
        let pipeline = self.context.get_or_create_pipeline(shaders::TRANSPOSE, "transpose");
        
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params { rows: u32, cols: u32 }
        
        let params = Params { rows: rows as u32, cols: cols as u32 };
        let params_buffer = self.context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        
        let in_buffer = self.context.create_buffer(&x.data(), wgpu::BufferUsages::STORAGE);
        let out_buffer = self.context.create_empty_buffer(
            x.numel(),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        
        let bind_group = self.context.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: in_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: out_buffer.as_entire_binding() },
            ],
        });
        
        let mut encoder = self.context.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(((x.numel() + 255) / 256) as u32, 1, 1);
        }
        self.context.queue.submit(Some(encoder.finish()));
        
        let result = self.context.read_buffer(&out_buffer, x.numel());
        Tensor::new(result, vec![cols, rows])
    }
    
    // Normalization with GPU support
    fn softmax(&self, x: &Tensor) -> TensorResult<Tensor> {
        if x.ndim() != 2 {
            // Fall back to CPU for non-2D tensors
            return x.softmax();
        }
        
        let (rows, cols) = (x.shape()[0], x.shape()[1]);
        let pipeline = self.context.get_or_create_pipeline(shaders::SOFTMAX, "softmax");
        
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params { rows: u32, cols: u32 }
        
        let params = Params { rows: rows as u32, cols: cols as u32 };
        let params_buffer = self.context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        
        let in_buffer = self.context.create_buffer(&x.data(), wgpu::BufferUsages::STORAGE);
        let out_buffer = self.context.create_empty_buffer(
            x.numel(),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        
        let bind_group = self.context.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: in_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: out_buffer.as_entire_binding() },
            ],
        });
        
        let mut encoder = self.context.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One workgroup per row
            pass.dispatch_workgroups(((rows + 255) / 256) as u32, 1, 1);
        }
        self.context.queue.submit(Some(encoder.finish()));
        
        let result = self.context.read_buffer(&out_buffer, x.numel());
        Tensor::new(result, vec![rows, cols])
    }
    
    fn layer_norm(&self, x: &Tensor, gamma: &Tensor, beta: &Tensor, eps: f32) -> TensorResult<Tensor> {
        // TODO: Implement GPU layer norm using shaders::LAYERNORM
        x.layer_norm(gamma, beta, eps)
    }
    
    fn rms_norm(&self, x: &Tensor, weight: &Tensor, eps: f32) -> TensorResult<Tensor> {
        // TODO: Implement GPU RMS norm using shaders::RMSNORM
        x.rms_norm(weight, eps)
    }
    
    // Reductions (CPU for now - GPU reductions need parallel reduction algorithm)
    fn sum(&self, x: &Tensor) -> f32 { x.sum().item() }
    fn max(&self, x: &Tensor) -> f32 { x.max().item() }
    fn mean(&self, x: &Tensor) -> f32 { x.mean().item() }
}
