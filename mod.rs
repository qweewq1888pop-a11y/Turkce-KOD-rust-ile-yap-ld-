//! Backend Module
//! 
//! Provides an abstraction layer for compute backends (CPU, GPU).
//! Follows the Strategy pattern for clean backend switching.

pub mod cpu;
pub mod gpu;

use crate::tensor::{Tensor, TensorResult};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Execution mode for tensor operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    /// Always use CPU
    Cpu,
    /// Always use GPU (falls back to CPU if unavailable)
    Gpu,
    /// Automatic: GPU for large tensors, CPU for small
    Hybrid,
}

impl Default for ExecutionMode {
    fn default() -> Self { ExecutionMode::Hybrid }
}

impl std::fmt::Display for ExecutionMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecutionMode::Cpu => write!(f, "CPU"),
            ExecutionMode::Gpu => write!(f, "GPU"),
            ExecutionMode::Hybrid => write!(f, "Hybrid"),
        }
    }
}

/// Global execution mode (0=GPU, 1=CPU, 2=Hybrid)
static EXEC_MODE: AtomicUsize = AtomicUsize::new(2);

/// Threshold for hybrid mode (tensor size above this uses GPU)
static HYBRID_THRESHOLD: AtomicUsize = AtomicUsize::new(10_000);

/// Set the global execution mode
pub fn set_execution_mode(mode: ExecutionMode) {
    let val = match mode {
        ExecutionMode::Gpu => 0,
        ExecutionMode::Cpu => 1,
        ExecutionMode::Hybrid => 2,
    };
    EXEC_MODE.store(val, Ordering::SeqCst);
}

/// Get the current execution mode
pub fn get_execution_mode() -> ExecutionMode {
    match EXEC_MODE.load(Ordering::SeqCst) {
        0 => ExecutionMode::Gpu,
        1 => ExecutionMode::Cpu,
        _ => ExecutionMode::Hybrid,
    }
}

/// Set hybrid mode threshold
pub fn set_hybrid_threshold(threshold: usize) {
    HYBRID_THRESHOLD.store(threshold, Ordering::SeqCst);
}

/// Check if GPU should be used for a given tensor size
pub fn should_use_gpu(size: usize) -> bool {
    match get_execution_mode() {
        ExecutionMode::Gpu => true,
        ExecutionMode::Cpu => false,
        ExecutionMode::Hybrid => size >= HYBRID_THRESHOLD.load(Ordering::Relaxed),
    }
}

/// Compute backend trait - abstraction for CPU/GPU operations
/// 
/// This trait defines the interface that all compute backends must implement.
/// It follows the Strategy pattern, allowing runtime backend switching.
pub trait ComputeBackend: Send + Sync {
    /// Get backend name
    fn name(&self) -> &str;
    
    /// Get backend info (e.g., GPU model)
    fn info(&self) -> String;
    
    /// Check if backend is available
    fn is_available(&self) -> bool;
    
    // -------------------------------------------------------------------------
    // Element-wise Operations
    // -------------------------------------------------------------------------
    
    fn add(&self, a: &Tensor, b: &Tensor) -> TensorResult<Tensor>;
    fn sub(&self, a: &Tensor, b: &Tensor) -> TensorResult<Tensor>;
    fn mul(&self, a: &Tensor, b: &Tensor) -> TensorResult<Tensor>;
    fn div(&self, a: &Tensor, b: &Tensor) -> TensorResult<Tensor>;
    
    // -------------------------------------------------------------------------
    // Activation Functions
    // -------------------------------------------------------------------------
    
    fn relu(&self, x: &Tensor) -> Tensor;
    fn gelu(&self, x: &Tensor) -> Tensor;
    fn sigmoid(&self, x: &Tensor) -> Tensor;
    fn silu(&self, x: &Tensor) -> Tensor;
    fn tanh(&self, x: &Tensor) -> Tensor;
    
    // -------------------------------------------------------------------------
    // Math Operations
    // -------------------------------------------------------------------------
    
    fn exp(&self, x: &Tensor) -> Tensor;
    fn log(&self, x: &Tensor) -> Tensor;
    fn sqrt(&self, x: &Tensor) -> Tensor;
    fn pow2(&self, x: &Tensor) -> Tensor;
    fn neg(&self, x: &Tensor) -> Tensor;
    fn abs(&self, x: &Tensor) -> Tensor;
    
    // -------------------------------------------------------------------------
    // Matrix Operations
    // -------------------------------------------------------------------------
    
    fn matmul(&self, a: &Tensor, b: &Tensor) -> TensorResult<Tensor>;
    fn transpose(&self, x: &Tensor) -> TensorResult<Tensor>;
    
    // -------------------------------------------------------------------------
    // Normalization
    // -------------------------------------------------------------------------
    
    fn softmax(&self, x: &Tensor) -> TensorResult<Tensor>;
    fn layer_norm(&self, x: &Tensor, gamma: &Tensor, beta: &Tensor, eps: f32) -> TensorResult<Tensor>;
    fn rms_norm(&self, x: &Tensor, weight: &Tensor, eps: f32) -> TensorResult<Tensor>;
    
    // -------------------------------------------------------------------------
    // Reduction Operations
    // -------------------------------------------------------------------------
    
    fn sum(&self, x: &Tensor) -> f32;
    fn max(&self, x: &Tensor) -> f32;
    fn mean(&self, x: &Tensor) -> f32;
}

/// Get the best available backend based on execution mode and tensor size
pub fn get_backend(size: usize) -> Box<dyn ComputeBackend> {
    if should_use_gpu(size) {
        if let Some(gpu) = gpu::GpuBackend::try_new() {
            return Box::new(gpu);
        }
    }
    Box::new(cpu::CpuBackend::new())
}

/// Get GPU info string (or "N/A" if not available)
pub fn get_gpu_info() -> String {
    match gpu::GpuBackend::try_new() {
        Some(gpu) => gpu.info(),
        None => "GPU bulunamadı".to_string(),
    }
}
