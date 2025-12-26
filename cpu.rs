//! CPU Backend Implementation
//! 
//! Provides CPU-based tensor operations as a fallback or for small tensors.

use super::ComputeBackend;
use crate::tensor::{Tensor, TensorResult};

/// CPU compute backend
pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CpuBackend {
    fn default() -> Self { Self::new() }
}

impl ComputeBackend for CpuBackend {
    fn name(&self) -> &str { "CPU" }
    
    fn info(&self) -> String {
        "CPU Backend (Rust native)".to_string()
    }
    
    fn is_available(&self) -> bool { true }
    
    // -------------------------------------------------------------------------
    // Element-wise Operations
    // -------------------------------------------------------------------------
    
    fn add(&self, a: &Tensor, b: &Tensor) -> TensorResult<Tensor> {
        (a + b)
    }
    
    fn sub(&self, a: &Tensor, b: &Tensor) -> TensorResult<Tensor> {
        (a - b)
    }
    
    fn mul(&self, a: &Tensor, b: &Tensor) -> TensorResult<Tensor> {
        (a * b)
    }
    
    fn div(&self, a: &Tensor, b: &Tensor) -> TensorResult<Tensor> {
        (a / b)
    }
    
    // -------------------------------------------------------------------------
    // Activation Functions
    // -------------------------------------------------------------------------
    
    fn relu(&self, x: &Tensor) -> Tensor { x.relu() }
    fn gelu(&self, x: &Tensor) -> Tensor { x.gelu() }
    fn sigmoid(&self, x: &Tensor) -> Tensor { x.sigmoid() }
    fn silu(&self, x: &Tensor) -> Tensor { x.silu() }
    fn tanh(&self, x: &Tensor) -> Tensor { x.tanh() }
    
    // -------------------------------------------------------------------------
    // Math Operations
    // -------------------------------------------------------------------------
    
    fn exp(&self, x: &Tensor) -> Tensor { x.exp() }
    fn log(&self, x: &Tensor) -> Tensor { x.log() }
    fn sqrt(&self, x: &Tensor) -> Tensor { x.sqrt() }
    fn pow2(&self, x: &Tensor) -> Tensor { x.pow2() }
    fn neg(&self, x: &Tensor) -> Tensor { x.neg() }
    fn abs(&self, x: &Tensor) -> Tensor { x.abs() }
    
    // -------------------------------------------------------------------------
    // Matrix Operations
    // -------------------------------------------------------------------------
    
    fn matmul(&self, a: &Tensor, b: &Tensor) -> TensorResult<Tensor> {
        a.matmul(b)
    }
    
    fn transpose(&self, x: &Tensor) -> TensorResult<Tensor> {
        x.transpose()
    }
    
    // -------------------------------------------------------------------------
    // Normalization
    // -------------------------------------------------------------------------
    
    fn softmax(&self, x: &Tensor) -> TensorResult<Tensor> {
        x.softmax()
    }
    
    fn layer_norm(&self, x: &Tensor, gamma: &Tensor, beta: &Tensor, eps: f32) -> TensorResult<Tensor> {
        x.layer_norm(gamma, beta, eps)
    }
    
    fn rms_norm(&self, x: &Tensor, weight: &Tensor, eps: f32) -> TensorResult<Tensor> {
        x.rms_norm(weight, eps)
    }
    
    // -------------------------------------------------------------------------
    // Reduction Operations
    // -------------------------------------------------------------------------
    
    fn sum(&self, x: &Tensor) -> f32 { x.sum().item() }
    fn max(&self, x: &Tensor) -> f32 { x.max().item() }
    fn mean(&self, x: &Tensor) -> f32 { x.mean().item() }
}
