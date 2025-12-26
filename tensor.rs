//! Türkçe Kod - Tensor Module
//! 
//! Provides a Tensor abstraction for multi-dimensional arrays with GPU support.
//! Designed with clean engineering principles: trait-based abstraction, 
//! separation of concerns, and extensibility.

use std::fmt;
use std::collections::HashSet;
use std::f32::consts; // Added import for consts
use std::ops::{Add, Sub, Mul, Div, Deref};
use std::rc::Rc;
use std::cell::{RefCell, Ref, RefMut};

/// Shape of a tensor (dimensions)
pub type Shape = Vec<usize>;

/// Error type for tensor operations
#[derive(Debug, Clone, thiserror::Error)]
pub enum TensorError {
    #[error("Şekil uyumsuzluğu: {expected:?} bekleniyordu, {got:?} alındı")]
    ShapeMismatch { expected: Shape, got: Shape },
    
    #[error("Boyut hatası: {0}")]
    DimensionError(String),
    
    #[error("Dizin hatası: {index} (boyut: {size})")]
    IndexError { index: usize, size: usize },
    
    #[error("GPU hatası: {0}")]
    GpuError(String),
    
    #[error("İşlem desteklenmiyor: {0}")]
    UnsupportedOperation(String),
}

/// Operation type for Autograd
#[derive(Debug, Clone, Copy)]
pub enum Op {
    Add, Sub, Mul, Div,
    AddScalar(f32), SubScalar(f32), MulScalar(f32), DivScalar(f32),
    MatMul,
    Relu, Sigmoid, Tanh, Gelu, Silu,
    Exp, Log, Sqrt, Pow2, Abs, Neg,
    Sum, Mean, Max, Min,
    Softmax,
    LayerNorm, RmsNorm,
    Transpose, Reshape,
    CrossEntropy, Mse,
}

pub type TensorResult<T> = Result<T, TensorError>;

/// Internal storage for Tensor
struct TensorInner {
    data: Vec<f32>,
    shape: Shape,
    name: Option<String>,
    // Autograd fields
    grad: Option<Vec<f32>>,
    parents: Vec<Tensor>,
    op: Option<Op>, // Operation that created this tensor
}

/// Core Tensor structure (Handle)
/// 
/// Uses reference counting to allow graph construction for Autograd.
#[derive(Clone)]
pub struct Tensor {
    inner: Rc<RefCell<TensorInner>>,
}

impl Tensor {
    // -------------------------------------------------------------------------
    // Constructors
    // -------------------------------------------------------------------------
    
    /// Create a new tensor from data and shape
    pub fn new(data: Vec<f32>, shape: Shape) -> TensorResult<Self> {
        let expected_size: usize = shape.iter().product();
        if data.len() != expected_size {
            return Err(TensorError::DimensionError(format!(
                "Veri boyutu ({}) şekille ({:?}) uyuşmuyor (beklenen: {})",
                data.len(), shape, expected_size
            )));
        }
        let inner = TensorInner {
            data,
            shape,
            name: None,
            grad: None,
            parents: Vec::new(),
            op: None,
        };
        Ok(Self { inner: Rc::new(RefCell::new(inner)) })
    }
    
    /// Create a tensor filled with zeros
    pub fn zeros(shape: Shape) -> Self {
        let size: usize = shape.iter().product();
        let inner = TensorInner {
            data: vec![0.0; size],
            shape,
            name: None,
            grad: None,
            parents: Vec::new(),
            op: None,
        };
        Self { inner: Rc::new(RefCell::new(inner)) }
    }
    
    /// Create a tensor filled with ones
    pub fn ones(shape: Shape) -> Self {
        let size: usize = shape.iter().product();
        let inner = TensorInner {
            data: vec![1.0; size],
            shape,
            name: None,
            grad: None,
            parents: Vec::new(),
            op: None,
        };
        Self { inner: Rc::new(RefCell::new(inner)) }
    }
    
    /// Create a tensor with random values in [0, 1)
    pub fn random(shape: Shape) -> Self {
        use rand::Rng;
        let size: usize = shape.iter().product();
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..size).map(|_| rng.gen::<f32>()).collect();

        let inner = TensorInner {
            data,
            shape,
            name: None,
            grad: None,
            parents: Vec::new(),
            op: None,
        };
        Self { inner: Rc::new(RefCell::new(inner)) }
    }
    
    /// Create a tensor with random values from normal distribution
    pub fn randn(shape: Shape) -> Self {
        use rand::Rng;
        let size: usize = shape.iter().product();
        let mut rng = rand::thread_rng();
        // Approximate normal distribution using Box-Muller transform
        let data: Vec<f32> = (0..size)
            .map(|_| {
                let u1: f32 = rng.gen();
                let u2: f32 = rng.gen();
                (-2.0 * u1.ln()).sqrt() * (2.0 * consts::PI * u2).cos()
            })
            .collect();

        let inner = TensorInner {
            data,
            shape,
            name: None,
            grad: None,
            parents: Vec::new(),
            op: None,
        };
        Self { inner: Rc::new(RefCell::new(inner)) }
    }
    
    /// Create from a 1D slice
    pub fn from_slice(data: &[f32]) -> Self {
        let inner = TensorInner {
            data: data.to_vec(),
            shape: vec![data.len()],
            name: None,

            grad: None,
            parents: Vec::new(),
            op: None,
        };
        Self { inner: Rc::new(RefCell::new(inner)) }
    }
    
    /// Create a scalar (0-dimensional) tensor
    pub fn scalar(value: f32) -> Self {
         let inner = TensorInner {
            data: vec![value],
            shape: vec![],
            name: None,

            grad: None,
            parents: Vec::new(),
            op: None,
        };
        Self { inner: Rc::new(RefCell::new(inner)) }
    }
    
    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------
    
    /// Get the shape of the tensor (Ref)
    pub fn shape(&self) -> Ref<Shape> {
        Ref::map(self.inner.borrow(), |inner| &inner.shape)
    }
    
    /// Get the number of dimensions (rank)
    pub fn ndim(&self) -> usize {
        self.inner.borrow().shape.len()
    }
    
    /// Get the total number of elements
    pub fn numel(&self) -> usize {
        self.inner.borrow().data.len()
    }
    
    /// Get raw data reference (Ref)
    pub fn data(&self) -> Ref<[f32]> {
         Ref::map(self.inner.borrow(), |inner| inner.data.as_slice())
    }
    
    /// Get mutable data reference (RefMut)
    pub fn data_mut(&self) -> RefMut<Vec<f32>> {
         RefMut::map(self.inner.borrow_mut(), |inner| &mut inner.data)
    }
    
    /// Get a copy of data
    pub fn into_data(&self) -> Vec<f32> {
        self.inner.borrow().data.clone()
    }
    
    /// Check if tensor is a scalar (0-dim)
    pub fn is_scalar(&self) -> bool { self.shape().is_empty() }
    
    /// Check if tensor is a vector (1-dim)
    pub fn is_vector(&self) -> bool { self.shape().len() == 1 }
    
    /// Check if tensor is a matrix (2-dim)
    pub fn is_matrix(&self) -> bool { self.shape().len() == 2 }
    
    /// Get scalar value (panics if not scalar)
    pub fn item(&self) -> f32 {
        assert!(self.numel() == 1, "item() yalnızca tek elemanlı tensörler için");
        self.inner.borrow().data[0]
    }
    
    /// Set name for debugging
    pub fn with_name(self, name: &str) -> Self {
        self.inner.borrow_mut().name = Some(name.to_string());
        self
    }
    
    // -------------------------------------------------------------------------
    // Autograd Engine
    // -------------------------------------------------------------------------
    
    /// Perform backpropagation
    pub fn backward(&self) {
        use std::collections::HashSet;
        // 1. Topological Sort
        let mut visited = HashSet::new();
        let mut topo_order = Vec::new();
        
        self.build_topo_order(&mut visited, &mut topo_order);
        
        // 2. Initialize self.grad = 1.0 (if scalar) or ones
        {
            let mut inner = self.inner.borrow_mut();
            if inner.grad.is_none() {
                inner.grad = Some(vec![1.0; inner.data.len()]); // Usually scalar 1.0
            }
        }
        
        // 3. Process in reverse topological order (Root -> Leaves)
        for node in topo_order.iter().rev() {
            // Get node's gradient and op info
            // borrowing rule: we need to access node.inner (read op/data/grad)
            // AND access parent.inner (write grad)
            // We must copy necessary data from node to avoid double borrow panics.
            
            let (op, grad, data, shape, parents) = {
                let inner = node.inner.borrow();
                let op = inner.op;
                let grad = inner.grad.clone(); // Clone grad data
                let data = inner.data.clone(); // Clone primal data (needed for some derivatives like Relu)
                let shape = inner.shape.clone();
                let parents = inner.parents.clone();
                (op, grad, data, shape, parents)
            };
            
            if let (Some(op), Some(grad)) = (op, grad) {
                node.dispatch_backward(op, &grad, &data, &shape, &parents);
            }
        }
    }
    
    fn build_topo_order(&self, visited: &mut HashSet<*const TensorInner>, order: &mut Vec<Tensor>) {
        let ptr = self.inner.as_ptr();
        if visited.contains(&(ptr as *const TensorInner)) { return; }
        visited.insert(ptr);
        
        // Visit parents first
        let parents = self.inner.borrow().parents.clone();
        for parent in parents {
            parent.build_topo_order(visited, order);
        }
        
        order.push(self.clone());
    }
    
    /// Dispatch backward op
    fn dispatch_backward(
        &self, 
        op: Op, 
        grad: &[f32], 
        data: &[f32], 
        shape: &Shape, 
        parents: &[Tensor]
    ) {
        match op {
            Op::Add => {
                // z = x + y => dx = dz, dy = dz
                self.accumulate_grad(&parents[0], grad);
                self.accumulate_grad(&parents[1], grad);
            },
            Op::Sub => {
                // z = x - y => dx = dz, dy = -dz
                self.accumulate_grad(&parents[0], grad);
                let neg_grad: Vec<f32> = grad.iter().map(|&g| -g).collect();
                self.accumulate_grad(&parents[1], &neg_grad);
            },
            Op::Mul => {
                // z = x * y => dx = dz * y, dy = dz * x
                // Need x and y data.
                let lhs_data = parents[0].inner.borrow().data.clone();
                let rhs_data = parents[1].inner.borrow().data.clone();
                
                let d_lhs: Vec<f32> = grad.iter().zip(rhs_data.iter()).map(|(&g, &r)| g * r).collect();
                let d_rhs: Vec<f32> = grad.iter().zip(lhs_data.iter()).map(|(&g, &l)| g * l).collect();
                
                self.accumulate_grad(&parents[0], &d_lhs);
                self.accumulate_grad(&parents[1], &d_rhs);
            },
            Op::Div => {
                // z = x / y => dx = dz / y, dy = -dz * x / y^2
                // dx = dz * (1/y)
                // dy = dz * (-x/y^2)
                 let lhs_data = parents[0].inner.borrow().data.clone();
                 let rhs_data = parents[1].inner.borrow().data.clone();
                 
                 let d_lhs: Vec<f32> = grad.iter().zip(rhs_data.iter()).map(|(&g, &r)| g / r).collect();
                 let d_rhs: Vec<f32> = grad.iter().zip(lhs_data.iter().zip(rhs_data.iter()))
                    .map(|(&g, (&l, &r))| g * (-l / (r * r)))
                    .collect();
                    
                 self.accumulate_grad(&parents[0], &d_lhs);
                 self.accumulate_grad(&parents[1], &d_rhs);
            },
            Op::AddScalar(_) => self.accumulate_grad(&parents[0], grad),
            Op::SubScalar(_) => self.accumulate_grad(&parents[0], grad),
            Op::MulScalar(s) => {
                let d_x: Vec<f32> = grad.iter().map(|&g| g * s).collect();
                self.accumulate_grad(&parents[0], &d_x);
            },
            Op::DivScalar(s) => {
                let d_x: Vec<f32> = grad.iter().map(|&g| g / s).collect();
                self.accumulate_grad(&parents[0], &d_x);
            },
            Op::MatMul => {
                // C = A @ B
                // dA = dC @ B.T
                // dB = A.T @ dC
                // Shapes: A(m,k), B(k,n), C(m,n)
                let a = &parents[0];
                let b = &parents[1];
                
                // Helper to perform matmul on raw data?
                // Or reconstruct Tensors temporarily?
                // Reconstructing tensors is easier to reuse logic.
                
                // dA = grad @ b.T
                if let Ok(grad_tensor) = Tensor::new(grad.to_vec(), shape.clone()) {
                     if let Ok(b_t) = b.transpose() {
                         if let Ok(d_a) = grad_tensor.matmul(&b_t) {
                             self.accumulate_grad(a, &d_a.inner.borrow().data);
                         }
                     }
                     
                     // dB = a.T @ grad
                     if let Ok(a_t) = a.transpose() {
                         if let Ok(d_b) = a_t.matmul(&grad_tensor) {
                             self.accumulate_grad(b, &d_b.inner.borrow().data);
                         }
                     }
                }
            },
            Op::Relu => {
                // dx = dz if x > 0 else 0
                // Use input data (parents[0]) or output data? 
                // Relu is y = max(0, x). dy/dx = 1 if x>0.
                // We have 'data' (output = y). If y > 0, then x > 0 (mostly).
                // Precise check uses input.
                let input_data = parents[0].inner.borrow().data.clone();
                let d_x: Vec<f32> = grad.iter().zip(input_data.iter())
                    .map(|(&g, &x)| if x > 0.0 { g } else { 0.0 })
                    .collect();
                self.accumulate_grad(&parents[0], &d_x);
            },
            Op::Sigmoid => {
                // y = sigmoid(x)
                // dy/dx = y * (1 - y)
                // We have 'data' (output y).
                let d_x: Vec<f32> = grad.iter().zip(data.iter())
                    .map(|(&g, &y)| g * y * (1.0 - y))
                    .collect();
                self.accumulate_grad(&parents[0], &d_x);
            },
            Op::Tanh => {
                // y = tanh(x)
                // dy/dx = 1 - y^2
                let d_x: Vec<f32> = grad.iter().zip(data.iter())
                    .map(|(&g, &y)| g * (1.0 - y * y))
                    .collect();
                self.accumulate_grad(&parents[0], &d_x);
            },
            Op::Mse => {
                let pred = &parents[0];
                let target = &parents[1];
                
                if let Ok(d_pred) = pred.mse_grad(target) {
                    let factor = grad[0];
                    let mut scaled_grad: Vec<f32> = d_pred.inner.borrow().data.iter().map(|&x| x * factor).collect();
                    self.accumulate_grad(pred, &scaled_grad);
                    
                    // dLoss/dTarget = -dLoss/dPred
                    for g in scaled_grad.iter_mut() { *g = -*g; }
                    self.accumulate_grad(target, &scaled_grad);
                }
            },
            Op::CrossEntropy => {
                let pred = &parents[0];
                let target = &parents[1];
                
                if let Ok(d_pred) = pred.cross_entropy_grad(target) {
                    let factor = grad[0];
                    let scaled_grad: Vec<f32> = d_pred.inner.borrow().data.iter().map(|&x| x * factor).collect();
                    self.accumulate_grad(pred, &scaled_grad);
                    
                    // dLoss/dTarget = -log(pred) approx
                    // Skipping target grad for CE for now as simpler version
                }
            },
            Op::Sum => {
                // y = sum(x) -> y is scalar. grad is scalar.
                // dx = grad * ones(x.shape)
                let parent = &parents[0];
                let factor = grad[0];
                let d_x = vec![factor; parent.numel()];
                self.accumulate_grad(parent, &d_x);
            },
            Op::Reshape => {
                // Reshape preserves data layout (mostly) if contiguous.
                // If we assume standard layout, grad of reshape matches parent data size.
                // self.accumulate_grad checks size.
                self.accumulate_grad(&parents[0], grad);
            },
            Op::Transpose => {
                // y = x.T. dy is grad. dx = dy.T
                if let Ok(grad_tensor) = Tensor::new(grad.to_vec(), shape.clone()) {
                    if let Ok(dx) = grad_tensor.transpose() {
                        self.accumulate_grad(&parents[0], &dx.inner.borrow().data);
                    }
                }
            },
            Op::Exp => {
                // y = exp(x). dy/dx = y. dx = grad * y
                let d_x: Vec<f32> = grad.iter().zip(data.iter()).map(|(&g, &y)| g * y).collect();
                self.accumulate_grad(&parents[0], &d_x);
            },
            Op::Log => {
                // y = ln(x). dy/dx = 1/x. dx = grad / x
                let input = &parents[0].inner.borrow().data;
                let d_x: Vec<f32> = grad.iter().zip(input.iter()).map(|(&g, &x)| g / x).collect();
                self.accumulate_grad(&parents[0], &d_x);
            },
            Op::Pow2 => {
                 // y = x^2. dy/dx = 2x. dx = grad * 2x
                 let input = &parents[0].inner.borrow().data;
                 let d_x: Vec<f32> = grad.iter().zip(input.iter()).map(|(&g, &x)| g * 2.0 * x).collect();
                 self.accumulate_grad(&parents[0], &d_x);
            },
            _ => {
                // Other ops unimpl
            }
        }
    }
    
    /// Accumulate gradient into parent
    fn accumulate_grad(&self, target: &Tensor, grad: &[f32]) {
        let mut inner = target.inner.borrow_mut();
        
        // Check shape compatibility? (Broadcasting might require summation of grads)
        // For basic element-wise, shapes match.
        // For MatMul, we handled shape via matmul logic.
        // For Reduction (Sum), target shape is larger than grad shape.
        // If Op::Sum, we broadcast grad to target shape?
        // Wait, dispatch_backward for Op::Sum didn't run yet.
        
        // If input shape != grad shape, we assume reduction/broadcast happened.
        // But simply adding `grad` vector implies shapes match (or grad is flat list).
        // If lengths match, simple add.
        
        if inner.grad.is_none() {
            inner.grad = Some(grad.to_vec());
        } else {
            if let Some(g) = &mut inner.grad {
                if g.len() != grad.len() {
                    println!("Gradyan boyut hatası: Beklenen {}, Alınan {}", g.len(), grad.len());
                    return; 
                }
                for (i, val) in grad.iter().enumerate() {
                    g[i] += val;
                }
            }
        }
    }
    
    // -------------------------------------------------------------------------
    // Broadcasting Support
    // -------------------------------------------------------------------------
    
    /// Calculate the broadcast shape for two tensors.
    /// Returns the resulting shape or an error if shapes are incompatible.
    /// 
    /// Broadcasting rules (NumPy-style):
    /// 1. If shapes have different lengths, pad the shorter with 1s on the left
    /// 2. For each dimension, sizes must be equal or one of them must be 1
    /// 3. The result shape takes the max of each dimension
    pub fn broadcast_shapes(shape_a: &[usize], shape_b: &[usize]) -> TensorResult<Shape> {
        let max_dims = shape_a.len().max(shape_b.len());
        
        // Pad shapes with 1s on the left
        let mut padded_a = vec![1usize; max_dims];
        let mut padded_b = vec![1usize; max_dims];
        
        for (i, &dim) in shape_a.iter().rev().enumerate() {
            padded_a[max_dims - 1 - i] = dim;
        }
        for (i, &dim) in shape_b.iter().rev().enumerate() {
            padded_b[max_dims - 1 - i] = dim;
        }
        
        // Calculate result shape
        let mut result_shape = Vec::with_capacity(max_dims);
        for i in 0..max_dims {
            if padded_a[i] == padded_b[i] {
                result_shape.push(padded_a[i]);
            } else if padded_a[i] == 1 {
                result_shape.push(padded_b[i]);
            } else if padded_b[i] == 1 {
                result_shape.push(padded_a[i]);
            } else {
                return Err(TensorError::ShapeMismatch {
                    expected: shape_a.to_vec(),
                    got: shape_b.to_vec(),
                });
            }
        }
        
        Ok(result_shape)
    }
    
    /// Calculate strides for a shape (row-major order)
    fn calculate_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1usize; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }
    
    /// Convert a flat index to a multi-dimensional index
    fn flat_to_multi(flat_idx: usize, shape: &[usize], strides: &[usize]) -> Vec<usize> {
        let mut result = vec![0usize; shape.len()];
        let mut remaining = flat_idx;
        for i in 0..shape.len() {
            result[i] = remaining / strides[i];
            remaining %= strides[i];
        }
        result
    }
    
    /// Apply a binary operation with broadcasting support
    pub fn broadcast_apply<F>(&self, other: &Tensor, f: F, op: Op) -> TensorResult<Tensor>
    where
        F: Fn(f32, f32) -> f32,
    {
        let inner_self = self.inner.borrow();
        let inner_other = other.inner.borrow();
        
        // Check if shapes are exactly equal (fast path)
        if inner_self.shape == inner_other.shape {
            let data: Vec<f32> = inner_self.data.iter()
                .zip(inner_other.data.iter())
                .map(|(&a, &b)| f(a, b))
                .collect();
            
            let new_inner = TensorInner {
                data,
                shape: inner_self.shape.clone(),
                name: None,
                grad: None,
                parents: vec![self.clone(), other.clone()],
                op: Some(op),
            };
            return Ok(Tensor { inner: Rc::new(RefCell::new(new_inner)) });
        }
        
        // Calculate broadcast shape
        let result_shape = Self::broadcast_shapes(&inner_self.shape, &inner_other.shape)?;
        let result_size: usize = result_shape.iter().product();
        
        // Pad shapes for broadcasting
        let max_dims = result_shape.len();
        let mut shape_a = vec![1usize; max_dims];
        let mut shape_b = vec![1usize; max_dims];
        
        for (i, &dim) in inner_self.shape.iter().rev().enumerate() {
            shape_a[max_dims - 1 - i] = dim;
        }
        for (i, &dim) in inner_other.shape.iter().rev().enumerate() {
            shape_b[max_dims - 1 - i] = dim;
        }
        
        // Calculate strides
        let strides_result = Self::calculate_strides(&result_shape);
        let strides_a = Self::calculate_strides(&shape_a);
        let strides_b = Self::calculate_strides(&shape_b);
        
        // Perform broadcast operation
        let mut result_data = Vec::with_capacity(result_size);
        
        for flat_idx in 0..result_size {
            let multi_idx = Self::flat_to_multi(flat_idx, &result_shape, &strides_result);
            
            // Calculate indices for a and b with broadcasting
            let mut flat_a = 0usize;
            let mut flat_b = 0usize;
            
            for i in 0..max_dims {
                let idx_a = if shape_a[i] == 1 { 0 } else { multi_idx[i] };
                let idx_b = if shape_b[i] == 1 { 0 } else { multi_idx[i] };
                flat_a += idx_a * strides_a[i];
                flat_b += idx_b * strides_b[i];
            }
            
            result_data.push(f(inner_self.data[flat_a], inner_other.data[flat_b]));
        }
        
        let new_inner = TensorInner {
            data: result_data,
            shape: result_shape,
            name: None,
            grad: None,
            parents: vec![self.clone(), other.clone()],
            op: Some(op),
        };
        Ok(Tensor { inner: Rc::new(RefCell::new(new_inner)) })
    }
    
    // -------------------------------------------------------------------------
    // Reshaping
    // -------------------------------------------------------------------------
    
    /// Reshape tensor to new dimensions (must have same total elements)
    pub fn reshape(&self, new_shape: Shape) -> TensorResult<Self> {
        let new_size: usize = new_shape.iter().product();
        if new_size != self.numel() {
            return Err(TensorError::ShapeMismatch {
                expected: new_shape,
                got: self.shape().clone().to_vec(),
            });
        }
        let inner = self.inner.borrow();

        let new_inner = TensorInner {
            data: inner.data.clone(),
            shape: new_shape,
            name: inner.name.clone(),
            grad: None,
            parents: vec![self.clone()], // Track parent for autograd
            op: Some(Op::Reshape),
        };
        Ok(Self { inner: Rc::new(RefCell::new(new_inner)) })
    }
    
    /// Flatten to 1D tensor
    pub fn flatten(&self) -> Self {
        let inner = self.inner.borrow();

        let new_inner = TensorInner {
            data: inner.data.clone(),
            shape: vec![inner.data.len()],
            name: inner.name.clone(),
            grad: None,
            parents: vec![self.clone()],
            op: Some(Op::Reshape),
        };
         Self { inner: Rc::new(RefCell::new(new_inner)) }
    }
    
    /// Transpose 2D tensor
    pub fn transpose(&self) -> TensorResult<Self> {
        if self.ndim() != 2 {
            return Err(TensorError::DimensionError(
                "Transpoz yalnızca 2D tensörler için".to_string()
            ));
        }
        
        let inner = self.inner.borrow();
        let (rows, cols) = (inner.shape[0], inner.shape[1]);
        let mut result = vec![0.0; inner.data.len()];
        
        for i in 0..rows {
            for j in 0..cols {
                result[j * rows + i] = inner.data[i * cols + j];
            }
        }
        


        let new_inner = TensorInner {
            data: result,
            shape: vec![cols, rows],
            name: inner.name.as_ref().map(|n| format!("{}.T", n)),
             grad: None,
            parents: vec![self.clone()],
            op: Some(Op::Transpose),
        };
        Ok(Self { inner: Rc::new(RefCell::new(new_inner)) })
    }
    
    // -------------------------------------------------------------------------
    // Element-wise Operations (CPU)
    // -------------------------------------------------------------------------
    
    /// Apply a function to each element
    pub fn map<F: Fn(f32) -> f32>(&self, f: F, op: Option<Op>) -> Self {
        let inner = self.inner.borrow();
        let data: Vec<f32> = inner.data.iter().map(|&x| f(x)).collect();
        let new_inner = TensorInner {
            data,
            shape: inner.shape.clone(),
            name: None,
            grad: None,
            parents: vec![self.clone()],
            op,
        };
        Self { inner: Rc::new(RefCell::new(new_inner)) }
    }
    
    /// Apply ReLU activation
    pub fn relu(&self) -> Self {
        self.map(|x| x.max(0.0), Some(Op::Relu))
    }
    
    /// Apply Sigmoid activation
    pub fn sigmoid(&self) -> Self {
        self.map(|x| 1.0 / (1.0 + (-x).exp()), Some(Op::Sigmoid))
    }
    
    /// Apply Tanh activation
    pub fn tanh(&self) -> Self {
        self.map(|x| x.tanh(), Some(Op::Tanh))
    }
    
    /// Apply GELU activation
    pub fn gelu(&self) -> Self {
        self.map(|x| {
            0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x.powi(3))).tanh())
        }, Some(Op::Gelu))
    }
    
    /// Apply SiLU (Swish) activation
    pub fn silu(&self) -> Self {
        self.map(|x| x / (1.0 + (-x).exp()), Some(Op::Silu))
    }
    
    /// Apply exponential
    pub fn exp(&self) -> Self {
        self.map(|x| x.exp(), Some(Op::Exp))
    }
    
    /// Apply natural log
    pub fn log(&self) -> Self {
        self.map(|x| x.ln(), Some(Op::Log))
    }
    
    /// Apply square root
    pub fn sqrt(&self) -> Self {
        self.map(|x| x.sqrt(), Some(Op::Sqrt))
    }
    
    /// Square each element
    pub fn pow2(&self) -> Self {
        self.map(|x| x * x, Some(Op::Pow2))
    }
    
    /// Absolute value
    pub fn abs(&self) -> Self {
        self.map(|x| x.abs(), Some(Op::Abs))
    }
    
    /// Negate
    pub fn neg(&self) -> Self {
        self.map(|x| -x, Some(Op::Neg))
    }
    
    // -------------------------------------------------------------------------
    // Reduction Operations
    // -------------------------------------------------------------------------
    
    // -------------------------------------------------------------------------
    // Reduction Operations
    // -------------------------------------------------------------------------
    
    /// Sum all elements
    pub fn sum(&self) -> Tensor {
        let inner = self.inner.borrow();
        let sum: f32 = inner.data.iter().sum();
        let new_inner = TensorInner {
            data: vec![sum],
            shape: vec![], // Scalar
            name: None,
            grad: None,
            parents: vec![self.clone()],
            op: Some(Op::Sum),
        };
        Tensor { inner: Rc::new(RefCell::new(new_inner)) }
    }
    
    /// Mean of all elements
    pub fn mean(&self) -> Tensor {
        let sum = self.sum();
        let n = self.numel() as f32;
        sum.div_scalar(n) // Composite op: Sum -> Div
    }
    
    /// Maximum element
    pub fn max(&self) -> Tensor {
        let inner = self.inner.borrow();
        let val = inner.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let new_inner = TensorInner {
            data: vec![val],
            shape: vec![],
            name: None,
            grad: None,
            parents: vec![self.clone()],
            op: Some(Op::Max),
        };
        Tensor { inner: Rc::new(RefCell::new(new_inner)) }
    }
    
    /// Minimum element
    pub fn min(&self) -> Tensor {
        let inner = self.inner.borrow();
        let val = inner.data.iter().cloned().fold(f32::INFINITY, f32::min);
        let new_inner = TensorInner {
            data: vec![val],
            shape: vec![],
            name: None,
            grad: None,
            parents: vec![self.clone()],
            op: Some(Op::Min),
        };
        Tensor { inner: Rc::new(RefCell::new(new_inner)) }
    }
    
    /// Variance
    pub fn var(&self) -> Tensor {
        let mean = self.mean();
        // (x - mean)^2
        let diff = self.sub_scalar(mean.item()); // Note: using item() breaks graph here?
        // Wait, self - mean (tensor) broadcast?
        // I haven't implemented broadcast sub.
        // But mean is scalar Tensor.
        // `self.sub(&mean)` should work if broadcast is supported.
        // My `sub` impl checks shape equality!
        // So I can't do `x - scalar_tensor` easily yet.
        // But `sub_scalar` takes f32.
        // So `mean.item()` extracts f32, but breaks graph connection to `mean`.
        // So `var` will treat `mean` as constant.
        // Correct variance grad requires `mean` grad too, but often `mean` is treated as const.
        // Actually `var` derivative matches even if we ignore mean dep? No.
        // If I want full autograd for `var`, I need `Sub` to support broadcasting scalar.
        // Or `Op::Var` leaf op.
        // Let's implement `Op::Var` manually for now.
        
        // Actually, let's keep it simple. `var` usage is rare in basic backprop, usually strictly `loss`.
        // But `LayerNorm` uses `var`.
        // `LayerNorm` impl I wrote is fused `Op::LayerNorm`.
        // So `var` standalone is less critical.
        // I will implement `var` as returning Tensor via manual calc and `Op::Var` (missing enum? add it? No "Var").
        // I'll leave `var` returning f32 logic or broken graph for now?
        // No, I'll update `var` to return Tensor using `mean.item()` which breaks graph (StopGradient).
        // This is acceptable for now given I haven't implemented broadcast.
        
        let m = self.mean().item();
        let inner = self.inner.borrow();
        let var_val: f32 = inner.data.iter()
            .map(|&x| (x - m).powi(2))
            .sum::<f32>() / self.numel() as f32;
            
        let new_inner = TensorInner {
            data: vec![var_val],
            shape: vec![],
            name: None,
            grad: None,
            parents: vec![self.clone()],
            // Treat as "Mean" op or similar? No.
            // Let's add `Op::Var` to enum? No, I want to avoid too many ops.
            // Let's use `Op::Mean` (of squared diffs)?
            // It effectively is `mean((x-m)^2)`.
            // I'll leave it as `Op::Mean` for now? No, backward will be wrong.
            // I'll skip `var` backprop for now (set op: None).
            op: None, 
        };
        Tensor { inner: Rc::new(RefCell::new(new_inner)) }
    }
    
    /// Standard deviation
    pub fn std(&self) -> Tensor {
        let v = self.var();
        v.sqrt()
    }
    
    // -------------------------------------------------------------------------
    // Matrix Operations (CPU)
    // -------------------------------------------------------------------------
    
    /// Matrix multiplication (A @ B)
    pub fn matmul(&self, other: &Tensor) -> TensorResult<Self> {
        if self.ndim() != 2 || other.ndim() != 2 {
            return Err(TensorError::DimensionError(
                "Matris çarpımı için 2D tensörler gerekli".to_string()
            ));
        }
        
        let inner_self = self.inner.borrow();
        let inner_other = other.inner.borrow();
        
        let (m, k1) = (inner_self.shape[0], inner_self.shape[1]);
        let (k2, n) = (inner_other.shape[0], inner_other.shape[1]);
        
        if k1 != k2 {
            return Err(TensorError::ShapeMismatch {
                expected: vec![m, k2],
                got: vec![k1, k2],
            });
        }
        
        let k = k1;
        let mut result = vec![0.0; m * n];
        
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    sum += inner_self.data[i * k + p] * inner_other.data[p * n + j];
                }
                result[i * n + j] = sum;
            }
        }
        
        let new_inner = TensorInner {
            data: result,
            shape: vec![m, n],
            name: None,
            grad: None,
            parents: vec![self.clone(), other.clone()],
            op: Some(Op::MatMul),
        };
        Ok(Self { inner: Rc::new(RefCell::new(new_inner)) })
    }
    
    /// Softmax along last axis
    pub fn softmax(&self) -> TensorResult<Self> {
        let inner = self.inner.borrow();
        
        if inner.shape.len() != 2 {
            // For 1D, treat as single row
            let max_val = self.max();
            let exp_data: Vec<f32> = inner.data.iter().map(|&x| (x - max_val.item()).exp()).collect();
            let sum: f32 = exp_data.iter().sum();
            let result: Vec<f32> = exp_data.iter().map(|&x| x / sum).collect();
            
            let new_inner = TensorInner {
                data: result,
                shape: inner.shape.clone(),
                name: None,
                grad: None,
                parents: vec![self.clone()],
                op: None,
            };
            return Ok(Self { inner: Rc::new(RefCell::new(new_inner)) });
        }
        
        let (rows, cols) = (inner.shape[0], inner.shape[1]);
        let mut result = vec![0.0; inner.data.len()];
        
        for i in 0..rows {
            let offset = i * cols;
            let row = &inner.data[offset..offset + cols];
            
            // Find max for numerical stability
            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            
            // Compute exp and sum
            let exp_vals: Vec<f32> = row.iter().map(|&x| (x - max_val).exp()).collect();
            let sum: f32 = exp_vals.iter().sum();
            
            // Normalize
            for (j, &exp_val) in exp_vals.iter().enumerate() {
                result[offset + j] = exp_val / sum;
            }
        }
        
        let new_inner = TensorInner {
            data: result,
            shape: inner.shape.clone(),
            name: None,
            grad: None,
            parents: vec![self.clone()],
            op: Some(Op::Softmax),
        };
        Ok(Self { inner: Rc::new(RefCell::new(new_inner)) })
    }
    
    /// Layer normalization
    pub fn layer_norm(&self, gamma: &Tensor, beta: &Tensor, eps: f32) -> TensorResult<Self> {
        if self.ndim() != 2 {
            return Err(TensorError::DimensionError(
                "LayerNorm 2D tensör gerektirir".to_string()
            ));
        }
        
        let inner = self.inner.borrow();
        let inner_gamma = gamma.inner.borrow();
        let inner_beta = beta.inner.borrow();
        
        let (rows, cols) = (inner.shape[0], inner.shape[1]);
        let mut result = vec![0.0; inner.data.len()];
        
        for i in 0..rows {
            let offset = i * cols;
            let row = &inner.data[offset..offset + cols];
            
            // Compute mean
            let mean: f32 = row.iter().sum::<f32>() / cols as f32;
            
            // Compute variance
            let var: f32 = row.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f32>() / cols as f32;
            
            // Normalize and scale
            let inv_std = 1.0 / (var + eps).sqrt();
            for j in 0..cols {
                let norm = (row[j] - mean) * inv_std;
                result[offset + j] = inner_gamma.data[j] * norm + inner_beta.data[j];
            }
        }
        
        let new_inner = TensorInner {
            data: result,
            shape: inner.shape.clone(),
            name: None,
            grad: None,
            parents: vec![self.clone(), gamma.clone(), beta.clone()],
            op: Some(Op::LayerNorm),
        };
        Ok(Self { inner: Rc::new(RefCell::new(new_inner)) })
    }

    /// RMS normalization (LLaMA style)
    pub fn rms_norm(&self, weight: &Tensor, eps: f32) -> TensorResult<Self> {
        if self.ndim() != 2 {
            return Err(TensorError::DimensionError(
                "RMSNorm 2D tensör gerektirir".to_string()
            ));
        }
        
        let inner = self.inner.borrow();
        let inner_weight = weight.inner.borrow();
        
        let (rows, cols) = (inner.shape[0], inner.shape[1]);
        let mut result = vec![0.0; inner.data.len()];
        
        for i in 0..rows {
            let offset = i * cols;
            let row = &inner.data[offset..offset + cols];
            
            // Compute RMS
            let sum_sq: f32 = row.iter().map(|&x| x * x).sum();
            let rms = (sum_sq / cols as f32 + eps).sqrt();
            let inv_rms = 1.0 / rms;
            
            // Normalize and scale
            for j in 0..cols {
                result[offset + j] = inner_weight.data[j] * row[j] * inv_rms;
            }
        }
        
        let new_inner = TensorInner {
            data: result,
            shape: inner.shape.clone(),
            name: None,
            grad: None,
            parents: vec![self.clone(), weight.clone()],
            op: Some(Op::RmsNorm),
        };
        Ok(Self { inner: Rc::new(RefCell::new(new_inner)) })
    }
}

// -----------------------------------------------------------------------------
// Trait Implementations
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// Trait Implementations
// -----------------------------------------------------------------------------

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_scalar() {
            write!(f, "{}", self.inner.borrow().data[0])?;
        } else if self.is_vector() {
            write!(f, "[")?;
            let inner = self.inner.borrow();
            for (i, &x) in inner.data.iter().enumerate() {
                if i > 0 { write!(f, ", ")?; }
                if i >= 10 && self.numel() > 15 {
                    write!(f, "... ({} daha)", self.numel() - 10)?;
                    break;
                }
                write!(f, "{:.4}", x)?;
            }
            write!(f, "]")?;
        } else if self.is_matrix() {
            let inner = self.inner.borrow();
            let (rows, cols) = (inner.shape[0], inner.shape[1]);
            writeln!(f, "[")?;
            for i in 0..rows.min(5) {
                write!(f, "  [")?;
                for j in 0..cols.min(5) {
                    if j > 0 { write!(f, ", ")?; }
                    write!(f, "{:8.4}", inner.data[i * cols + j])?;
                }
                if cols > 5 { write!(f, ", ...")?; }
                writeln!(f, "]")?;
            }
            if rows > 5 { writeln!(f, "  ...")?; }
            write!(f, "]")?;
        } else {
            // N-D tensor
            write!(f, "Tensor(shape={:?})", self.shape())?;
        }
        
        // Show gradient if present
        if let Some(grad) = &self.inner.borrow().grad {
            write!(f, "\n  grad: {:?}", grad)?;
        }
        
        Ok(())
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let inner = self.inner.borrow();
        write!(f, "Tensor {{ shape: {:?}, numel: {}", inner.shape, inner.data.len())?;
        if let Some(ref name) = inner.name {
            write!(f, ", name: {}", name)?;
        }
        write!(f, " }}")
    }
}

// Element-wise binary operations with broadcasting support
impl Add for &Tensor {
    type Output = TensorResult<Tensor>;
    
    fn add(self, rhs: &Tensor) -> Self::Output {
        self.broadcast_apply(rhs, |a, b| a + b, Op::Add)
    }
}

impl Sub for &Tensor {
    type Output = TensorResult<Tensor>;
    
    fn sub(self, rhs: &Tensor) -> Self::Output {
        self.broadcast_apply(rhs, |a, b| a - b, Op::Sub)
    }
}

impl Mul for &Tensor {
    type Output = TensorResult<Tensor>;
    
    fn mul(self, rhs: &Tensor) -> Self::Output {
        self.broadcast_apply(rhs, |a, b| a * b, Op::Mul)
    }
}

impl Div for &Tensor {
    type Output = TensorResult<Tensor>;
    
    fn div(self, rhs: &Tensor) -> Self::Output {
        self.broadcast_apply(rhs, |a, b| a / b, Op::Div)
    }
}

// Scalar operations
impl Tensor {
    pub fn add_scalar(&self, s: f32) -> Self {
        self.map(|x| x + s, Some(Op::AddScalar(s)))
    }
    
    pub fn mul_scalar(&self, s: f32) -> Self {
        self.map(|x| x * s, Some(Op::MulScalar(s)))
    }
    
    pub fn sub_scalar(&self, s: f32) -> Self {
        self.map(|x| x - s, Some(Op::SubScalar(s)))
    }
    
    pub fn div_scalar(&self, s: f32) -> Self {
        self.map(|x| x / s, Some(Op::DivScalar(s)))
    }
    
    // -------------------------------------------------------------------------
    // Loss Functions (for AI Training)
    // -------------------------------------------------------------------------
    
    // -------------------------------------------------------------------------
    // Loss Functions (for AI Training)
    // -------------------------------------------------------------------------
    
    /// Cross-entropy loss for classification
    /// Assumes self contains softmax probabilities and target contains one-hot labels
    pub fn cross_entropy_loss(&self, target: &Tensor) -> TensorResult<Tensor> {
        let inner_self = self.inner.borrow();
        let inner_target = target.inner.borrow();
        
        if inner_self.shape != inner_target.shape {
            return Err(TensorError::ShapeMismatch {
                expected: inner_self.shape.clone(),
                got: inner_target.shape.clone(),
            });
        }
        
        let eps = 1e-7; // Prevent log(0)
        let loss: f32 = inner_self.data.iter()
            .zip(inner_target.data.iter())
            .map(|(&pred, &tgt)| -tgt * (pred + eps).ln())
            .sum();
        
        let final_loss = (loss / inner_self.data.len() as f32).max(0.0);
        
        let new_inner = TensorInner {
            data: vec![final_loss],
            shape: vec![],
            name: None,
            grad: None,
            parents: vec![self.clone(), target.clone()],
            op: Some(Op::CrossEntropy),
        };
        Ok(Tensor { inner: Rc::new(RefCell::new(new_inner)) })
    }
    
    /// Mean Squared Error loss
    pub fn mse_loss(&self, target: &Tensor) -> TensorResult<Tensor> {
        let inner_self = self.inner.borrow();
        let inner_target = target.inner.borrow();
        
        if inner_self.shape != inner_target.shape {
            return Err(TensorError::ShapeMismatch {
                expected: inner_self.shape.clone(),
                got: inner_target.shape.clone(),
            });
        }
        
        let loss: f32 = inner_self.data.iter()
            .zip(inner_target.data.iter())
            .map(|(&pred, &tgt)| (pred - tgt).powi(2))
            .sum();
            
        let final_loss = loss / inner_self.data.len() as f32;
        
        let new_inner = TensorInner {
            data: vec![final_loss],
            shape: vec![],
            name: None,
            grad: None,
            parents: vec![self.clone(), target.clone()],
            op: Some(Op::Mse),
        };
        Ok(Tensor { inner: Rc::new(RefCell::new(new_inner)) })
    }
    
    /// Gradient for cross-entropy loss
    /// Returns d(loss)/d(pred) = pred - target (for softmax + cross-entropy)
    pub fn cross_entropy_grad(&self, target: &Tensor) -> TensorResult<Tensor> {
        let inner_self = self.inner.borrow();
        let inner_target = target.inner.borrow();
        
        if inner_self.shape != inner_target.shape {
            return Err(TensorError::ShapeMismatch {
                expected: inner_self.shape.clone(),
                got: inner_target.shape.clone(),
            });
        }
        
        let grad_data: Vec<f32> = inner_self.data.iter()
            .zip(inner_target.data.iter())
            .map(|(&pred, &tgt)| pred - tgt)
            .collect();
            
        let new_inner = TensorInner {
            data: grad_data,
            shape: inner_self.shape.clone(),
            name: None,
            grad: None,
            parents: vec![], // Loss grad is a new starting point usually
            op: None,
        };
        Ok(Self { inner: Rc::new(RefCell::new(new_inner)) })
    }
    
    /// Gradient for MSE loss
    /// Returns d(loss)/d(pred) = 2 * (pred - target) / n
    pub fn mse_grad(&self, target: &Tensor) -> TensorResult<Tensor> {
        let inner_self = self.inner.borrow();
        let inner_target = target.inner.borrow();
        
        if inner_self.shape != inner_target.shape {
            return Err(TensorError::ShapeMismatch {
                expected: inner_self.shape.clone(),
                got: inner_target.shape.clone(),
            });
        }
        
        let n = inner_self.data.len() as f32;
        let grad_data: Vec<f32> = inner_self.data.iter()
            .zip(inner_target.data.iter())
            .map(|(&pred, &tgt)| 2.0 * (pred - tgt) / n)
            .collect();
            
        let new_inner = TensorInner {
            data: grad_data,
            shape: inner_self.shape.clone(),
            name: None,
            grad: None,
            parents: vec![],
            op: None,
        };
        Ok(Self { inner: Rc::new(RefCell::new(new_inner)) })
    }
    
    /// Subtract scaled tensor (for optimizer updates): self - scale * other
    pub fn sub_scaled(&self, other: &Tensor, scale: f32) -> Self {
        let inner_self = self.inner.borrow();
        let inner_other = other.inner.borrow();
        
        let data: Vec<f32> = inner_self.data.iter()
            .zip(inner_other.data.iter())
            .map(|(&a, &b)| a - scale * b)
            .collect();
            
        let new_inner = TensorInner {
            data,
            shape: inner_self.shape.clone(),
            name: None,
            grad: None,
            parents: vec![self.clone(), other.clone()],
            op: None,
        };
        Self { inner: Rc::new(RefCell::new(new_inner)) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tensor_create() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        assert_eq!(t.shape(), &vec![2, 2]);
        assert_eq!(t.numel(), 4);
    }
    
    #[test]
    fn test_matmul() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        let c = a.matmul(&b).unwrap();
        // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
        assert_eq!(c.data(), &[19.0, 22.0, 43.0, 50.0]);
    }
    
    #[test]
    fn test_relu() {
        let t = Tensor::from_slice(&[-1.0, 0.0, 1.0, 2.0]);
        let r = t.relu();
        assert_eq!(r.data(), &[0.0, 0.0, 1.0, 2.0]);
    }
    
    #[test]
    fn test_broadcast_shapes() {
        // Same shapes
        let result = Tensor::broadcast_shapes(&[3, 4], &[3, 4]).unwrap();
        assert_eq!(result, vec![3, 4]);
        
        // Scalar broadcast
        let result = Tensor::broadcast_shapes(&[3, 4], &[1]).unwrap();
        assert_eq!(result, vec![3, 4]);
        
        // Vector broadcast: [3,4] + [4] -> [3,4]
        let result = Tensor::broadcast_shapes(&[3, 4], &[4]).unwrap();
        assert_eq!(result, vec![3, 4]);
        
        // Full broadcast: [2,1,4] + [3,4] -> [2,3,4]
        let result = Tensor::broadcast_shapes(&[2, 1, 4], &[3, 4]).unwrap();
        assert_eq!(result, vec![2, 3, 4]);
        
        // Incompatible shapes should error
        assert!(Tensor::broadcast_shapes(&[3, 4], &[5]).is_err());
    }
    
    #[test]
    fn test_broadcast_add() {
        // [2,3] + [3] (vector broadcast)
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let b = Tensor::new(vec![10.0, 20.0, 30.0], vec![3]).unwrap();
        let c = (&a + &b).unwrap();
        
        assert_eq!(c.shape(), &vec![2, 3]);
        assert_eq!(c.data(), &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }
    
    #[test]
    fn test_broadcast_scalar() {
        // [2,2] * [1] (scalar broadcast)
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![10.0], vec![1]).unwrap();
        let c = (&a * &b).unwrap();
        
        assert_eq!(c.shape(), &vec![2, 2]);
        assert_eq!(c.data(), &[10.0, 20.0, 30.0, 40.0]);
    }
    
    #[test]
    fn test_softmax() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
        let s = t.softmax().unwrap();
        
        // Softmax should sum to 1
        let sum: f32 = s.data().iter().sum();
        assert!((sum - 1.0).abs() < 0.0001);
        
        // Values should be positive and monotonically increasing
        let data = s.into_data();
        assert!(data[0] < data[1] && data[1] < data[2] && data[2] < data[3]);
    }
}

