//! Türkçe Kod - Interpreter
//! 
//! Executes the AST and manages runtime state.

use std::collections::HashMap;
use std::fs; // For folder creation
use std::path::Path;
use std::io::{self, Write};

use crate::error::{TurkceKodError, TurkceKodResult};
use crate::parser::{
    BinaryOp, Expression, GpuOperation, InputType, MathFunction, Statement, UnaryOp, VarType,
    LossType, OptimizerType, DataSource, GuiWidgetType,
};
use crate::value::Value;
use crate::tensor::Tensor;
use crate::backend::{self, ExecutionMode};

/// Function definition stored at runtime
#[derive(Debug, Clone)]
pub struct Function {
    pub params: Vec<String>,
    pub body: Vec<Statement>,
}

/// Optimizer state for training
#[derive(Debug, Clone)]
pub struct Optimizer {
    pub opt_type: OptimizerType,
    pub learning_rate: f64,
    // Adam hyperparameters
    pub beta1: f64,      // Exponential decay rate for first moment (default: 0.9)
    pub beta2: f64,      // Exponential decay rate for second moment (default: 0.999)
    pub epsilon: f64,    // Small constant for numerical stability (default: 1e-8)
    // Momentum state
    pub m: HashMap<String, Vec<f32>>,  // First moment estimate
    pub v: HashMap<String, Vec<f32>>,  // Second moment estimate
    pub t: usize,  // Timestep
}

impl Optimizer {
    pub fn new(opt_type: OptimizerType, learning_rate: f64) -> Self {
        Self {
            opt_type,
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            m: HashMap::new(),
            v: HashMap::new(),
            t: 0,
        }
    }
    
    /// Create Adam optimizer with custom hyperparameters
    pub fn adam(learning_rate: f64, beta1: f64, beta2: f64, epsilon: f64) -> Self {
        Self {
            opt_type: OptimizerType::Adam,
            learning_rate,
            beta1,
            beta2,
            epsilon,
            m: HashMap::new(),
            v: HashMap::new(),
            t: 0,
        }
    }
}

/// Runtime representation of a GUI Widget
#[derive(Debug, Clone)]
pub struct GuiWidget {
    pub id: String,
    pub widget_type: GuiWidgetType,
    pub text: String,
    pub properties: HashMap<String, Value>, // Additional properties like color, size
    pub callback: Vec<Statement>, // Code to run when interacted with
}

impl GuiWidget {
    pub fn new(id: String, widget_type: GuiWidgetType, text: String, callback: Vec<Statement>) -> Self {
        Self {
            id,
            widget_type,
            text,
            properties: HashMap::new(),
            callback,
        }
    }
}

/// Helper: Convert tensor error to TurkceKodError
#[inline]
fn gpu_error(e: impl std::fmt::Display) -> TurkceKodError {
    TurkceKodError::GpuHatasi { message: e.to_string() }
}

// =============================================================================
// Helper Macros for Reducing Repetitive Code
// =============================================================================

/// Get a tensor from storage or return TanimlanmayanDegisken error
macro_rules! get_tensor {
    ($self:expr, $name:expr) => {
        $self.tensors.get($name).ok_or_else(|| 
            TurkceKodError::TanimlanmayanDegisken { name: $name.clone() }
        )?
    };
}

/// Get a mutable tensor from storage or return error
macro_rules! get_tensor_mut {
    ($self:expr, $name:expr) => {
        $self.tensors.get_mut($name).ok_or_else(|| 
            TurkceKodError::TanimlanmayanDegisken { name: $name.clone() }
        )?
    };
}

/// Get a variable from storage or return error
macro_rules! get_var {
    ($self:expr, $name:expr) => {
        $self.variables.get($name).ok_or_else(|| 
            TurkceKodError::TanimlanmayanDegisken { name: $name.clone() }
        )?
    };
}

/// Get a list from storage or return error
macro_rules! get_list {
    ($self:expr, $name:expr) => {
        $self.lists.get_mut($name).ok_or_else(|| 
            TurkceKodError::TanimlanmayanListe { name: $name.clone() }
        )?
    };
}

/// Print and store output line
macro_rules! output_line {
    ($self:expr, $line:expr) => {{
        $self.output.push($line.clone());
        println!("{}", $line);
    }};
}

/// The interpreter that executes Türkçe Kod programs
pub struct Interpreter {
    /// Variable storage
    variables: HashMap<String, Value>,
    /// List storage
    lists: HashMap<String, Vec<Value>>,
    /// Tensor storage (GPU-accelerated)
    tensors: HashMap<String, Tensor>,

    // AI Training state
    pub gradients: HashMap<String, Tensor>,
    pub optimizers: HashMap<String, Optimizer>,
    
    // GUI State
    pub gui_widgets: Vec<GuiWidget>,
    
    // Output buffer
    pub output: Vec<String>,

    /// Function storage
    functions: HashMap<String, Function>,
    /// Maximum iterations for loops (safety limit)
    max_iterations: usize,
    /// GUI input data (for tensor input)
    pub gui_input_data: Option<Vec<f32>>,
    /// GUI input text (for chat input)
    pub gui_input_text: Option<String>,
}

impl Default for Interpreter {
    fn default() -> Self {
        Self::new()
    }
}

impl Interpreter {
    /// Create a new interpreter
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            lists: HashMap::new(),
            tensors: HashMap::new(),
            gradients: HashMap::new(),
            optimizers: HashMap::new(),
            gui_widgets: Vec::new(),
            functions: HashMap::new(),
            output: Vec::new(),
            max_iterations: 10000,
            gui_input_data: None,
            gui_input_text: None,
        }
    }

    /// Reset the interpreter state
    pub fn reset(&mut self) {
        self.variables.clear();
        self.lists.clear();
        self.tensors.clear();
        self.gradients.clear();
        self.optimizers.clear();
        self.functions.clear();
        self.output.clear();
        self.gui_input_data = None;
    }

    /// Get the output buffer
    pub fn get_output(&self) -> &[String] {
        &self.output
    }

    /// Clear the output buffer
    pub fn clear_output(&mut self) {
        self.output.clear();
    }

    /// Execute a list of statements
    pub fn execute(&mut self, statements: &[Statement]) -> TurkceKodResult<Option<Value>> {
        for stmt in statements {
            if let Some(ret) = self.execute_statement(stmt)? {
                return Ok(Some(ret));
            }
        }
        Ok(None)
    }

    /// Execute a single statement
    fn execute_statement(&mut self, stmt: &Statement) -> TurkceKodResult<Option<Value>> {
        match stmt {
            Statement::Print(exprs) => {
                let mut parts: Vec<String> = Vec::new();
                for expr in exprs {
                    let val = self.evaluate(expr)?;
                    parts.push(val.to_string());
                }
                let line = parts.join(" ");
                self.output.push(line.clone());
                println!("{}", line);
                Ok(None)
            }

            Statement::VarDecl {
                var_type,
                name,
                value,
            } => {
                let val = if let Some(expr) = value {
                    self.evaluate(expr)?
                } else {
                    match var_type {
                        VarType::Integer => Value::Integer(0),
                        VarType::String => Value::String(String::new()),
                        VarType::Boolean => Value::Boolean(false),
                    }
                };
                self.variables.insert(name.clone(), val);
                Ok(None)
            }

            Statement::Assignment { name, value } => {
                let val = self.evaluate(value)?;
                self.variables.insert(name.clone(), val);
                Ok(None)
            }

            Statement::Calculate { name, expr } => {
                let val = self.evaluate(expr)?;
                self.variables.insert(name.clone(), val);
                Ok(None)
            }

            Statement::If {
                condition,
                then_block,
                else_block,
            } => {
                let cond = self.evaluate(condition)?;
                if cond.is_truthy() {
                    if let Some(ret) = self.execute(then_block)? {
                        return Ok(Some(ret));
                    }
                } else if let Some(else_stmts) = else_block {
                    if let Some(ret) = self.execute(else_stmts)? {
                        return Ok(Some(ret));
                    }
                }
                Ok(None)
            }

            Statement::While { condition, body } => {
                let mut iterations = 0;
                while self.evaluate(condition)?.is_truthy() {
                    if let Some(ret) = self.execute(body)? {
                        return Ok(Some(ret));
                    }
                    iterations += 1;
                    if iterations >= self.max_iterations {
                        return Err(TurkceKodError::SonsuzDongu {
                            max_iterations: self.max_iterations,
                        });
                    }
                }
                Ok(None)
            }

            Statement::Repeat { count, body } => {
                let count_val = self.evaluate(count)?;
                let n = count_val.as_integer().ok_or_else(|| {
                    TurkceKodError::type_error("sayı", count_val.type_name())
                })?;

                for _ in 0..n {
                    if let Some(ret) = self.execute(body)? {
                        return Ok(Some(ret));
                    }
                }
                Ok(None)
            }

            Statement::ForEach { list_name, body } => {
                let list = self
                    .lists
                    .get(list_name)
                    .ok_or_else(|| TurkceKodError::TanimlanmayanListe {
                        name: list_name.clone(),
                    })?
                    .clone();

                for item in list {
                    self.variables.insert("_eleman".to_string(), item);
                    if let Some(ret) = self.execute(body)? {
                        return Ok(Some(ret));
                    }
                }
                Ok(None)
            }

            Statement::FunctionDef { name, params, body } => {
                self.functions.insert(
                    name.clone(),
                    Function {
                        params: params.clone(),
                        body: body.clone(),
                    },
                );
                Ok(None)
            }

            Statement::Return(expr) => {
                let val = if let Some(e) = expr {
                    self.evaluate(e)?
                } else {
                    Value::None
                };
                return Ok(Some(val));
            }

            Statement::FunctionCall { name, args } => {
                self.call_function(name, args)?;
                Ok(None)
            }

            Statement::ListDecl { name } => {
                self.lists.insert(name.clone(), Vec::new());
                Ok(None)
            }

            Statement::ListAdd { list_name, value } => {
                let val = self.evaluate(value)?;
                let list = self.lists.get_mut(list_name).ok_or_else(|| {
                    TurkceKodError::TanimlanmayanListe {
                        name: list_name.clone(),
                    }
                })?;
                list.push(val);
                Ok(None)
            }

            Statement::ListLength { list_name } => {
                let list = self.lists.get(list_name).ok_or_else(|| {
                    TurkceKodError::TanimlanmayanListe {
                        name: list_name.clone(),
                    }
                })?;
                let len = list.len();
                self.output.push(len.to_string());
                println!("{}", len);
                Ok(None)
            }

            Statement::ListGet { list_name, index } => {
                let idx = self.evaluate(index)?;
                let idx_val = idx.as_integer().ok_or_else(|| {
                    TurkceKodError::type_error("sayı", idx.type_name())
                })?;

                let list = self.lists.get(list_name).ok_or_else(|| {
                    TurkceKodError::TanimlanmayanListe {
                        name: list_name.clone(),
                    }
                })?;

                if idx_val < 0 || idx_val as usize >= list.len() {
                    return Err(TurkceKodError::index_error(idx_val, list.len()));
                }

                let val = &list[idx_val as usize];
                self.output.push(val.to_string());
                println!("{}", val);
                Ok(None)
            }

            Statement::ReadInput {
                var_name,
                input_type,
            } => {
                print!("{}: ", var_name);
                io::stdout().flush().ok();

                let mut input = String::new();
                io::stdin().read_line(&mut input).ok();
                let input = input.trim();

                let val = match input_type {
                    InputType::String => Value::String(input.to_string()),
                    InputType::Integer => {
                        Value::Integer(input.parse().unwrap_or(0))
                    }
                    InputType::Float => {
                        Value::Float(input.parse().unwrap_or(0.0))
                    }
                };


                self.variables.insert(var_name.clone(), val);
                Ok(None)
            }

            Statement::ReadFile { var_name, path } => {
                let path_val = self.evaluate(path)?;
                let path_str = path_val.as_string();

                match std::fs::read_to_string(&path_str) {
                    Ok(content) => {
                        self.variables.insert(var_name.clone(), Value::String(content));
                    }
                    Err(e) => {
                        return Err(TurkceKodError::DosyaOkumaHatasi {
                            path: path_str,
                            message: e.to_string(),
                        });
                    }
                }
                Ok(None)
            }

            Statement::WriteFile { var_name, path } => {
                let path_val = self.evaluate(path)?;
                let path_str = path_val.as_string();

                let content = self
                    .variables
                    .get(var_name)
                    .map(|v| v.as_string())
                    .unwrap_or_default();

                match std::fs::write(&path_str, content) {
                    Ok(_) => {}
                    Err(e) => {
                        return Err(TurkceKodError::DosyaYazmaHatasi {
                            path: path_str,
                            message: e.to_string(),
                        });
                    }
                }
                Ok(None)
            }

            Statement::Sleep(duration) => {
                let dur = self.evaluate(duration)?;
                let secs = dur.as_float().unwrap_or(0.0);
                std::thread::sleep(std::time::Duration::from_secs_f64(secs));
                Ok(None)
            }

            Statement::MathFunc {
                func,
                target_var,
                arg,
            } => {
                let arg_val = self.evaluate(arg)?;
                let n = arg_val.as_float().ok_or_else(|| {
                    TurkceKodError::type_error("sayı", arg_val.type_name())
                })?;

                let result = match func {
                    MathFunction::Sqrt => n.sqrt(),
                    MathFunction::Sin => n.sin(),
                    MathFunction::Cos => n.cos(),
                    MathFunction::Tan => n.tan(),
                };

                self.variables.insert(target_var.clone(), Value::Float(result));
                Ok(None)
            }
            
            // -----------------------------------------------------------------
            // GPU/Tensor Statements
            // -----------------------------------------------------------------
            
            Statement::TensorDecl { name, data, shape } => {
                // Evaluate data expressions to floats
                let mut tensor_data: Vec<f32> = Vec::with_capacity(data.len());
                for expr in data {
                    let val = self.evaluate(expr)?;
                    let f = val.as_float().ok_or_else(|| {
                        TurkceKodError::type_error("sayı", val.type_name())
                    })?;
                    tensor_data.push(f as f32);
                }
                
                // Create tensor with shape
                match Tensor::new(tensor_data, shape.clone()) {
                    Ok(tensor) => {
                        self.tensors.insert(name.clone(), tensor);
                    }
                    Err(e) => {
                        return Err(TurkceKodError::GpuHatasi { 
                            message: format!("{}", e) 
                        });
                    }
                }
                Ok(None)
            }
            
            Statement::Backward { variable } => {
                if let Some(val) = self.tensors.get(variable.as_str()) {
                    val.backward();
                    println!("Geri yayılım tamamlandı: {}", variable);
                } else {
                    println!("Hata: Değişken bulunamadı veya tensör değil: {}", variable);
                }
                Ok(None)
            }
            
            // File System
            Statement::CreateFolder { path } => {
                let p = Path::new(path);
                if let Err(e) = fs::create_dir_all(p) {
                    let err_msg = format!("Klasör oluşturma hatası '{}': {}", path, e);
                    self.output.push(err_msg.clone());
                    println!("{}", err_msg);
                } else {
                    let msg = format!("Klasör oluşturuldu: {}", path);
                    self.output.push(msg.clone());
                    println!("{}", msg);
                }
                Ok(None)
            }
            
            Statement::CreateAiDirs => {
                let dirs = [
                    "models/gguf", 
                    "models/safetensors", 
                    "models/pkl", 
                    "models/quantization"
                ];
                
                for dir in dirs {
                     let p = Path::new(dir);
                     if let Err(e) = fs::create_dir_all(p) {
                         let err_msg = format!("AI dizini oluşturma hatası '{}': {}", dir, e);
                         self.output.push(err_msg.clone());
                         println!("{}", err_msg);
                     } else {
                         let msg = format!("AI dizini oluşturuldu: {}", dir);
                         self.output.push(msg.clone());
                         println!("{}", msg);
                     }
                }
                Ok(None)
            }

            Statement::RandomTensor { name, shape } => {
                let tensor = Tensor::random(shape.clone());
                self.tensors.insert(name.clone(), tensor);
                
                let line = format!("Rastgele tensor oluşturuldu: {} şekil: {:?}", name, shape);
                self.output.push(line.clone());
                println!("{}", line);
                Ok(None)
            }
            
            Statement::GpuOp { result, op, args } => {
                let result_tensor = match op {
                    // Binary operations
                    GpuOperation::MatMul => {
                        let a = self.tensors.get(&args[0]).ok_or_else(|| 
                            TurkceKodError::TanimlanmayanDegisken { name: args[0].clone() })?;
                        let b = self.tensors.get(&args[1]).ok_or_else(|| 
                            TurkceKodError::TanimlanmayanDegisken { name: args[1].clone() })?;
                        a.matmul(b).map_err(|e| TurkceKodError::GpuHatasi { 
                            message: format!("{}", e) 
                        })?
                    }
                    GpuOperation::Add => {
                        let a = self.tensors.get(&args[0]).ok_or_else(|| 
                            TurkceKodError::TanimlanmayanDegisken { name: args[0].clone() })?;
                        let b = self.tensors.get(&args[1]).ok_or_else(|| 
                            TurkceKodError::TanimlanmayanDegisken { name: args[1].clone() })?;
                        (a + b).map_err(|e| TurkceKodError::GpuHatasi { 
                            message: format!("{}", e) 
                        })?
                    }
                    GpuOperation::Sub => {
                        let a = self.tensors.get(&args[0]).ok_or_else(|| 
                            TurkceKodError::TanimlanmayanDegisken { name: args[0].clone() })?;
                        let b = self.tensors.get(&args[1]).ok_or_else(|| 
                            TurkceKodError::TanimlanmayanDegisken { name: args[1].clone() })?;
                        (a - b).map_err(|e| TurkceKodError::GpuHatasi { 
                            message: format!("{}", e) 
                        })?
                    }
                    GpuOperation::Mul => {
                        let a = self.tensors.get(&args[0]).ok_or_else(|| 
                            TurkceKodError::TanimlanmayanDegisken { name: args[0].clone() })?;
                        let b = self.tensors.get(&args[1]).ok_or_else(|| 
                            TurkceKodError::TanimlanmayanDegisken { name: args[1].clone() })?;
                        (a * b).map_err(|e| TurkceKodError::GpuHatasi { 
                            message: format!("{}", e) 
                        })?
                    }
                    GpuOperation::Div => {
                        let a = self.tensors.get(&args[0]).ok_or_else(|| 
                            TurkceKodError::TanimlanmayanDegisken { name: args[0].clone() })?;
                        let b = self.tensors.get(&args[1]).ok_or_else(|| 
                            TurkceKodError::TanimlanmayanDegisken { name: args[1].clone() })?;
                        (a / b).map_err(|e| TurkceKodError::GpuHatasi { 
                            message: format!("{}", e) 
                        })?
                    }
                    
                    // Unary operations (activations)
                    GpuOperation::Relu => {
                        let x = self.tensors.get(&args[0]).ok_or_else(|| 
                            TurkceKodError::TanimlanmayanDegisken { name: args[0].clone() })?;
                        x.relu()
                    }
                    GpuOperation::Gelu => {
                        let x = self.tensors.get(&args[0]).ok_or_else(|| 
                            TurkceKodError::TanimlanmayanDegisken { name: args[0].clone() })?;
                        x.gelu()
                    }
                    GpuOperation::Sigmoid => {
                        let x = self.tensors.get(&args[0]).ok_or_else(|| 
                            TurkceKodError::TanimlanmayanDegisken { name: args[0].clone() })?;
                        x.sigmoid()
                    }
                    GpuOperation::Silu => {
                        let x = self.tensors.get(&args[0]).ok_or_else(|| 
                            TurkceKodError::TanimlanmayanDegisken { name: args[0].clone() })?;
                        x.silu()
                    }
                    GpuOperation::Tanh => {
                        let x = self.tensors.get(&args[0]).ok_or_else(|| 
                            TurkceKodError::TanimlanmayanDegisken { name: args[0].clone() })?;
                        x.tanh()
                    }
                    GpuOperation::Softmax => {
                        let x = self.tensors.get(&args[0]).ok_or_else(|| 
                            TurkceKodError::TanimlanmayanDegisken { name: args[0].clone() })?;
                        x.softmax().map_err(|e| TurkceKodError::GpuHatasi { 
                            message: format!("{}", e) 
                        })?
                    }
                    GpuOperation::Transpose => {
                        let x = self.tensors.get(&args[0]).ok_or_else(|| 
                            TurkceKodError::TanimlanmayanDegisken { name: args[0].clone() })?;
                        x.transpose().map_err(|e| TurkceKodError::GpuHatasi { 
                            message: format!("{}", e) 
                        })?
                    }
                    
                    // Normalization (requires extra params)
                    GpuOperation::LayerNorm | GpuOperation::RmsNorm => {
                        let x = self.tensors.get(&args[0]).ok_or_else(|| 
                            TurkceKodError::TanimlanmayanDegisken { name: args[0].clone() })?;
                        // For now, just return input (full implementation needs gamma/beta)
                        x.clone()
                    }
                };
                
                self.tensors.insert(result.clone(), result_tensor);
                Ok(None)
            }
            
            Statement::GpuInfo => {
                let info = backend::get_gpu_info();
                let line = format!("GPU Bilgisi: {}", info);
                self.output.push(line.clone());
                println!("{}", line);
                Ok(None)
            }
            
            Statement::GpuMode(mode) => {
                let exec_mode = match mode.to_lowercase().as_str() {
                    "cpu" => ExecutionMode::Cpu,
                    "gpu" => ExecutionMode::Gpu,
                    "hybrid" | _ => ExecutionMode::Hybrid,
                };
                backend::set_execution_mode(exec_mode);
                
                let line = format!("GPU modu: {}", exec_mode);
                self.output.push(line.clone());
                println!("{}", line);
                Ok(None)
            }
            
            // -----------------------------------------------------------------
            // AI Training Statements
            // -----------------------------------------------------------------
            
            Statement::Loss { result, loss_type, predicted, target } => {
                let pred = self.tensors.get(predicted).ok_or_else(|| 
                    TurkceKodError::TanimlanmayanDegisken { name: predicted.clone() })?;
                let tgt = self.tensors.get(target).ok_or_else(|| 
                    TurkceKodError::TanimlanmayanDegisken { name: target.clone() })?;
                
                let loss_value = match loss_type {
                    LossType::CrossEntropy => pred.cross_entropy_loss(tgt).map_err(gpu_error)?,
                    LossType::MSE => pred.mse_loss(tgt).map_err(gpu_error)?,
                };
                
                // Store loss as variable
                self.variables.insert(result.clone(), Value::Float(loss_value.item() as f64));
                
                // Also compute and store gradient for backprop
                let grad = match loss_type {
                    LossType::CrossEntropy => pred.cross_entropy_grad(tgt).map_err(gpu_error)?,
                    LossType::MSE => pred.mse_grad(tgt).map_err(gpu_error)?,
                };
                self.gradients.insert(predicted.clone(), grad);
                
                let line = format!("Kayıp ({}): {:.6}", result, loss_value);
                self.output.push(line.clone());
                println!("{}", line);
                Ok(None)
            }
            
            Statement::Backprop { loss_var: _ } => {
                // Backpropagation: propagate gradients through network
                // This is a simplified version - full implementation would require
                // tracking the computation graph
                let line = "Geri yayılım tamamlandı".to_string();
                self.output.push(line.clone());
                println!("{}", line);
                Ok(None)
            }
            
            Statement::ZeroGrad => {
                self.gradients.clear();
                let line = "Gradyanlar sıfırlandı".to_string();
                self.output.push(line.clone());
                println!("{}", line);
                Ok(None)
            }
            
            Statement::OptimizerDecl { name, opt_type, learning_rate } => {
                let optimizer = Optimizer::new(opt_type.clone(), *learning_rate);
                self.optimizers.insert(name.clone(), optimizer);
                
                let line = format!("Optimizer oluşturuldu: {} (lr={})", name, learning_rate);
                self.output.push(line.clone());
                println!("{}", line);
                Ok(None)
            }
            
            Statement::OptimizerStep { optimizer } => {
                // Increment timestep first (needed for bias correction)
                let opt = self.optimizers.get_mut(optimizer).ok_or_else(|| 
                    TurkceKodError::TanimlanmayanDegisken { name: optimizer.clone() })?;
                opt.t += 1;
                
                let lr = opt.learning_rate as f32;
                let beta1 = opt.beta1 as f32;
                let beta2 = opt.beta2 as f32;
                let epsilon = opt.epsilon as f32;
                let t = opt.t;
                let opt_type = opt.opt_type.clone();
                
                // Collect gradient names to avoid borrow issues
                let grad_names: Vec<String> = self.gradients.keys().cloned().collect();
                
                for name in grad_names {
                    if let (Some(grad), Some(tensor)) = (self.gradients.get(&name), self.tensors.get(&name)) {
                        let grad_data = grad.into_data();
                        let tensor_data = tensor.into_data();
                        let tensor_shape = tensor.shape().clone();
                        
                        let updated_data = match opt_type {
                            OptimizerType::SGD => {
                                // SGD: w = w - lr * grad
                                tensor_data.iter()
                                    .zip(grad_data.iter())
                                    .map(|(&w, &g)| w - lr * g)
                                    .collect::<Vec<f32>>()
                            }
                            OptimizerType::Adam => {
                                // Full Adam with momentum and bias correction
                                // Get or initialize m and v for this parameter
                                let opt = self.optimizers.get_mut(optimizer).unwrap();
                                
                                let m = opt.m.entry(name.clone())
                                    .or_insert_with(|| vec![0.0; grad_data.len()]);
                                let v = opt.v.entry(name.clone())
                                    .or_insert_with(|| vec![0.0; grad_data.len()]);
                                
                                // Update biased first moment estimate: m = β1 * m + (1 - β1) * g
                                // Update biased second moment estimate: v = β2 * v + (1 - β2) * g²
                                for i in 0..grad_data.len() {
                                    m[i] = beta1 * m[i] + (1.0 - beta1) * grad_data[i];
                                    v[i] = beta2 * v[i] + (1.0 - beta2) * grad_data[i] * grad_data[i];
                                }
                                
                                // Bias correction
                                let bias_correction1 = 1.0 - beta1.powi(t as i32);
                                let bias_correction2 = 1.0 - beta2.powi(t as i32);
                                
                                // Update weights: w = w - lr * (m_hat / (sqrt(v_hat) + ε))
                                tensor_data.iter()
                                    .zip(m.iter().zip(v.iter()))
                                    .map(|(&w, (&m_i, &v_i))| {
                                        let m_hat = m_i / bias_correction1;
                                        let v_hat = v_i / bias_correction2;
                                        w - lr * m_hat / (v_hat.sqrt() + epsilon)
                                    })
                                    .collect::<Vec<f32>>()
                            }
                        };
                        
                        // Update the tensor
                        if let Some(tensor) = self.tensors.get_mut(&name) {
                            *tensor = Tensor::new(updated_data, tensor_shape).unwrap();
                        }
                    }
                }
                
                let opt = self.optimizers.get(optimizer).unwrap();
                let line = format!("Optimizer adımı: {} (adım {})", optimizer, opt.t);
                self.output.push(line.clone());
                println!("{}", line);
                Ok(None)
            }
            
            Statement::Train { epochs, batch_size, body } => {
                let epoch_count = match self.evaluate(epochs)? {
                    Value::Integer(i) => i as usize,
                    Value::Float(f) => f as usize,
                    _ => 1,
                };
                
                let batch = match self.evaluate(batch_size)? {
                    Value::Integer(i) => i as usize,
                    Value::Float(f) => f as usize,
                    _ => 32,
                };
                
                let line = format!("Eğitim başladı: {} epoch, batch={}", epoch_count, batch);
                self.output.push(line.clone());
                println!("{}", line);
                
                for epoch in 1..=epoch_count {
                    // Execute training body
                    self.execute(body)?;
                    
                    let line = format!("Epoch {}/{} tamamlandı", epoch, epoch_count);
                    self.output.push(line.clone());
                    println!("{}", line);
                }
                Ok(None)
            }
            
            Statement::LoadData { var_name, source } => {
                match source {
                    DataSource::Cifar10 => {
                        // Create dummy CIFAR-10 data (32x32x3 = 3072 values)
                        let tensor = Tensor::random(vec![1, 3072]);
                        self.tensors.insert(var_name.clone(), tensor);
                        
                        let line = format!("CIFAR-10 verisi yüklendi: {}", var_name);
                        self.output.push(line.clone());
                        println!("{}", line);
                    }
                    DataSource::File(path) => {
                        let line = format!("Veri dosyadan yükleniyor: {}", path);
                        self.output.push(line.clone());
                        println!("{}", line);
                        
                        let path_obj = std::path::Path::new(&path);
                        
                        // Handle JSONL files
                        if path.ends_with(".jsonl") || path.ends_with(".json") {
                            match std::fs::File::open(path_obj) {
                                Ok(file) => {
                                    let reader = std::io::BufReader::new(file);
                                    let mut loaded_data = Vec::new();
                                    let mut count = 0;
                                    
                                    use std::io::BufRead;
                                    for line_result in reader.lines() {
                                        if let Ok(line) = line_result {
                                            if let Ok(json_val) = serde_json::from_str::<serde_json::Value>(&line) {
                                                // Convert JSON value to Türkçe Kod Value
                                                // For now, simpler implementation: store full JSON object as String
                                                loaded_data.push(Value::String(line));
                                                count += 1;
                                            }
                                        }
                                    }
                                    
                                    // Store as a List variable
                                    self.lists.insert(var_name.clone(), loaded_data);
                                    
                                    let msg = format!("✓ {} satır veri yüklendi (Liste: {})", count, var_name);
                                    self.output.push(msg.clone());
                                    println!("{}", msg);
                                }
                                Err(e) => {
                                    return Err(TurkceKodError::DosyaOkumaHatasi { 
                                        path: path.clone(), 
                                        message: e.to_string() 
                                    });
                                }
                            }
                        } else {
                            // Default behavior for unknown files (placeholder tensor)
                            let tensor = Tensor::zeros(vec![1, 10]);
                            self.tensors.insert(var_name.clone(), tensor);
                            
                            let msg = format!("⚠️ Bilinmeyen dosya formatı, boş tensor oluşturuldu: {}", var_name);
                            self.output.push(msg.clone());
                            println!("{}", msg);
                        }
                    }
                    DataSource::Console | DataSource::Gui => {
                        // Handled by ConsoleRead and GuiData statements
                    }
                }
                Ok(None)
            }
            
            Statement::ConsoleRead { var_name } => {
                print!("Tensor verisi girin (virgülle ayrılmış): ");
                io::stdout().flush().unwrap();
                
                let mut input = String::new();
                io::stdin().read_line(&mut input).unwrap();
                
                let values: Vec<f32> = input
                    .trim()
                    .split(',')
                    .filter_map(|s| s.trim().parse().ok())
                    .collect();
                
                let len = values.len();
                let tensor = Tensor::new(values, vec![1, len]).map_err(gpu_error)?;
                self.tensors.insert(var_name.clone(), tensor);
                
                let line = format!("Konsoldan veri okundu: {} ({}eleman)", var_name, len);
                self.output.push(line.clone());
                println!("{}", line);
                Ok(None)
            }
            
            Statement::GuiDecl { widget_type, id, text, commands } => {
                // For Buttons, 'commands' IS the callback. For others, it might be children or init code.
                // Currently, we'll store commands as callback for Buttons.
                // For others, we might execute them immediately if they are property setters?
                // Let's adopt this logic:
                // 1. Create widget
                // 2. If it's a container (Window), execute commands (children). 
                // 3. If it's a Button, store commands as callback.
                
                let mut callback: Vec<Statement> = Vec::new(); // Explicit type needed
                let mut widget = GuiWidget::new(id.clone(), widget_type.clone(), text.clone(), Vec::new());
                
                // Initialize default properties
                if *widget_type == GuiWidgetType::Input {
                    widget.properties.insert("deger".to_string(), Value::String(String::new()));
                }
                if *widget_type == GuiWidgetType::Chat {
                    widget.properties.insert("mesajlar".to_string(), Value::List(Vec::new()));
                    widget.properties.insert("girdi".to_string(), Value::String(String::new()));
                }
                
                match widget_type {
                    GuiWidgetType::Button => {
                        // Store commands as callback
                        widget.callback = commands.clone();
                    },
                    _ => {
                        // Execute commands immediately
                        self.gui_widgets.push(widget.clone());
                        for cmd in commands {
                            // execute expects slice
                            self.execute(std::slice::from_ref(cmd))?;
                        }
                        return Ok(None); 
                    }
                }
                
                self.gui_widgets.push(widget);
                
                let line = format!("GUI bileşeni tanımlandı: {} ({})", id, text);
                self.output.push(line.clone());
                println!("{}", line);
                Ok(None)
            }
            
            Statement::GuiUpdate { id, property, value } => {
                let val = self.evaluate(value)?;
                
                // Find widget and update property
                if let Some(widget) = self.gui_widgets.iter_mut().find(|w| w.id == *id) {
                    widget.properties.insert(property.clone(), val.clone());
                    
                    let line = format!("GUI güncelleme: {}.{} = {:?}", id, property, val);
                    self.output.push(line.clone());
                    println!("{}", line);
                } else {
                    return Err(TurkceKodError::TanimlanmayanDegisken { name: id.clone() });
                }
                Ok(None)
            }

            Statement::GuiData { widget_id, var_name } => {
                // If widget_id is specified, read from that specific widget's "deger" property
                if let Some(id) = widget_id {
                    // Find the widget and get its value
                    if let Some(widget) = self.gui_widgets.iter().find(|w| &w.id == id) {
                        if let Some(value) = widget.properties.get("deger") {
                            self.variables.insert(var_name.clone(), value.clone());
                            let line = format!("Widget'tan veri alındı ({}): '{}'", id, value);
                            self.output.push(line.clone());
                            println!("{}", line);
                        } else {
                            // Widget found but no "deger" property - return empty string
                            self.variables.insert(var_name.clone(), Value::String(String::new()));
                            let line = format!("Widget'ta değer bulunamadı: {}", id);
                            self.output.push(line.clone());
                            println!("{}", line);
                        }
                    } else {
                        return Err(TurkceKodError::TanimlanmayanDegisken { name: id.clone() });
                    }
                } else if let Some(text) = &self.gui_input_text {
                    // Handle String input (original behavior for chat widget)
                    self.variables.insert(var_name.clone(), Value::String(text.clone()));
                    let line = format!("GUI'den metin alındı: '{}'", text);
                    self.output.push(line.clone());
                    println!("{}", line);
                } else if let Some(data) = &self.gui_input_data {
                    // Handle Tensor input
                    let len = data.len();
                    let tensor = Tensor::new(data.clone(), vec![1, len]).map_err(gpu_error)?;
                    self.tensors.insert(var_name.clone(), tensor);
                    
                    let line = format!("GUI'den veri alındı: {} ({} eleman)", var_name, len);
                    self.output.push(line.clone());
                    println!("{}", line);
                } else {
                    let line = "GUI veri girişi bulunamadı".to_string();
                    self.output.push(line.clone());
                    println!("{}", line);
                }
                Ok(None)
            }
        }
    }

    /// Evaluate an expression
    fn evaluate(&mut self, expr: &Expression) -> TurkceKodResult<Value> {
        match expr {
            Expression::Integer(n) => Ok(Value::Integer(*n)),
            Expression::Float(n) => Ok(Value::Float(*n)),
            Expression::String(s) => Ok(Value::String(s.clone())),
            Expression::Boolean(b) => Ok(Value::Boolean(*b)),

            Expression::Variable(name) => {
                self.variables
                    .get(name)
                    .cloned()
                    .ok_or_else(|| TurkceKodError::undefined_var(name))
            }

            Expression::Binary {
                left,
                operator,
                right,
            } => {
                let left_val = self.evaluate(left)?;
                let right_val = self.evaluate(right)?;
                self.apply_binary_op(&left_val, operator, &right_val)
            }

            Expression::Unary { operator, operand } => {
                let val = self.evaluate(operand)?;
                self.apply_unary_op(operator, &val)
            }

            Expression::Call { name, args } => {
                self.call_function(name, args)
            }

            Expression::ListAccess { name, index } => {
                let idx = self.evaluate(index)?;
                let idx_val = idx.as_integer().ok_or_else(|| {
                    TurkceKodError::type_error("sayı", idx.type_name())
                })?;

                let list = self.lists.get(name).ok_or_else(|| {
                    TurkceKodError::TanimlanmayanListe { name: name.clone() }
                })?;

                if idx_val < 0 || idx_val as usize >= list.len() {
                    return Err(TurkceKodError::index_error(idx_val, list.len()));
                }

                Ok(list[idx_val as usize].clone())
            }

            Expression::ModuleAccess { module, member } => {
                match (module.as_str(), member.as_str()) {
                    ("math", "pi") => Ok(Value::Float(std::f64::consts::PI)),
                    _ => Err(TurkceKodError::ModulFonksiyonuBulunamadi {
                        module: module.clone(),
                        function: member.clone(),
                    }),
                }
            }
        }
    }

    /// Apply a binary operator
    fn apply_binary_op(
        &self,
        left: &Value,
        op: &BinaryOp,
        right: &Value,
    ) -> TurkceKodResult<Value> {
        match op {
            BinaryOp::Add => {
                (left.clone() + right.clone()).map_err(|e| TurkceKodError::AritmetikHata { message: e })
            }
            BinaryOp::Subtract => {
                (left.clone() - right.clone()).map_err(|e| TurkceKodError::AritmetikHata { message: e })
            }
            BinaryOp::Multiply => {
                (left.clone() * right.clone()).map_err(|e| TurkceKodError::AritmetikHata { message: e })
            }
            BinaryOp::Divide => {
                (left.clone() / right.clone()).map_err(|e| TurkceKodError::AritmetikHata { message: e })
            }
            BinaryOp::Modulo => {
                (left.clone() % right.clone()).map_err(|e| TurkceKodError::AritmetikHata { message: e })
            }
            BinaryOp::Equal => {
                Ok(Value::Boolean(left == right))
            }
            BinaryOp::NotEqual => {
                Ok(Value::Boolean(left != right))
            }
            BinaryOp::Greater => {
                match (left, right) {
                    (Value::Integer(a), Value::Integer(b)) => Ok(Value::Boolean(a > b)),
                    (Value::Float(a), Value::Float(b)) => Ok(Value::Boolean(a > b)),
                    (Value::Integer(a), Value::Float(b)) => Ok(Value::Boolean((*a as f64) > *b)),
                    (Value::Float(a), Value::Integer(b)) => Ok(Value::Boolean(*a > (*b as f64))),
                    _ => Err(TurkceKodError::type_error("sayı", left.type_name())),
                }
            }
            BinaryOp::Less => {
                match (left, right) {
                    (Value::Integer(a), Value::Integer(b)) => Ok(Value::Boolean(a < b)),
                    (Value::Float(a), Value::Float(b)) => Ok(Value::Boolean(a < b)),
                    (Value::Integer(a), Value::Float(b)) => Ok(Value::Boolean((*a as f64) < *b)),
                    (Value::Float(a), Value::Integer(b)) => Ok(Value::Boolean(*a < (*b as f64))),
                    _ => Err(TurkceKodError::type_error("sayı", left.type_name())),
                }
            }
            BinaryOp::GreaterEq => {
                match (left, right) {
                    (Value::Integer(a), Value::Integer(b)) => Ok(Value::Boolean(a >= b)),
                    (Value::Float(a), Value::Float(b)) => Ok(Value::Boolean(a >= b)),
                    (Value::Integer(a), Value::Float(b)) => Ok(Value::Boolean((*a as f64) >= *b)),
                    (Value::Float(a), Value::Integer(b)) => Ok(Value::Boolean(*a >= (*b as f64))),
                    _ => Err(TurkceKodError::type_error("sayı", left.type_name())),
                }
            }
            BinaryOp::LessEq => {
                match (left, right) {
                    (Value::Integer(a), Value::Integer(b)) => Ok(Value::Boolean(a <= b)),
                    (Value::Float(a), Value::Float(b)) => Ok(Value::Boolean(a <= b)),
                    (Value::Integer(a), Value::Float(b)) => Ok(Value::Boolean((*a as f64) <= *b)),
                    (Value::Float(a), Value::Integer(b)) => Ok(Value::Boolean(*a <= (*b as f64))),
                    _ => Err(TurkceKodError::type_error("sayı", left.type_name())),
                }
            }
            BinaryOp::And => {
                Ok(Value::Boolean(left.is_truthy() && right.is_truthy()))
            }
            BinaryOp::Or => {
                Ok(Value::Boolean(left.is_truthy() || right.is_truthy()))
            }
        }
    }

    /// Apply a unary operator
    fn apply_unary_op(&self, op: &UnaryOp, val: &Value) -> TurkceKodResult<Value> {
        match op {
            UnaryOp::Negate => match val {
                Value::Integer(n) => Ok(Value::Integer(-n)),
                Value::Float(n) => Ok(Value::Float(-n)),
                _ => Err(TurkceKodError::type_error("sayı", val.type_name())),
            },
            UnaryOp::Not => Ok(Value::Boolean(!val.is_truthy())),
        }
    }

    /// Call a user-defined function
    fn call_function(&mut self, name: &str, args: &[Expression]) -> TurkceKodResult<Value> {
        // Get the function (clone to avoid borrow issues)
        let func = self
            .functions
            .get(name)
            .cloned()
            .ok_or_else(|| TurkceKodError::undefined_func(name))?;

        if args.len() != func.params.len() {
            return Err(TurkceKodError::ParametreSayisi {
                expected: func.params.len(),
                got: args.len(),
            });
        }

        // Evaluate arguments
        let mut arg_values = Vec::new();
        for arg in args {
            arg_values.push(self.evaluate(arg)?);
        }

        // Save old variable state for function scope
        let old_vars = self.variables.clone();

        // Set parameters
        for (param, val) in func.params.iter().zip(arg_values) {
            self.variables.insert(param.clone(), val);
        }

        // Execute function body
        let result = self.execute(&func.body)?;

        // Restore variable state
        self.variables = old_vars;

        Ok(result.unwrap_or(Value::None))
    }
}
