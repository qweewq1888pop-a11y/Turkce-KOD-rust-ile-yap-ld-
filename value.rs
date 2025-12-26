//! Türkçe Kod - Runtime Values
//! 
//! This module defines all possible runtime value types in the language.

use std::fmt;

/// Represents a value in the Türkçe Kod language
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// Integer (sayı)
    Integer(i64),
    /// Floating point (ondalık)
    Float(f64),
    /// String (metin)
    String(String),
    /// Boolean (mantıksal) - doğru/yanlış
    Boolean(bool),
    /// List (liste)
    List(Vec<Value>),
    /// Tensor (matris) - GPU-accelerated multi-dimensional array
    Tensor {
        data: Vec<f32>,
        shape: Vec<usize>,
    },
    /// None/Null (hiç)
    None,
}

impl Value {
    /// Check if value is truthy
    pub fn is_truthy(&self) -> bool {
        match self {
            Value::Boolean(b) => *b,
            Value::Integer(n) => *n != 0,
            Value::Float(n) => *n != 0.0,
            Value::String(s) => !s.is_empty(),
            Value::List(l) => !l.is_empty(),
            Value::Tensor { data, .. } => !data.is_empty(),
            Value::None => false,
        }
    }

    /// Get type name in Turkish
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Integer(_) => "sayı",
            Value::Float(_) => "ondalık",
            Value::String(_) => "metin",
            Value::Boolean(_) => "mantıksal",
            Value::List(_) => "liste",
            Value::Tensor { .. } => "matris",
            Value::None => "hiç",
        }
    }

    /// Try to convert to integer
    pub fn as_integer(&self) -> Option<i64> {
        match self {
            Value::Integer(n) => Some(*n),
            Value::Float(n) => Some(*n as i64),
            Value::String(s) => s.parse().ok(),
            Value::Boolean(b) => Some(if *b { 1 } else { 0 }),
            _ => None,
        }
    }

    /// Try to convert to float
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Value::Integer(n) => Some(*n as f64),
            Value::Float(n) => Some(*n),
            Value::String(s) => s.parse().ok(),
            Value::Boolean(b) => Some(if *b { 1.0 } else { 0.0 }),
            _ => None,
        }
    }

    /// Try to convert to string
    pub fn as_string(&self) -> String {
        match self {
            Value::Integer(n) => n.to_string(),
            Value::Float(n) => n.to_string(),
            Value::String(s) => s.clone(),
            Value::Boolean(b) => if *b { "doğru".to_string() } else { "yanlış".to_string() },
            Value::List(l) => format!("{:?}", l),
            Value::Tensor { shape, .. } => format!("Tensor şekil: {:?}", shape),
            Value::None => "hiç".to_string(),
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Integer(n) => write!(f, "{}", n),
            Value::Float(n) => write!(f, "{}", n),
            Value::String(s) => write!(f, "{}", s),
            Value::Boolean(b) => write!(f, "{}", if *b { "doğru" } else { "yanlış" }),
            Value::List(l) => {
                let items: Vec<String> = l.iter().map(|v| v.to_string()).collect();
                write!(f, "[{}]", items.join(", "))
            }
            Value::Tensor { data, shape } => {
                if data.len() <= 10 {
                    write!(f, "Tensor({:?}, şekil: {:?})", data, shape)
                } else {
                    write!(f, "Tensor([{:.4}, {:.4}, ... {} eleman], şekil: {:?})", 
                           data[0], data[1], data.len(), shape)
                }
            }
            Value::None => write!(f, "hiç"),
        }
    }
}

// Arithmetic operations
impl std::ops::Add for Value {
    type Output = Result<Value, String>;

    fn add(self, rhs: Self) -> Self::Output {
        match (&self, &rhs) {
            (Value::Integer(a), Value::Integer(b)) => Ok(Value::Integer(a + b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a + b)),
            (Value::Integer(a), Value::Float(b)) => Ok(Value::Float(*a as f64 + b)),
            (Value::Float(a), Value::Integer(b)) => Ok(Value::Float(a + *b as f64)),
            (Value::String(a), Value::String(b)) => Ok(Value::String(format!("{}{}", a, b))),
            // Tensor element-wise addition
            (Value::Tensor { data: d1, shape: s1 }, Value::Tensor { data: d2, shape: s2 }) => {
                if s1 != s2 {
                    return Err(format!("Tensor şekilleri uyuşmuyor: {:?} vs {:?}", s1, s2));
                }
                let result: Vec<f32> = d1.iter().zip(d2.iter()).map(|(a, b)| a + b).collect();
                Ok(Value::Tensor { data: result, shape: s1.clone() })
            }
            _ => Err(format!(
                "'{}' ve '{}' tipleri toplanamaz",
                self.type_name(),
                rhs.type_name()
            )),
        }
    }
}

impl std::ops::Sub for Value {
    type Output = Result<Value, String>;

    fn sub(self, rhs: Self) -> Self::Output {
        match (&self, &rhs) {
            (Value::Integer(a), Value::Integer(b)) => Ok(Value::Integer(a - b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a - b)),
            (Value::Integer(a), Value::Float(b)) => Ok(Value::Float(*a as f64 - b)),
            (Value::Float(a), Value::Integer(b)) => Ok(Value::Float(a - *b as f64)),
            // Tensor element-wise subtraction
            (Value::Tensor { data: d1, shape: s1 }, Value::Tensor { data: d2, shape: s2 }) => {
                if s1 != s2 {
                    return Err(format!("Tensor şekilleri uyuşmuyor: {:?} vs {:?}", s1, s2));
                }
                let result: Vec<f32> = d1.iter().zip(d2.iter()).map(|(a, b)| a - b).collect();
                Ok(Value::Tensor { data: result, shape: s1.clone() })
            }
            _ => Err(format!(
                "'{}' ve '{}' tipleri çıkarılamaz",
                self.type_name(),
                rhs.type_name()
            )),
        }
    }
}

impl std::ops::Mul for Value {
    type Output = Result<Value, String>;

    fn mul(self, rhs: Self) -> Self::Output {
        match (&self, &rhs) {
            (Value::Integer(a), Value::Integer(b)) => Ok(Value::Integer(a * b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a * b)),
            (Value::Integer(a), Value::Float(b)) => Ok(Value::Float(*a as f64 * b)),
            (Value::Float(a), Value::Integer(b)) => Ok(Value::Float(a * *b as f64)),
            (Value::String(s), Value::Integer(n)) => Ok(Value::String(s.repeat(*n as usize))),
            // Tensor element-wise multiplication
            (Value::Tensor { data: d1, shape: s1 }, Value::Tensor { data: d2, shape: s2 }) => {
                if s1 != s2 {
                    return Err(format!("Tensor şekilleri uyuşmuyor: {:?} vs {:?}", s1, s2));
                }
                let result: Vec<f32> = d1.iter().zip(d2.iter()).map(|(a, b)| a * b).collect();
                Ok(Value::Tensor { data: result, shape: s1.clone() })
            }
            _ => Err(format!(
                "'{}' ve '{}' tipleri çarpılamaz",
                self.type_name(),
                rhs.type_name()
            )),
        }
    }
}

impl std::ops::Div for Value {
    type Output = Result<Value, String>;

    fn div(self, rhs: Self) -> Self::Output {
        match (&self, &rhs) {
            (Value::Integer(a), Value::Integer(b)) => {
                if *b == 0 {
                    Err("Sıfıra bölme hatası".to_string())
                } else {
                    Ok(Value::Integer(a / b))
                }
            }
            (Value::Float(a), Value::Float(b)) => {
                if *b == 0.0 {
                    Err("Sıfıra bölme hatası".to_string())
                } else {
                    Ok(Value::Float(a / b))
                }
            }
            (Value::Integer(a), Value::Float(b)) => {
                if *b == 0.0 {
                    Err("Sıfıra bölme hatası".to_string())
                } else {
                    Ok(Value::Float(*a as f64 / b))
                }
            }
            (Value::Float(a), Value::Integer(b)) => {
                if *b == 0 {
                    Err("Sıfıra bölme hatası".to_string())
                } else {
                    Ok(Value::Float(a / *b as f64))
                }
            }
            _ => Err(format!(
                "'{}' ve '{}' tipleri bölünemez",
                self.type_name(),
                rhs.type_name()
            )),
        }
    }
}

impl std::ops::Rem for Value {
    type Output = Result<Value, String>;

    fn rem(self, rhs: Self) -> Self::Output {
        match (&self, &rhs) {
            (Value::Integer(a), Value::Integer(b)) => {
                if *b == 0 {
                    Err("Sıfıra bölme hatası".to_string())
                } else {
                    Ok(Value::Integer(a % b))
                }
            }
            _ => Err(format!(
                "'{}' ve '{}' tipleri mod alınamaz",
                self.type_name(),
                rhs.type_name()
            )),
        }
    }
}
