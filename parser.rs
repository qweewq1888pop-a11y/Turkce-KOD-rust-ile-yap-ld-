//! Türkçe Kod - Parser
//! 
//! Converts tokens into an Abstract Syntax Tree (AST).

use crate::error::{TurkceKodError, TurkceKodResult};
use crate::lexer::{Token, TokenInfo};

/// Expression types in the AST
#[derive(Debug, Clone)]
pub enum Expression {
    /// Integer literal
    Integer(i64),
    /// Float literal
    Float(f64),
    /// String literal
    String(String),
    /// Boolean literal
    Boolean(bool),
    /// Variable reference
    Variable(String),
    /// Binary operation
    Binary {
        left: Box<Expression>,
        operator: BinaryOp,
        right: Box<Expression>,
    },
    /// Unary operation
    Unary {
        operator: UnaryOp,
        operand: Box<Expression>,
    },
    /// Function call
    Call {
        name: String,
        args: Vec<Expression>,
    },
    /// List access: liste[index]
    ListAccess {
        name: String,
        index: Box<Expression>,
    },
    /// Module access: math.pi
    ModuleAccess {
        module: String,
        member: String,
    },
}

/// Binary operators
#[derive(Debug, Clone)]
pub enum BinaryOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    Equal,
    NotEqual,
    Greater,
    Less,
    GreaterEq,
    LessEq,
    And,
    Or,
}

/// Unary operators
#[derive(Debug, Clone)]
pub enum UnaryOp {
    Negate,
    Not,
}

/// Statement types in the AST
#[derive(Debug, Clone)]
pub enum Statement {
    /// Print statement: yaz "text"
    Print(Vec<Expression>),
    
    /// Variable declaration: sayıv x = 5
    VarDecl {
        var_type: VarType,
        name: String,
        value: Option<Expression>,
    },
    
    /// Assignment: x = value
    Assignment {
        name: String,
        value: Expression,
    },
    
    /// Calculate: hesapla x = expr
    Calculate {
        name: String,
        expr: Expression,
    },
    
    /// If statement: eğer condition { ... } değilse { ... }
    If {
        condition: Expression,
        then_block: Vec<Statement>,
        else_block: Option<Vec<Statement>>,
    },
    
    /// While loop: iken condition { ... }
    While {
        condition: Expression,
        body: Vec<Statement>,
    },
    
    /// Repeat loop: tekrar count { ... }
    Repeat {
        count: Expression,
        body: Vec<Statement>,
    },
    
    /// For each: her_eleman_için liste { ... }
    ForEach {
        list_name: String,
        body: Vec<Statement>,
    },
    
    /// Function definition: işlev name(params) { ... }
    FunctionDef {
        name: String,
        params: Vec<String>,
        body: Vec<Statement>,
    },
    
    /// Return: dön value
    Return(Option<Expression>),
    
    /// Function call as statement
    FunctionCall {
        name: String,
        args: Vec<Expression>,
    },
    
    /// List declaration: listav name
    ListDecl {
        name: String,
    },
    
    /// Add to list: ekle list value
    ListAdd {
        list_name: String,
        value: Expression,
    },
    
    /// Get length: uzunluk list
    ListLength {
        list_name: String,
    },
    
    /// Get element: al list index
    ListGet {
        list_name: String,
        index: Expression,
    },
    
    /// Read input: oku, oku_int, oku_float
    ReadInput {
        var_name: String,
        input_type: InputType,
    },
    
    /// Read file: oku_dosya var path
    ReadFile {
        var_name: String,
        path: Expression,
    },
    
    /// Write file: yaz_dosya var path
    WriteFile {
        var_name: String,
        path: Expression,
    },
    
    /// Sleep: bekle seconds
    Sleep(Expression),
    
    /// Math function: karekok, sinus, etc.
    MathFunc {
        func: MathFunction,
        target_var: String,
        arg: Expression,
    },
    
    // -------------------------------------------------------------------------
    // GPU/Tensor Statements
    // -------------------------------------------------------------------------
    
    /// Tensor declaration: matris a = [1,2,3] boyut [3]
    TensorDecl {
        name: String,
        data: Vec<Expression>,
        shape: Vec<usize>,
    },
    
    /// Autograd Backward: geri_yayilim variable
    Backward {
        variable: String,
    },
    
    // File System Statements
    CreateFolder { path: String },
    CreateAiDirs,
    
    /// Random tensor: matris a = rastgele [100, 100]
    RandomTensor {
        name: String,
        shape: Vec<usize>,
    },
    
    /// GPU operation: matris c = gpu_çarp a b
    GpuOp {
        result: String,
        op: GpuOperation,
        args: Vec<String>,
    },
    
    /// GPU info: gpu_bilgi
    GpuInfo,
    
    /// GPU mode: gpu_mod "hybrid"
    GpuMode(String),
    
    // -------------------------------------------------------------------------
    // AI Training Statements
    // -------------------------------------------------------------------------
    
    /// Loss calculation: capraz_entropi kayip tahmin hedef
    Loss {
        result: String,
        loss_type: LossType,
        predicted: String,
        target: String,
    },
    
    /// Backpropagation: geri_yayilim kayip_degeri
    Backprop {
        loss_var: String,
    },
    
    /// Zero gradients: sifir_gradyan
    ZeroGrad,
    
    /// Optimizer declaration: optimizer sgd "sgd" 0.01
    OptimizerDecl {
        name: String,
        opt_type: OptimizerType,
        learning_rate: f64,
    },
    
    /// Optimizer step: adim optimizer_name
    OptimizerStep {
        optimizer: String,
    },
    
    /// Training loop: egit epochs batch_size { ... }
    Train {
        epochs: Expression,
        batch_size: Expression,
        body: Vec<Statement>,
    },
    
    /// Data loading: veri_yukle var "path" (or type specific loaders)
    LoadData {
        var_name: String,
        source: DataSource,
    },
    
    // ============================================
    // GUI Statements (TürkçeInter)
    // ============================================
    
    /// GUI Widget declaration: type id "text" { children }
    GuiDecl {
        widget_type: GuiWidgetType,
        id: String,
        text: String,
        commands: Vec<Statement>, // For children or callbacks
    },
    
    /// Update widget property: degistir id "text" new_value
    GuiUpdate {
        id: String,
        property: String, // "metin" (text), "renk" (color), etc.
        value: Expression,
    },
    
    /// Read from console: konsol_oku var
    ConsoleRead {
        var_name: String,
    },
    
    /// Get GUI input data: gui_veri [widget_id] var
    GuiData {
        widget_id: Option<String>,
        var_name: String,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum GuiWidgetType {
    Window, // pencere
    Button, // buton
    Label,  // etiket
    Input,  // girdi
    Layout, // yerlesim
    Chat,   // sohbet
}

/// GPU Operations
#[derive(Debug, Clone)]
pub enum GpuOperation {
    // Matrix operations
    MatMul,     // gpu_çarp
    Add,        // gpu_topla
    Sub,        // gpu_çıkar
    Mul,        // element-wise multiply
    Div,        // gpu_böl
    Transpose,  // gpu_transpoz
    
    // Activation functions
    Relu,       // gpu_relu
    Gelu,       // gpu_gelu
    Sigmoid,    // gpu_sigmoid
    Silu,       // gpu_silu
    Tanh,       // gpu_tanh
    
    // Normalization
    Softmax,    // gpu_softmax
    LayerNorm,  // gpu_layernorm
    RmsNorm,    // gpu_rmsnorm
}

/// Variable types
#[derive(Debug, Clone)]
pub enum VarType {
    Integer,  // sayıv
    String,   // metinv
    Boolean,  // mantıksalv
}

/// Input types
#[derive(Debug, Clone)]
pub enum InputType {
    String,  // oku
    Integer, // oku_int
    Float,   // oku_float
}

/// Math functions
#[derive(Debug, Clone)]
pub enum MathFunction {
    Sqrt,    // karekok
    Sin,     // sinus
    Cos,     // cosinus
    Tan,     // tanjant
}

/// Loss function types
#[derive(Debug, Clone)]
pub enum LossType {
    CrossEntropy,  // capraz_entropi
    MSE,           // mse_kayip
}

/// Optimizer types
#[derive(Debug, Clone)]
pub enum OptimizerType {
    SGD,   // sgd
    Adam,  // adam
}

/// Data source types
#[derive(Debug, Clone)]
pub enum DataSource {
    File(String),  // file path
    Cifar10,       // cifar_yukle
    Console,       // konsol_oku
    Gui,           // gui_veri
}

/// Parser for Türkçe Kod
pub struct Parser {
    tokens: Vec<TokenInfo>,
    pos: usize,
}

impl Parser {
    /// Create a new parser from tokens
    pub fn new(tokens: Vec<TokenInfo>) -> Self {
        Self { tokens, pos: 0 }
    }

    /// Parse all tokens into a list of statements
    pub fn parse(&mut self) -> TurkceKodResult<Vec<Statement>> {
        let mut statements = Vec::new();

        while !self.is_at_end() {
            if let Some(stmt) = self.parse_statement()? {
                statements.push(stmt);
            }
        }

        Ok(statements)
    }

    fn is_at_end(&self) -> bool {
        matches!(self.current().token, Token::Eof)
    }

    fn current(&self) -> &TokenInfo {
        &self.tokens[self.pos.min(self.tokens.len() - 1)]
    }

    fn peek(&self) -> &TokenInfo {
        &self.tokens[(self.pos + 1).min(self.tokens.len() - 1)]
    }

    fn advance(&mut self) -> &TokenInfo {
        if !self.is_at_end() {
            self.pos += 1;
        }
        &self.tokens[self.pos - 1]
    }

    fn check(&self, token: &Token) -> bool {
        std::mem::discriminant(&self.current().token) == std::mem::discriminant(token)
    }

    fn consume(&mut self, expected: Token, error_msg: &str) -> TurkceKodResult<&TokenInfo> {
        if std::mem::discriminant(&self.current().token) == std::mem::discriminant(&expected) {
            Ok(self.advance())
        } else {
            Err(TurkceKodError::BeklenmeyenSimge {
                line: self.current().line,
                expected: error_msg.to_string(),
                found: format!("{:?}", self.current().token),
            })
        }
    }

    /// Helper: Extract identifier or return error
    /// Reduces ~20 duplicate identifier extraction patterns to single function
    fn expect_identifier(&mut self, error_msg: &str) -> TurkceKodResult<String> {
        match &self.current().token {
            Token::Identifier(n) => {
                let name = n.clone();
                self.advance();
                Ok(name)
            }
            _ => Err(TurkceKodError::syntax_error(self.current().line, error_msg)),
        }
    }

    /// Convert GPU token to GpuOperation enum
    fn token_to_gpu_op(token: &Token) -> Option<GpuOperation> {
        Some(match token {
            Token::GpuCarp => GpuOperation::MatMul,
            Token::GpuTopla => GpuOperation::Add,
            Token::GpuCikar => GpuOperation::Sub,
            Token::GpuBol => GpuOperation::Div,
            Token::GpuRelu => GpuOperation::Relu,
            Token::GpuGelu => GpuOperation::Gelu,
            Token::GpuSigmoid => GpuOperation::Sigmoid,
            Token::GpuSilu => GpuOperation::Silu,
            Token::GpuTanh => GpuOperation::Tanh,
            Token::GpuSoftmax => GpuOperation::Softmax,
            Token::GpuLayernorm => GpuOperation::LayerNorm,
            Token::GpuRmsnorm => GpuOperation::RmsNorm,
            Token::GpuTranspoz => GpuOperation::Transpose,
            _ => return None,
        })
    }

    /// Check if current token is a GPU operation
    fn is_gpu_op(&self) -> bool {
        Self::token_to_gpu_op(&self.current().token).is_some()
    }

    fn parse_statement(&mut self) -> TurkceKodResult<Option<Statement>> {
        let token = self.current().clone();

        let stmt = match &token.token {
            Token::Yaz => self.parse_print()?,
            Token::Sayiv => self.parse_var_decl(VarType::Integer)?,
            Token::Metinv => self.parse_var_decl(VarType::String)?,
            Token::Mantiksalv => self.parse_var_decl(VarType::Boolean)?,
            Token::Hesapla => self.parse_calculate()?,
            Token::Eger => self.parse_if()?,
            Token::Iken => self.parse_while()?,
            Token::Tekrar => self.parse_repeat()?,
            Token::HerElemanIcin => self.parse_foreach()?,
            Token::Islev => self.parse_function_def()?,
            Token::Don => self.parse_return()?,
            Token::Listav => self.parse_list_decl()?,
            Token::Ekle => self.parse_list_add()?,
            Token::Uzunluk => self.parse_list_length()?,
            Token::Al => self.parse_list_get()?,
            Token::Oku => self.parse_input(InputType::String)?,
            Token::OkuInt => self.parse_input(InputType::Integer)?,
            Token::OkuFloat => self.parse_input(InputType::Float)?,
            Token::OkuDosya => self.parse_read_file()?,
            Token::YazDosya => self.parse_write_file()?,
            Token::Bekle => self.parse_sleep()?,
            Token::Karekok => self.parse_math_func(MathFunction::Sqrt)?,
            Token::Sinus => self.parse_math_func(MathFunction::Sin)?,
            Token::Cosinus => self.parse_math_func(MathFunction::Cos)?,
            Token::Tanjant => self.parse_math_func(MathFunction::Tan)?,
            
            // GPU/Tensor statements
            Token::Matris => self.parse_tensor_decl()?,
            Token::GpuBilgi => { self.advance(); Statement::GpuInfo },
            Token::GpuMod => self.parse_gpu_mode()?,
            Token::GpuCarp => self.parse_gpu_op(GpuOperation::MatMul)?,
            Token::GpuTopla => self.parse_gpu_op(GpuOperation::Add)?,
            Token::GpuCikar => self.parse_gpu_op(GpuOperation::Sub)?,
            Token::GpuBol => self.parse_gpu_op(GpuOperation::Div)?,
            Token::GpuRelu => self.parse_gpu_unary_op(GpuOperation::Relu)?,
            Token::GpuGelu => self.parse_gpu_unary_op(GpuOperation::Gelu)?,
            Token::GpuSigmoid => self.parse_gpu_unary_op(GpuOperation::Sigmoid)?,
            Token::GpuSilu => self.parse_gpu_unary_op(GpuOperation::Silu)?,
            Token::GpuTanh => self.parse_gpu_unary_op(GpuOperation::Tanh)?,
            Token::GpuSoftmax => self.parse_gpu_unary_op(GpuOperation::Softmax)?,
            Token::GpuLayernorm => self.parse_gpu_op(GpuOperation::LayerNorm)?,
            Token::GpuRmsnorm => self.parse_gpu_op(GpuOperation::RmsNorm)?,
            Token::GpuTranspoz => self.parse_gpu_unary_op(GpuOperation::Transpose)?,
            
            // AI Training statements
            Token::CaprazEntropi => self.parse_loss(LossType::CrossEntropy)?,
            Token::MseKayip => self.parse_loss(LossType::MSE)?,
            Token::GeriYayilim => self.parse_backprop()?,
            Token::SifirGradyan => { self.advance(); Statement::ZeroGrad },
            Token::OptimizerTanim => self.parse_optimizer_decl()?,
            Token::Adim => self.parse_optimizer_step()?,
            Token::Egit => self.parse_train()?,
            Token::VeriYukle => self.parse_load_data()?,

            Token::KonsolOku => self.parse_console_read()?,
            Token::GuiVeri => self.parse_gui_data()?,
            
            // GUI Statements
            Token::Pencere => self.parse_gui_decl(GuiWidgetType::Window)?,
            Token::Buton => self.parse_gui_decl(GuiWidgetType::Button)?,
            Token::Etiket => self.parse_gui_decl(GuiWidgetType::Label)?,
            Token::Girdi => self.parse_gui_decl(GuiWidgetType::Input)?,
            Token::Yerlesim => self.parse_gui_decl(GuiWidgetType::Layout)?,
            Token::Sohbet => self.parse_gui_decl(GuiWidgetType::Chat)?,
            Token::Degistir => self.parse_gui_update()?,
            
            // File System
            Token::KlasorOlustur => self.parse_create_folder()?,
            Token::AiDizinleriOlustur => {
                self.advance();
                Statement::CreateAiDirs
            },
            Token::GeriYayilim => {
                self.advance();
                let name = self.expect_identifier("Geri yayılım için değişken adı bekleniyor")?;
                Statement::Backward { variable: name }
            },
            
            // Standalone random tensor: rastgele name [shape]
            Token::Rastgele => self.parse_random_tensor()?,
            
            Token::Identifier(_) => self.parse_identifier_statement()?,
            Token::Degilse => {
                // Skip standalone değilse (handled by if)
                self.advance();
                self.skip_block()?;
                return Ok(None);
            }
            Token::CloseBrace => {
                self.advance();
                return Ok(None);
            }
            Token::Eof => return Ok(None),
            _ => {
                return Err(TurkceKodError::BilinmeyenKomut {
                    command: format!("{:?}", token.token),
                });
            }
        };

        Ok(Some(stmt))
    }

    fn parse_print(&mut self) -> TurkceKodResult<Statement> {
        self.advance(); // consume 'yaz'
        let mut exprs = Vec::new();

        while !self.is_at_end() && !self.check(&Token::Eof) {
            let expr = self.parse_expression()?;
            exprs.push(expr);

            if !self.check(&Token::Comma) {
                break;
            }
            self.advance(); // consume comma
        }

        Ok(Statement::Print(exprs))
    }

    fn parse_var_decl(&mut self, var_type: VarType) -> TurkceKodResult<Statement> {
        self.advance(); // consume type keyword

        let name = match &self.current().token {
            Token::Identifier(n) => n.clone(),
            _ => {
                return Err(TurkceKodError::syntax_error(
                    self.current().line,
                    "Değişken adı bekleniyor",
                ));
            }
        };
        self.advance();

        let value = if self.check(&Token::Equals) {
            self.advance(); // consume '='
            Some(self.parse_expression()?)
        } else if self.check(&Token::String("".to_string())) {
            // metinv isim "value" format
            Some(self.parse_expression()?)
        } else {
            None
        };

        Ok(Statement::VarDecl {
            var_type,
            name,
            value,
        })
    }

    fn parse_calculate(&mut self) -> TurkceKodResult<Statement> {
        self.advance(); // consume 'hesapla'

        let name = match &self.current().token {
            Token::Identifier(n) => n.clone(),
            _ => {
                return Err(TurkceKodError::syntax_error(
                    self.current().line,
                    "Değişken adı bekleniyor",
                ));
            }
        };
        self.advance();

        self.consume(Token::Equals, "'=' bekleniyor")?;
        let expr = self.parse_expression()?;

        Ok(Statement::Calculate { name, expr })
    }

    fn parse_if(&mut self) -> TurkceKodResult<Statement> {
        self.advance(); // consume 'eğer'

        let condition = self.parse_expression()?;

        self.consume(Token::OpenBrace, "'{' bekleniyor")?;
        let then_block = self.parse_block()?;

        let else_block = if self.check(&Token::Degilse) {
            self.advance(); // consume 'değilse'
            self.consume(Token::OpenBrace, "'{' bekleniyor")?;
            Some(self.parse_block()?)
        } else {
            None
        };

        Ok(Statement::If {
            condition,
            then_block,
            else_block,
        })
    }

    fn parse_while(&mut self) -> TurkceKodResult<Statement> {
        self.advance(); // consume 'iken'

        let condition = self.parse_expression()?;

        self.consume(Token::OpenBrace, "'{' bekleniyor")?;
        let body = self.parse_block()?;

        Ok(Statement::While { condition, body })
    }

    fn parse_repeat(&mut self) -> TurkceKodResult<Statement> {
        self.advance(); // consume 'tekrar'

        let count = self.parse_expression()?;

        self.consume(Token::OpenBrace, "'{' bekleniyor")?;
        let body = self.parse_block()?;

        Ok(Statement::Repeat { count, body })
    }

    fn parse_foreach(&mut self) -> TurkceKodResult<Statement> {
        self.advance(); // consume 'her_eleman_için'

        let list_name = match &self.current().token {
            Token::Identifier(n) => n.clone(),
            _ => {
                return Err(TurkceKodError::syntax_error(
                    self.current().line,
                    "Liste adı bekleniyor",
                ));
            }
        };
        self.advance();

        self.consume(Token::OpenBrace, "'{' bekleniyor")?;
        let body = self.parse_block()?;

        Ok(Statement::ForEach { list_name, body })
    }

    fn parse_function_def(&mut self) -> TurkceKodResult<Statement> {
        self.advance(); // consume 'işlev'

        let name = match &self.current().token {
            Token::Identifier(n) => n.clone(),
            _ => {
                return Err(TurkceKodError::syntax_error(
                    self.current().line,
                    "Fonksiyon adı bekleniyor",
                ));
            }
        };
        self.advance();

        self.consume(Token::OpenParen, "'(' bekleniyor")?;

        let mut params = Vec::new();
        while !self.check(&Token::CloseParen) {
            if let Token::Identifier(p) = &self.current().token {
                params.push(p.clone());
                self.advance();
            }
            if self.check(&Token::Comma) {
                self.advance();
            }
        }

        self.consume(Token::CloseParen, "')' bekleniyor")?;
        self.consume(Token::OpenBrace, "'{' bekleniyor")?;

        let body = self.parse_block()?;

        Ok(Statement::FunctionDef { name, params, body })
    }

    fn parse_return(&mut self) -> TurkceKodResult<Statement> {
        self.advance(); // consume 'dön'

        let value = if !self.is_at_end() && !self.check(&Token::CloseBrace) {
            Some(self.parse_expression()?)
        } else {
            None
        };

        Ok(Statement::Return(value))
    }

    fn parse_list_decl(&mut self) -> TurkceKodResult<Statement> {
        self.advance(); // consume 'listav'

        let name = match &self.current().token {
            Token::Identifier(n) => n.clone(),
            _ => {
                return Err(TurkceKodError::syntax_error(
                    self.current().line,
                    "Liste adı bekleniyor",
                ));
            }
        };
        self.advance();

        Ok(Statement::ListDecl { name })
    }

    fn parse_list_add(&mut self) -> TurkceKodResult<Statement> {
        self.advance(); // consume 'ekle'

        let list_name = match &self.current().token {
            Token::Identifier(n) => n.clone(),
            _ => {
                return Err(TurkceKodError::syntax_error(
                    self.current().line,
                    "Liste adı bekleniyor",
                ));
            }
        };
        self.advance();

        let value = self.parse_expression()?;

        Ok(Statement::ListAdd { list_name, value })
    }

    fn parse_list_length(&mut self) -> TurkceKodResult<Statement> {
        self.advance(); // consume 'uzunluk'

        let list_name = match &self.current().token {
            Token::Identifier(n) => n.clone(),
            _ => {
                return Err(TurkceKodError::syntax_error(
                    self.current().line,
                    "Liste adı bekleniyor",
                ));
            }
        };
        self.advance();

        Ok(Statement::ListLength { list_name })
    }

    fn parse_list_get(&mut self) -> TurkceKodResult<Statement> {
        self.advance(); // consume 'al'

        let list_name = match &self.current().token {
            Token::Identifier(n) => n.clone(),
            _ => {
                return Err(TurkceKodError::syntax_error(
                    self.current().line,
                    "Liste adı bekleniyor",
                ));
            }
        };
        self.advance();

        let index = self.parse_expression()?;

        Ok(Statement::ListGet { list_name, index })
    }

    fn parse_input(&mut self, input_type: InputType) -> TurkceKodResult<Statement> {
        self.advance(); // consume input keyword

        let var_name = match &self.current().token {
            Token::Identifier(n) => n.clone(),
            _ => {
                return Err(TurkceKodError::syntax_error(
                    self.current().line,
                    "Değişken adı bekleniyor",
                ));
            }
        };
        self.advance();

        Ok(Statement::ReadInput {
            var_name,
            input_type,
        })
    }

    fn parse_read_file(&mut self) -> TurkceKodResult<Statement> {
        self.advance(); // consume 'oku_dosya'

        let var_name = match &self.current().token {
            Token::Identifier(n) => n.clone(),
            _ => {
                return Err(TurkceKodError::syntax_error(
                    self.current().line,
                    "Değişken adı bekleniyor",
                ));
            }
        };
        self.advance();

        let path = self.parse_expression()?;

        Ok(Statement::ReadFile { var_name, path })
    }

    fn parse_write_file(&mut self) -> TurkceKodResult<Statement> {
        self.advance(); // consume 'yaz_dosya'

        let var_name = match &self.current().token {
            Token::Identifier(n) => n.clone(),
            _ => {
                return Err(TurkceKodError::syntax_error(
                    self.current().line,
                    "Değişken adı bekleniyor",
                ));
            }
        };
        self.advance();

        let path = self.parse_expression()?;

        Ok(Statement::WriteFile { var_name, path })
    }

    fn parse_sleep(&mut self) -> TurkceKodResult<Statement> {
        self.advance(); // consume 'bekle'
        let duration = self.parse_expression()?;
        Ok(Statement::Sleep(duration))
    }

    fn parse_math_func(&mut self, func: MathFunction) -> TurkceKodResult<Statement> {
        self.advance(); // consume function keyword

        let target_var = match &self.current().token {
            Token::Identifier(n) => n.clone(),
            _ => {
                return Err(TurkceKodError::syntax_error(
                    self.current().line,
                    "Değişken adı bekleniyor",
                ));
            }
        };
        self.advance();

        let arg = self.parse_expression()?;

        Ok(Statement::MathFunc {
            func,
            target_var,
            arg,
        })
    }

    fn parse_identifier_statement(&mut self) -> TurkceKodResult<Statement> {
        let name = match &self.current().token {
            Token::Identifier(n) => n.clone(),
            _ => unreachable!(),
        };
        self.advance();

        // Check for assignment
        if self.check(&Token::Equals) {
            self.advance();
            let value = self.parse_expression()?;
            return Ok(Statement::Assignment { name, value });
        }

        // Check for function call
        if self.check(&Token::OpenParen) {
            self.advance();
            let mut args = Vec::new();

            while !self.check(&Token::CloseParen) {
                args.push(self.parse_expression()?);
                if self.check(&Token::Comma) {
                    self.advance();
                }
            }

            self.consume(Token::CloseParen, "')' bekleniyor")?;
            return Ok(Statement::FunctionCall { name, args });
        }

        // Treat as function call without parens
        Ok(Statement::FunctionCall {
            name,
            args: Vec::new(),
        })
    }

    fn parse_block(&mut self) -> TurkceKodResult<Vec<Statement>> {
        let mut statements = Vec::new();

        while !self.check(&Token::CloseBrace) && !self.is_at_end() {
            if let Some(stmt) = self.parse_statement()? {
                statements.push(stmt);
            }
        }

        if self.check(&Token::CloseBrace) {
            self.advance();
        }

        Ok(statements)
    }

    fn skip_block(&mut self) -> TurkceKodResult<()> {
        if self.check(&Token::OpenBrace) {
            self.advance();
            let mut depth = 1;
            while depth > 0 && !self.is_at_end() {
                if self.check(&Token::OpenBrace) {
                    depth += 1;
                } else if self.check(&Token::CloseBrace) {
                    depth -= 1;
                }
                self.advance();
            }
        }
        Ok(())
    }

    // Expression parsing with precedence
    fn parse_expression(&mut self) -> TurkceKodResult<Expression> {
        self.parse_or()
    }

    fn parse_or(&mut self) -> TurkceKodResult<Expression> {
        let mut left = self.parse_and()?;

        while self.check(&Token::Or) {
            self.advance();
            let right = self.parse_and()?;
            left = Expression::Binary {
                left: Box::new(left),
                operator: BinaryOp::Or,
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn parse_and(&mut self) -> TurkceKodResult<Expression> {
        let mut left = self.parse_comparison()?;

        while self.check(&Token::And) {
            self.advance();
            let right = self.parse_comparison()?;
            left = Expression::Binary {
                left: Box::new(left),
                operator: BinaryOp::And,
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn parse_comparison(&mut self) -> TurkceKodResult<Expression> {
        let mut left = self.parse_term()?;

        loop {
            let op = match &self.current().token {
                Token::Greater | Token::Buyukse => BinaryOp::Greater,
                Token::Less | Token::Kucukse => BinaryOp::Less,
                Token::GreaterEq => BinaryOp::GreaterEq,
                Token::LessEq => BinaryOp::LessEq,
                Token::EqualsEquals | Token::Esitse => BinaryOp::Equal,
                Token::NotEquals => BinaryOp::NotEqual,
                _ => break,
            };

            self.advance();
            let right = self.parse_term()?;
            left = Expression::Binary {
                left: Box::new(left),
                operator: op,
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn parse_term(&mut self) -> TurkceKodResult<Expression> {
        let mut left = self.parse_factor()?;

        loop {
            let op = match &self.current().token {
                Token::Plus => BinaryOp::Add,
                Token::Minus => BinaryOp::Subtract,
                _ => break,
            };

            self.advance();
            let right = self.parse_factor()?;
            left = Expression::Binary {
                left: Box::new(left),
                operator: op,
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn parse_factor(&mut self) -> TurkceKodResult<Expression> {
        let mut left = self.parse_unary()?;

        loop {
            let op = match &self.current().token {
                Token::Star => BinaryOp::Multiply,
                Token::Slash => BinaryOp::Divide,
                Token::Percent => BinaryOp::Modulo,
                _ => break,
            };

            self.advance();
            let right = self.parse_unary()?;
            left = Expression::Binary {
                left: Box::new(left),
                operator: op,
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn parse_unary(&mut self) -> TurkceKodResult<Expression> {
        if self.check(&Token::Minus) {
            self.advance();
            let operand = self.parse_unary()?;
            return Ok(Expression::Unary {
                operator: UnaryOp::Negate,
                operand: Box::new(operand),
            });
        }

        self.parse_primary()
    }

    fn parse_primary(&mut self) -> TurkceKodResult<Expression> {
        let token = self.current().clone();

        match &token.token {
            Token::Integer(n) => {
                let val = *n;
                self.advance();
                Ok(Expression::Integer(val))
            }
            Token::Float(n) => {
                let val = *n;
                self.advance();
                Ok(Expression::Float(val))
            }
            Token::String(s) => {
                let val = s.clone();
                self.advance();
                Ok(Expression::String(val))
            }
            Token::Boolean(b) => {
                let val = *b;
                self.advance();
                Ok(Expression::Boolean(val))
            }
            Token::Pi => {
                self.advance();
                Ok(Expression::Float(std::f64::consts::PI))
            }
            Token::Identifier(name) => {
                let name = name.clone();
                self.advance();

                // Check for module access (math.pi)
                if self.check(&Token::Dot) {
                    self.advance();
                    if let Token::Identifier(member) = &self.current().token {
                        let member = member.clone();
                        self.advance();
                        return Ok(Expression::ModuleAccess {
                            module: name,
                            member,
                        });
                    }
                }

                // Check for list access
                if self.check(&Token::OpenBracket) {
                    self.advance();
                    let index = self.parse_expression()?;
                    self.consume(Token::CloseBracket, "']' bekleniyor")?;
                    return Ok(Expression::ListAccess {
                        name,
                        index: Box::new(index),
                    });
                }

                // Check for function call
                if self.check(&Token::OpenParen) {
                    self.advance();
                    let mut args = Vec::new();

                    while !self.check(&Token::CloseParen) {
                        args.push(self.parse_expression()?);
                        if self.check(&Token::Comma) {
                            self.advance();
                        }
                    }

                    self.consume(Token::CloseParen, "')' bekleniyor")?;
                    return Ok(Expression::Call { name, args });
                }

                Ok(Expression::Variable(name))
            }
            Token::OpenParen => {
                self.advance();
                let expr = self.parse_expression()?;
                self.consume(Token::CloseParen, "')' bekleniyor")?;
                Ok(expr)
            }
            _ => Err(TurkceKodError::syntax_error(
                token.line,
                format!("Beklenmeyen simge: {:?}", token.token),
            )),
        }
    }
    
    // -------------------------------------------------------------------------
    // GPU/Tensor Parsing
    // -------------------------------------------------------------------------
    
    /// Parse tensor declaration: matris a = [1,2,3] boyut [3] OR matris a = rastgele [100, 100]
    fn parse_tensor_decl(&mut self) -> TurkceKodResult<Statement> {
        self.advance(); // consume 'matris'
        
        let name = match &self.current().token {
            Token::Identifier(n) => n.clone(),
            _ => {
                return Err(TurkceKodError::syntax_error(
                    self.current().line,
                    "Tensor adı bekleniyor",
                ));
            }
        };
        self.advance();
        
        self.consume(Token::Equals, "'=' bekleniyor")?;
        
        // Check for random tensor: matris a = rastgele [100, 100]
        if self.check(&Token::Rastgele) {
            self.advance(); // consume 'rastgele'
            let shape = self.parse_shape()?;
            return Ok(Statement::RandomTensor { name, shape });
        }
        
        // Check for GPU operation result: matris c = gpu_çarp a b
        match &self.current().token {
            Token::GpuCarp | Token::GpuTopla | Token::GpuCikar | Token::GpuBol |
            Token::GpuRelu | Token::GpuGelu | Token::GpuSigmoid | Token::GpuSilu |
            Token::GpuTanh | Token::GpuSoftmax | Token::GpuLayernorm | Token::GpuRmsnorm |
            Token::GpuTranspoz => {
                let op = match &self.current().token {
                    Token::GpuCarp => GpuOperation::MatMul,
                    Token::GpuTopla => GpuOperation::Add,
                    Token::GpuCikar => GpuOperation::Sub,
                    Token::GpuBol => GpuOperation::Div,
                    Token::GpuRelu => GpuOperation::Relu,
                    Token::GpuGelu => GpuOperation::Gelu,
                    Token::GpuSigmoid => GpuOperation::Sigmoid,
                    Token::GpuSilu => GpuOperation::Silu,
                    Token::GpuTanh => GpuOperation::Tanh,
                    Token::GpuSoftmax => GpuOperation::Softmax,
                    Token::GpuLayernorm => GpuOperation::LayerNorm,
                    Token::GpuRmsnorm => GpuOperation::RmsNorm,
                    Token::GpuTranspoz => GpuOperation::Transpose,
                    _ => unreachable!(),
                };
                self.advance(); // consume gpu op
                
                // Parse operand names
                let mut args = Vec::new();
                while let Token::Identifier(arg_name) = &self.current().token {
                    args.push(arg_name.clone());
                    self.advance();
                }
                
                return Ok(Statement::GpuOp { result: name, op, args });
            }
            _ => {}
        }
        
        // Regular tensor: matris a = [1,2,3] boyut [3]
        self.consume(Token::OpenBracket, "'[' bekleniyor")?;
        
        let mut data = Vec::new();
        while !self.check(&Token::CloseBracket) {
            data.push(self.parse_expression()?);
            if self.check(&Token::Comma) {
                self.advance();
            }
        }
        self.consume(Token::CloseBracket, "']' bekleniyor")?;
        
        // Parse shape: boyut [dim1, dim2, ...]
        let shape = if self.check(&Token::Boyut) {
            self.advance();
            self.parse_shape()?
        } else {
            vec![data.len()] // Default to 1D
        };
        
        Ok(Statement::TensorDecl { name, data, shape })
    }
    
    /// Parse shape: [dim1, dim2, ...]
    fn parse_shape(&mut self) -> TurkceKodResult<Vec<usize>> {
        self.consume(Token::OpenBracket, "'[' bekleniyor")?;
        
        let mut shape = Vec::new();
        while !self.check(&Token::CloseBracket) {
            if let Token::Integer(n) = &self.current().token {
                shape.push(*n as usize);
                self.advance();
            } else {
                return Err(TurkceKodError::syntax_error(
                    self.current().line,
                    "Boyut için sayı bekleniyor",
                ));
            }
            if self.check(&Token::Comma) {
                self.advance();
            }
        }
        self.consume(Token::CloseBracket, "']' bekleniyor")?;
        
        Ok(shape)
    }
    
    /// Parse GPU mode: gpu_mod "hybrid"
    fn parse_gpu_mode(&mut self) -> TurkceKodResult<Statement> {
        self.advance(); // consume 'gpu_mod'
        
        if let Token::String(mode) = &self.current().token {
            let mode = mode.clone();
            self.advance();
            Ok(Statement::GpuMode(mode))
        } else {
            Err(TurkceKodError::syntax_error(
                self.current().line,
                "GPU modu bekleniyor: \"cpu\", \"gpu\", veya \"hybrid\"",
            ))
        }
    }
    
    /// Parse binary GPU operation: gpu_çarp a b
    fn parse_gpu_op(&mut self, op: GpuOperation) -> TurkceKodResult<Statement> {
        self.advance(); // consume op token
        
        let mut args = Vec::new();
        while let Token::Identifier(arg_name) = &self.current().token {
            args.push(arg_name.clone());
            self.advance();
        }
        
        // For standalone gpu ops, we need a result variable
        // This handles: gpu_çarp a b (stores result to last arg)
        if args.len() >= 2 {
            let result = args.remove(0);
            Ok(Statement::GpuOp { result, op, args })
        } else {
            Err(TurkceKodError::syntax_error(
                self.current().line,
                "GPU işlemi için en az 2 argüman gerekli",
            ))
        }
    }
    
    /// Parse unary GPU operation: gpu_relu a
    fn parse_gpu_unary_op(&mut self, op: GpuOperation) -> TurkceKodResult<Statement> {
        self.advance(); // consume op token
        
        if let Token::Identifier(result) = &self.current().token {
            let result = result.clone();
            self.advance();
            
            if let Token::Identifier(arg) = &self.current().token {
                let arg = arg.clone();
                self.advance();
                Ok(Statement::GpuOp { result, op, args: vec![arg] })
            } else {
                // In-place operation: gpu_relu a -> a = relu(a)
                Ok(Statement::GpuOp { result: result.clone(), op, args: vec![result] })
            }
        } else {
            Err(TurkceKodError::syntax_error(
                self.current().line,
                "Tensor adı bekleniyor",
            ))
        }
    }
    
    // -------------------------------------------------------------------------
    // AI Training Parsing Functions
    // -------------------------------------------------------------------------
    
    /// Parse loss function: capraz_entropi result predicted target
    fn parse_loss(&mut self, loss_type: LossType) -> TurkceKodResult<Statement> {
        self.advance(); // consume loss token
        
        let result = self.expect_identifier("Kayıp değişken adı bekleniyor")?;
        let predicted = self.expect_identifier("Tahmin tensor adı bekleniyor")?;
        let target = self.expect_identifier("Hedef tensor adı bekleniyor")?;
        
        Ok(Statement::Loss { result, loss_type, predicted, target })
    }
    
    /// Parse backpropagation: geri_yayilim loss_var
    fn parse_backprop(&mut self) -> TurkceKodResult<Statement> {
        self.advance(); // consume geri_yayilim
        
        let loss_var = self.expect_identifier("Kayıp değişken adı bekleniyor")?;
        
        Ok(Statement::Backprop { loss_var })
    }
    
    /// Parse optimizer declaration: optimizer name "type" learning_rate
    fn parse_optimizer_decl(&mut self) -> TurkceKodResult<Statement> {
        self.advance(); // consume optimizer
        
        let name = self.expect_identifier("Optimizer adı bekleniyor")?;
        
        // Parse optimizer type string
        let opt_type = match &self.current().token {
            Token::String(s) => {
                let t = match s.to_lowercase().as_str() {
                    "sgd" => OptimizerType::SGD,
                    "adam" => OptimizerType::Adam,
                    _ => return Err(TurkceKodError::syntax_error(
                        self.current().line,
                        "Geçersiz optimizer türü (sgd veya adam olmalı)",
                    )),
                };
                self.advance();
                t
            }
            _ => return Err(TurkceKodError::syntax_error(
                self.current().line,
                "Optimizer türü bekleniyor (\"sgd\" veya \"adam\")",
            )),
        };
        
        // Parse learning rate
        let learning_rate = match &self.current().token {
            Token::Float(f) => { let v = *f; self.advance(); v }
            Token::Integer(i) => { let v = *i as f64; self.advance(); v }
            _ => 0.01, // Default learning rate
        };
        
        Ok(Statement::OptimizerDecl { name, opt_type, learning_rate })
    }
    
    /// Parse optimizer step: adim optimizer_name
    fn parse_optimizer_step(&mut self) -> TurkceKodResult<Statement> {
        self.advance(); // consume adim
        
        let optimizer = self.expect_identifier("Optimizer adı bekleniyor")?;
        
        Ok(Statement::OptimizerStep { optimizer })
    }
    
    /// Parse training loop: egit epochs batch_size { ... }
    fn parse_train(&mut self) -> TurkceKodResult<Statement> {
        self.advance(); // consume egit
        
        let epochs = self.parse_expression()?;
        let batch_size = self.parse_expression()?;
        
        self.consume(Token::OpenBrace, "'{' bekleniyor")?;
        let body = self.parse_block()?;
        
        Ok(Statement::Train { epochs, batch_size, body })
    }
    
    /// Parse data loading: veri_yukle var "path"
    fn parse_load_data(&mut self) -> TurkceKodResult<Statement> {
        self.advance(); // consume veri_yukle
        
        let var_name = self.expect_identifier("Değişken adı bekleniyor")?;
        
        let source = match &self.current().token {
            Token::String(s) => {
                let path = s.clone();
                self.advance();
                DataSource::File(path)
            }
            _ => DataSource::File(String::new()),
        };
        
        Ok(Statement::LoadData { var_name, source })
    }
    
    /// Parse CIFAR-10 loading: cifar_yukle var
    fn parse_cifar_load(&mut self) -> TurkceKodResult<Statement> {
        self.advance(); // consume cifar_yukle
        
        let var_name = self.expect_identifier("Değişken adı bekleniyor")?;
        
        Ok(Statement::LoadData { var_name, source: DataSource::Cifar10 })
    }
    
    /// Parse console read: konsol_oku var
    fn parse_console_read(&mut self) -> TurkceKodResult<Statement> {
        self.advance(); // consume konsol_oku
        
        let var_name = self.expect_identifier("Değişken adı bekleniyor")?;
        
        Ok(Statement::ConsoleRead { var_name })
    }
    
    /// Parse GUI data: gui_veri [widget_id] var
    /// New syntax: gui_veri sayi_girdi x (reads from widget "sayi_girdi" into var "x")
    /// Old syntax: gui_veri x (reads from global gui_input_text)
    fn parse_gui_data(&mut self) -> TurkceKodResult<Statement> {
        self.advance(); // consume gui_veri
        
        let first_id = self.expect_identifier("Değişken adı bekleniyor")?;
        
        // Check if there's a second identifier (new syntax with widget_id)
        if let Token::Identifier(second_id) = &self.current().token {
            let var_name = second_id.clone();
            self.advance();
            Ok(Statement::GuiData { widget_id: Some(first_id), var_name })
        } else {
            // Old syntax: just variable name
            Ok(Statement::GuiData { widget_id: None, var_name: first_id })
        }
    }
    
    /// Parse standalone random tensor: rastgele name [shape]
    fn parse_random_tensor(&mut self) -> TurkceKodResult<Statement> {
        self.advance(); // consume rastgele
        
        let name = self.expect_identifier("Tensor adı bekleniyor")?;
        let shape = self.parse_shape()?;
        
        Ok(Statement::RandomTensor { name, shape })
    }
    
    // ============================================
    // File System Parsing
    // ============================================
    
    fn parse_create_folder(&mut self) -> TurkceKodResult<Statement> {
        self.advance(); // consume klasor_olustur
        
        let path = if let Token::String(s) = &self.current().token {
            s.clone()
        } else {
             return Err(TurkceKodError::syntax_error(self.current().line, "Klasör yolu (string) bekleniyor"));
        };
        self.advance();
        
        Ok(Statement::CreateFolder { path })
    }

    // ============================================
    // GUI Parsing Functions
    // ============================================
    
    /// Parse widget declaration: type id "text" { children }
    fn parse_gui_decl(&mut self, widget_type: GuiWidgetType) -> TurkceKodResult<Statement> {
        self.advance(); // consume type keyword
        
        let id = self.expect_identifier("Widget kimliği (id) bekleniyor")?;
        
        // Optional text/title
        let text = if let Token::String(s) = &self.current().token {
            let t = s.clone();
            self.advance();
            t
        } else {
            String::new()
        };
        
        // Optional block for children or callbacks
        let commands = if self.check(&Token::OpenBrace) {
            self.consume(Token::OpenBrace, "'{' bekleniyor")?;
            self.parse_block()?
        } else {
            Vec::new()
        };
        
        Ok(Statement::GuiDecl { widget_type, id, text, commands })
    }
    
    /// Parse widget update: degistir id "property" value
    fn parse_gui_update(&mut self) -> TurkceKodResult<Statement> {
        self.advance(); // consume degistir
        
        let id = self.expect_identifier("Widget kimliği (id) bekleniyor")?;
        
        // Property name usually a keyword or identifier, treat as string?
        // Let's expect identifier for property name like 'baslik', 'metin'
        let property = self.expect_identifier("Özellik adı bekleniyor (örn: metin)")?;
        
        let value = self.parse_expression()?;
        
        Ok(Statement::GuiUpdate { id, property, value })
    }
}

