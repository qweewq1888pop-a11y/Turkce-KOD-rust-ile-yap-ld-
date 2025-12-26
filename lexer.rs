//! Türkçe Kod - Lexer (Tokenizer)
//! 
//! Converts Turkish source code into tokens for parsing.

use crate::error::{TurkceKodError, TurkceKodResult};
use std::collections::HashMap;
use std::sync::OnceLock;

/// Token types for the Türkçe Kod language
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Keywords - Variable declarations
    Sayiv,      // sayıv - integer variable
    Metinv,     // metinv - string variable
    Mantiksalv, // mantıksalv - boolean variable
    Listav,     // listav - list variable

    // Keywords - I/O
    Yaz,       // yaz - print
    Oku,       // oku - read input
    OkuInt,    // oku_int - read integer
    OkuFloat,  // oku_float - read float

    // Keywords - Control flow
    Eger,     // eğer - if
    Degilse,  // değilse - else
    Tekrar,   // tekrar - repeat
    Iken,     // iken - while
    HerElemanIcin, // her_eleman_için - for each

    // Keywords - Functions
    Islev, // işlev - function
    Don,   // dön - return

    // Keywords - List operations
    Ekle,    // ekle - add to list
    Uzunluk, // uzunluk - length
    Al,      // al - get from list

    // Keywords - File operations
    OkuDosya, // oku_dosya - read file
    YazDosya, // yaz_dosya - write file

    // Keywords - Math
    Hesapla,  // hesapla - calculate
    Karekok,  // karekok - sqrt
    Sinus,    // sinus - sin
    Cosinus,  // cosinus - cos
    Tanjant,  // tanjant - tan
    Pi,       // pi

    // Keywords - Time
    Bekle, // bekle - sleep

    // Keywords - Tensors
    Matris,      // matris - tensor/matrix declaration
    Boyut,       // boyut - shape specification
    Rastgele,    // rastgele - random tensor

    // Keywords - GPU operations
    GpuBilgi,    // gpu_bilgi - GPU info
    GpuMod,      // gpu_mod - execution mode
    GpuCarp,     // gpu_çarp - GPU matrix multiply
    GpuTopla,    // gpu_topla - GPU add
    GpuCikar,    // gpu_çıkar - GPU subtract
    GpuBol,      // gpu_böl - GPU divide
    GpuRelu,     // gpu_relu
    GpuGelu,     // gpu_gelu
    GpuSigmoid,  // gpu_sigmoid
    GpuSilu,     // gpu_silu
    GpuTanh,     // gpu_tanh
    GpuSoftmax,  // gpu_softmax
    GpuLayernorm,// gpu_layernorm
    GpuRmsnorm,  // gpu_rmsnorm
    GpuTranspoz, // gpu_transpoz

    // Keywords - AI Training
    Kayip,           // kayip - loss value
    CaprazEntropi,   // capraz_entropi - cross entropy loss
    MseKayip,        // mse_kayip - MSE loss
    GeriYayilim,     // geri_yayilim - backpropagation
    Gradyan,         // gradyan - gradient
    OptimizerTanim,  // optimizer - define optimizer
    Adim,            // adim - optimizer step
    OgrenmeOrani,    // ogrenme_orani - learning rate
    Egit,            // egit - training loop
    VeriYukle,       // veri_yukle - load data
    KonsolOku,       // konsol_oku - read from console
    GuiVeri,         // gui_veri - get GUI input data
    SifirGradyan,    // sifir_gradyan - zero gradients

    // Keywords - GUI Library (TürkçeInter)
    Pencere,         // pencere - window
    Buton,           // buton - button
    Etiket,          // etiket - label
    Girdi,           // girdi - input
    Degistir,        // degistir - update widget property
    Yerlesim,        // yerlesim - layout configuration
    Sohbet,          // sohbet - text chat widget
    
    // File System
    KlasorOlustur,      // klasor_olustur
    AiDizinleriOlustur, // ai_dizinleri_olustur

    // Literals
    Integer(i64),
    Float(f64),
    String(String),
    Boolean(bool),
    Identifier(String),

    // Operators
    Plus,        // +
    Minus,       // -
    Star,        // *
    Slash,       // /
    Percent,     // %
    Equals,      // =
    EqualsEquals, // ==
    NotEquals,   // !=
    Greater,     // >
    Less,        // <
    GreaterEq,   // >=
    LessEq,      // <=
    And,         // ve
    Or,          // veya

    // Turkish comparison operators (aliases)
    Buyukse,  // büyükse - greater than
    Kucukse,  // küçükse - less than
    Esitse,   // eşitse - equals

    // Delimiters
    OpenParen,   // (
    CloseParen,  // )
    OpenBrace,   // {
    CloseBrace,  // }
    OpenBracket, // [
    CloseBracket, // ]
    Comma,       // ,
    Dot,         // .

    // Special
    Newline,
    Eof,
}

/// Lazy-initialized keyword map for O(1) lookup
/// Replaces the linear match statement with ~80 patterns
static KEYWORDS: OnceLock<HashMap<&'static str, Token>> = OnceLock::new();

fn get_keywords() -> &'static HashMap<&'static str, Token> {
    KEYWORDS.get_or_init(|| {
        let mut m = HashMap::with_capacity(80);
        
        // Variable declarations
        m.insert("sayıv", Token::Sayiv);
        m.insert("sayiv", Token::Sayiv);
        m.insert("metinv", Token::Metinv);
        m.insert("mantıksalv", Token::Mantiksalv);
        m.insert("mantiksalv", Token::Mantiksalv);
        m.insert("listav", Token::Listav);
        
        // I/O
        m.insert("yaz", Token::Yaz);
        m.insert("oku", Token::Oku);
        m.insert("oku_int", Token::OkuInt);
        m.insert("oku_float", Token::OkuFloat);
        
        // Control flow
        m.insert("eğer", Token::Eger);
        m.insert("eger", Token::Eger);
        m.insert("değilse", Token::Degilse);
        m.insert("degilse", Token::Degilse);
        m.insert("tekrar", Token::Tekrar);
        m.insert("iken", Token::Iken);
        m.insert("her_eleman_için", Token::HerElemanIcin);
        m.insert("her_eleman_icin", Token::HerElemanIcin);
        
        // Functions
        m.insert("işlev", Token::Islev);
        m.insert("islev", Token::Islev);
        m.insert("dön", Token::Don);
        m.insert("don", Token::Don);
        
        // List operations
        m.insert("ekle", Token::Ekle);
        m.insert("uzunluk", Token::Uzunluk);
        m.insert("al", Token::Al);
        
        // File operations
        m.insert("oku_dosya", Token::OkuDosya);
        m.insert("yaz_dosya", Token::YazDosya);
        
        // Math
        m.insert("hesapla", Token::Hesapla);
        m.insert("karekok", Token::Karekok);
        m.insert("sinus", Token::Sinus);
        m.insert("cosinus", Token::Cosinus);
        m.insert("tanjant", Token::Tanjant);
        m.insert("pi", Token::Pi);
        
        // Time
        m.insert("bekle", Token::Bekle);
        
        // Tensors
        m.insert("matris", Token::Matris);
        m.insert("boyut", Token::Boyut);
        m.insert("rastgele", Token::Rastgele);
        
        // GPU operations
        m.insert("gpu_bilgi", Token::GpuBilgi);
        m.insert("gpu_mod", Token::GpuMod);
        m.insert("gpu_çarp", Token::GpuCarp);
        m.insert("gpu_carp", Token::GpuCarp);
        m.insert("gpu_topla", Token::GpuTopla);
        m.insert("gpu_çıkar", Token::GpuCikar);
        m.insert("gpu_cikar", Token::GpuCikar);
        m.insert("gpu_böl", Token::GpuBol);
        m.insert("gpu_bol", Token::GpuBol);
        m.insert("gpu_relu", Token::GpuRelu);
        m.insert("gpu_gelu", Token::GpuGelu);
        m.insert("gpu_sigmoid", Token::GpuSigmoid);
        m.insert("gpu_silu", Token::GpuSilu);
        m.insert("gpu_tanh", Token::GpuTanh);
        m.insert("gpu_softmax", Token::GpuSoftmax);
        m.insert("gpu_layernorm", Token::GpuLayernorm);
        m.insert("gpu_rmsnorm", Token::GpuRmsnorm);
        m.insert("gpu_transpoz", Token::GpuTranspoz);
        
        // AI Training keywords
        // kayip reserved keyword removed to allow variable name 'kayip'
        // m.insert("kayip", Token::Kayip);
        // m.insert("kayıp", Token::Kayip);
        m.insert("capraz_entropi", Token::CaprazEntropi);
        m.insert("çapraz_entropi", Token::CaprazEntropi);
        m.insert("mse_kayip", Token::MseKayip);
        m.insert("mse_kayıp", Token::MseKayip);
        m.insert("geri_yayilim", Token::GeriYayilim);
        m.insert("geri_yayılım", Token::GeriYayilim);
        m.insert("gradyan", Token::Gradyan);
        m.insert("optimizer", Token::OptimizerTanim);
        m.insert("adim", Token::Adim);
        m.insert("adım", Token::Adim);
        m.insert("ogrenme_orani", Token::OgrenmeOrani);
        m.insert("öğrenme_oranı", Token::OgrenmeOrani);
        m.insert("egit", Token::Egit);
        m.insert("eğit", Token::Egit);
        m.insert("veri_yukle", Token::VeriYukle);
        m.insert("veri_yükle", Token::VeriYukle);
        m.insert("konsol_oku", Token::KonsolOku);
        m.insert("gui_veri", Token::GuiVeri);
        m.insert("sifir_gradyan", Token::SifirGradyan);
        m.insert("sıfır_gradyan", Token::SifirGradyan);
        
        // GUI Library keywords
        m.insert("pencere", Token::Pencere);
        m.insert("buton", Token::Buton);
        m.insert("etiket", Token::Etiket);
        m.insert("girdi", Token::Girdi);
        m.insert("yerlesim", Token::Yerlesim);
        m.insert("degistir", Token::Degistir);
        m.insert("değiştir", Token::Degistir);
        m.insert("sohbet", Token::Sohbet);
        
        // File System
        m.insert("klasor_olustur", Token::KlasorOlustur);
        m.insert("klasör_oluştur", Token::KlasorOlustur);
        m.insert("ai_dizinleri_olustur", Token::AiDizinleriOlustur);
        m.insert("ai_dizinleri_oluştur", Token::AiDizinleriOlustur);
        
        // Boolean literals
        m.insert("doğru", Token::Boolean(true));
        m.insert("dogru", Token::Boolean(true));
        m.insert("yanlış", Token::Boolean(false));
        m.insert("yanlis", Token::Boolean(false));
        
        // Logical operators
        m.insert("ve", Token::And);
        m.insert("veya", Token::Or);
        
        // Turkish comparison aliases
        m.insert("büyükse", Token::Buyukse);
        m.insert("buyukse", Token::Buyukse);
        m.insert("küçükse", Token::Kucukse);
        m.insert("kucukse", Token::Kucukse);
        m.insert("eşitse", Token::Esitse);
        m.insert("esitse", Token::Esitse);
        
        m
    })
}

/// A token with its position information
#[derive(Debug, Clone)]
pub struct TokenInfo {
    pub token: Token,
    pub line: usize,
    pub column: usize,
}

/// Lexer for Türkçe Kod
pub struct Lexer {
    source: Vec<char>,
    pos: usize,
    line: usize,
    column: usize,
}

impl Lexer {
    /// Create a new lexer from source code
    pub fn new(source: &str) -> Self {
        Self {
            source: source.chars().collect(),
            pos: 0,
            line: 1,
            column: 1,
        }
    }

    /// Tokenize the entire source code
    pub fn tokenize(&mut self) -> TurkceKodResult<Vec<TokenInfo>> {
        let mut tokens = Vec::new();

        while !self.is_at_end() {
            self.skip_whitespace();
            if self.is_at_end() {
                break;
            }

            let token_info = self.scan_token()?;
            tokens.push(token_info);
        }

        tokens.push(TokenInfo {
            token: Token::Eof,
            line: self.line,
            column: self.column,
        });

        Ok(tokens)
    }

    fn is_at_end(&self) -> bool {
        self.pos >= self.source.len()
    }

    fn current(&self) -> char {
        if self.is_at_end() {
            '\0'
        } else {
            self.source[self.pos]
        }
    }

    fn peek(&self) -> char {
        if self.pos + 1 >= self.source.len() {
            '\0'
        } else {
            self.source[self.pos + 1]
        }
    }

    fn advance(&mut self) -> char {
        let c = self.current();
        self.pos += 1;
        if c == '\n' {
            self.line += 1;
            self.column = 1;
        } else {
            self.column += 1;
        }
        c
    }

    fn skip_whitespace(&mut self) {
        while !self.is_at_end() {
            match self.current() {
                ' ' | '\t' | '\r' => {
                    self.advance();
                }
                '\n' => {
                    self.advance();
                }
                '#' => {
                    // Skip comment until end of line
                    while !self.is_at_end() && self.current() != '\n' {
                        self.advance();
                    }
                }
                ';' => {
                    // Semicolon also acts as statement separator, skip it
                    self.advance();
                }
                _ => break,
            }
        }
    }

    fn scan_token(&mut self) -> TurkceKodResult<TokenInfo> {
        let start_line = self.line;
        let start_column = self.column;

        let c = self.advance();

        let token = match c {
            // Single character tokens
            '+' => Token::Plus,
            '-' => Token::Minus,
            '*' => Token::Star,
            '/' => Token::Slash,
            '%' => Token::Percent,
            '(' => Token::OpenParen,
            ')' => Token::CloseParen,
            '{' => Token::OpenBrace,
            '}' => Token::CloseBrace,
            '[' => Token::OpenBracket,
            ']' => Token::CloseBracket,
            ',' => Token::Comma,
            '.' => Token::Dot,

            // Multi-character tokens
            '=' => {
                if self.current() == '=' {
                    self.advance();
                    Token::EqualsEquals
                } else {
                    Token::Equals
                }
            }
            '!' => {
                if self.current() == '=' {
                    self.advance();
                    Token::NotEquals
                } else {
                    return Err(TurkceKodError::BeklenmeyenKarakter {
                        line: start_line,
                        character: c,
                    });
                }
            }
            '>' => {
                if self.current() == '=' {
                    self.advance();
                    Token::GreaterEq
                } else {
                    Token::Greater
                }
            }
            '<' => {
                if self.current() == '=' {
                    self.advance();
                    Token::LessEq
                } else {
                    Token::Less
                }
            }

            // String literals
            '"' => self.scan_string()?,

            // Numbers
            '0'..='9' => self.scan_number(c)?,

            // Identifiers and keywords
            c if c.is_alphabetic() || c == '_' => self.scan_identifier(c)?,

            _ => {
                return Err(TurkceKodError::BeklenmeyenKarakter {
                    line: start_line,
                    character: c,
                });
            }
        };

        Ok(TokenInfo {
            token,
            line: start_line,
            column: start_column,
        })
    }

    fn scan_string(&mut self) -> TurkceKodResult<Token> {
        let start_line = self.line;
        let mut value = String::new();

        while !self.is_at_end() && self.current() != '"' {
            if self.current() == '\n' {
                return Err(TurkceKodError::TamamlanmamisMetin { line: start_line });
            }
            if self.current() == '\\' {
                self.advance();
                match self.current() {
                    'n' => value.push('\n'),
                    't' => value.push('\t'),
                    'r' => value.push('\r'),
                    '"' => value.push('"'),
                    '\\' => value.push('\\'),
                    _ => value.push(self.current()),
                }
                self.advance();
            } else {
                value.push(self.advance());
            }
        }

        if self.is_at_end() {
            return Err(TurkceKodError::TamamlanmamisMetin { line: start_line });
        }

        self.advance(); // Consume closing quote
        Ok(Token::String(value))
    }

    fn scan_number(&mut self, first: char) -> TurkceKodResult<Token> {
        let start_line = self.line;
        let mut value = String::from(first);

        while !self.is_at_end() && self.current().is_ascii_digit() {
            value.push(self.advance());
        }

        // Check for decimal
        if self.current() == '.' && self.peek().is_ascii_digit() {
            value.push(self.advance()); // consume '.'
            while !self.is_at_end() && self.current().is_ascii_digit() {
                value.push(self.advance());
            }
            // Safe parsing with error handling
            value.parse::<f64>()
                .map(Token::Float)
                .map_err(|_| TurkceKodError::SozdizimHatasi {
                    line: start_line,
                    message: format!("Geçersiz ondalık sayı: {}", value),
                })
        } else {
            // Safe parsing with error handling
            value.parse::<i64>()
                .map(Token::Integer)
                .map_err(|_| TurkceKodError::SozdizimHatasi {
                    line: start_line,
                    message: format!("Geçersiz tam sayı: {}", value),
                })
        }
    }

    fn scan_identifier(&mut self, first: char) -> TurkceKodResult<Token> {
        let mut value = String::from(first);

        while !self.is_at_end() && (self.current().is_alphanumeric() || self.current() == '_') {
            value.push(self.advance());
        }

        // O(1) HashMap lookup instead of linear match with ~80 patterns
        let token = get_keywords()
            .get(value.as_str())
            .cloned()
            .unwrap_or_else(|| Token::Identifier(value));

        Ok(token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tokens() {
        let mut lexer = Lexer::new("sayıv x = 5");
        let tokens = lexer.tokenize().unwrap();
        
        assert!(matches!(tokens[0].token, Token::Sayiv));
        assert!(matches!(tokens[1].token, Token::Identifier(_)));
        assert!(matches!(tokens[2].token, Token::Equals));
        assert!(matches!(tokens[3].token, Token::Integer(5)));
    }

    #[test]
    fn test_string_literal() {
        let mut lexer = Lexer::new("yaz \"Merhaba Dünya\"");
        let tokens = lexer.tokenize().unwrap();
        
        assert!(matches!(tokens[0].token, Token::Yaz));
        assert!(matches!(tokens[1].token, Token::String(_)));
    }
    
    #[test]
    fn test_operators() {
        let mut lexer = Lexer::new("+ - * / == != >= <=");
        let tokens = lexer.tokenize().unwrap();
        
        assert!(matches!(tokens[0].token, Token::Plus));
        assert!(matches!(tokens[1].token, Token::Minus));
        assert!(matches!(tokens[2].token, Token::Star));
        assert!(matches!(tokens[3].token, Token::Slash));
        assert!(matches!(tokens[4].token, Token::EqualsEquals));
        assert!(matches!(tokens[5].token, Token::NotEquals));
        assert!(matches!(tokens[6].token, Token::GreaterEq));
        assert!(matches!(tokens[7].token, Token::LessEq));
    }
    
    #[test]
    fn test_float_numbers() {
        let mut lexer = Lexer::new("3.14 0.5 100.0");
        let tokens = lexer.tokenize().unwrap();
        
        assert!(matches!(tokens[0].token, Token::Float(f) if (f - 3.14).abs() < 0.001));
        assert!(matches!(tokens[1].token, Token::Float(f) if (f - 0.5).abs() < 0.001));
        assert!(matches!(tokens[2].token, Token::Float(f) if (f - 100.0).abs() < 0.001));
    }
    
    #[test]
    fn test_gpu_keywords() {
        let mut lexer = Lexer::new("gpu_carp gpu_relu gpu_softmax");
        let tokens = lexer.tokenize().unwrap();
        
        assert!(matches!(tokens[0].token, Token::GpuCarp));
        assert!(matches!(tokens[1].token, Token::GpuRelu));
        assert!(matches!(tokens[2].token, Token::GpuSoftmax));
    }
    
    #[test]
    fn test_ai_training_keywords() {
        let mut lexer = Lexer::new("egit optimizer capraz_entropi geri_yayilim");
        let tokens = lexer.tokenize().unwrap();
        
        assert!(matches!(tokens[0].token, Token::Egit));
        assert!(matches!(tokens[1].token, Token::OptimizerTanim));
        assert!(matches!(tokens[2].token, Token::CaprazEntropi));
        assert!(matches!(tokens[3].token, Token::GeriYayilim));
    }
    
    #[test]
    fn test_turkish_keywords() {
        let mut lexer = Lexer::new("eğer değilse tekrar doğru yanlış");
        let tokens = lexer.tokenize().unwrap();
        
        assert!(matches!(tokens[0].token, Token::Eger));
        assert!(matches!(tokens[1].token, Token::Degilse));
        assert!(matches!(tokens[2].token, Token::Tekrar));
        assert!(matches!(tokens[3].token, Token::Boolean(true)));
        assert!(matches!(tokens[4].token, Token::Boolean(false)));
    }
}
