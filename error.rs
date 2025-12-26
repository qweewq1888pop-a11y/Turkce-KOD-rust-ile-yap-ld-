//! Türkçe Kod - Error Types
//! 
//! All error messages are in Turkish for better user experience.

use thiserror::Error;

/// Main error type for the Türkçe Kod interpreter
#[derive(Error, Debug, Clone)]
pub enum TurkceKodError {
    // Lexer errors
    #[error("Sözdizimi hatası (satır {line}): {message}")]
    SozdizimHatasi { line: usize, message: String },

    #[error("Beklenmeyen karakter (satır {line}): '{character}'")]
    BeklenmeyenKarakter { line: usize, character: char },

    #[error("Tamamlanmamış metin (satır {line}): Kapanış tırnağı eksik")]
    TamamlanmamisMetin { line: usize },

    // Parser errors
    #[error("Beklenmeyen simge (satır {line}): '{found}' bekleniyordu: {expected}")]
    BeklenmeyenSimge {
        line: usize,
        expected: String,
        found: String,
    },

    #[error("Eksik blok (satır {line}): '{{' bekleniyor")]
    EksikBlok { line: usize },

    #[error("Kapanmamış blok (satır {line}): '}}' bekleniyor")]
    KapanmamisBlok { line: usize },

    // Runtime errors
    #[error("Tanımlanmayan değişken: '{name}'")]
    TanimlanmayanDegisken { name: String },

    #[error("Tanımlanmayan fonksiyon: '{name}'")]
    TanimlanmayanFonksiyon { name: String },

    #[error("Tanımlanmayan liste: '{name}'")]
    TanimlanmayanListe { name: String },

    #[error("Tip hatası: '{expected}' bekleniyordu, '{got}' bulundu")]
    TipHatasi { expected: String, got: String },

    #[error("Dizin hatası: '{index}' dizini geçersiz (liste uzunluğu: {length})")]
    DizinHatasi { index: i64, length: usize },

    #[error("Sıfıra bölme hatası")]
    SifiraBolme,

    #[error("Aritmetik hata: {message}")]
    AritmetikHata { message: String },

    #[error("Parametre sayısı uyumsuz: {expected} bekleniyordu, {got} verildi")]
    ParametreSayisi { expected: usize, got: usize },

    // File errors
    #[error("Dosya bulunamadı: '{path}'")]
    DosyaBulunamadi { path: String },

    #[error("Dosya okuma hatası: '{path}' - {message}")]
    DosyaOkumaHatasi { path: String, message: String },

    #[error("Dosya yazma hatası: '{path}' - {message}")]
    DosyaYazmaHatasi { path: String, message: String },

    // Module errors
    #[error("Modül bulunamadı: '{name}'")]
    ModulBulunamadi { name: String },

    #[error("Modül fonksiyonu bulunamadı: '{module}.{function}'")]
    ModulFonksiyonuBulunamadi { module: String, function: String },

    // General errors
    #[error("Sonsuz döngü önlendi (maksimum {max_iterations} iterasyon)")]
    SonsuzDongu { max_iterations: usize },

    #[error("Bilinmeyen komut: '{command}'")]
    BilinmeyenKomut { command: String },
    
    // GPU/Tensor errors
    #[error("GPU hatası: {message}")]
    GpuHatasi { message: String },
}

/// Result type alias for Türkçe Kod operations
pub type TurkceKodResult<T> = Result<T, TurkceKodError>;

impl TurkceKodError {
    /// Create a syntax error
    pub fn syntax_error(line: usize, message: impl Into<String>) -> Self {
        TurkceKodError::SozdizimHatasi {
            line,
            message: message.into(),
        }
    }

    /// Create an undefined variable error
    pub fn undefined_var(name: impl Into<String>) -> Self {
        TurkceKodError::TanimlanmayanDegisken { name: name.into() }
    }

    /// Create an undefined function error
    pub fn undefined_func(name: impl Into<String>) -> Self {
        TurkceKodError::TanimlanmayanFonksiyon { name: name.into() }
    }

    /// Create a type error
    pub fn type_error(expected: impl Into<String>, got: impl Into<String>) -> Self {
        TurkceKodError::TipHatasi {
            expected: expected.into(),
            got: got.into(),
        }
    }

    /// Create an index out of bounds error
    pub fn index_error(index: i64, length: usize) -> Self {
        TurkceKodError::DizinHatasi { index, length }
    }
    
    /// Get error code (for debugging reference)
    pub fn error_code(&self) -> &'static str {
        match self {
            TurkceKodError::SozdizimHatasi { .. } => "E001",
            TurkceKodError::BeklenmeyenKarakter { .. } => "E002",
            TurkceKodError::TamamlanmamisMetin { .. } => "E003",
            TurkceKodError::BeklenmeyenSimge { .. } => "E010",
            TurkceKodError::EksikBlok { .. } => "E011",
            TurkceKodError::KapanmamisBlok { .. } => "E012",
            TurkceKodError::TanimlanmayanDegisken { .. } => "E020",
            TurkceKodError::TanimlanmayanFonksiyon { .. } => "E021",
            TurkceKodError::TanimlanmayanListe { .. } => "E022",
            TurkceKodError::TipHatasi { .. } => "E030",
            TurkceKodError::DizinHatasi { .. } => "E031",
            TurkceKodError::SifiraBolme => "E040",
            TurkceKodError::AritmetikHata { .. } => "E041",
            TurkceKodError::ParametreSayisi { .. } => "E050",
            TurkceKodError::DosyaBulunamadi { .. } => "E060",
            TurkceKodError::DosyaOkumaHatasi { .. } => "E061",
            TurkceKodError::DosyaYazmaHatasi { .. } => "E062",
            TurkceKodError::ModulBulunamadi { .. } => "E070",
            TurkceKodError::ModulFonksiyonuBulunamadi { .. } => "E071",
            TurkceKodError::SonsuzDongu { .. } => "E080",
            TurkceKodError::BilinmeyenKomut { .. } => "E090",
            TurkceKodError::GpuHatasi { .. } => "E100",
        }
    }
    
    /// Check if this error is fatal (cannot continue execution)
    pub fn is_fatal(&self) -> bool {
        matches!(self, 
            TurkceKodError::SonsuzDongu { .. } |
            TurkceKodError::GpuHatasi { .. } |
            TurkceKodError::DosyaBulunamadi { .. }
        )
    }
}
