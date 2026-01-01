//! Tharvexal Explorer - Built-in File Browser
//!
//! TharvexalOS için gömülü dosya gezgini.
//! Native file dialogs (rfd) kiosk modunda çalışmadığı için bu modül kullanılır.

use std::path::{Path, PathBuf};
use std::fs;
use eframe::egui;

/// Dosya sistemi girişi
#[derive(Clone, Debug)]
pub struct FileEntry {
    pub name: String,
    pub path: PathBuf,
    pub is_dir: bool,
    pub size: u64,
}

impl FileEntry {
    fn from_path(path: &Path) -> Option<Self> {
        let metadata = fs::metadata(path).ok()?;
        let name = path.file_name()?.to_string_lossy().to_string();
        
        Some(Self {
            name,
            path: path.to_path_buf(),
            is_dir: metadata.is_dir(),
            size: metadata.len(),
        })
    }
}

/// Explorer durumu
pub struct TharvexalExplorer {
    /// Mevcut dizin
    current_dir: PathBuf,
    /// Dizin içeriği
    entries: Vec<FileEntry>,
    /// Seçili dosya
    selected: Option<PathBuf>,
    /// Hata mesajı
    error: Option<String>,
    /// Explorer görünür mü?
    pub visible: bool,
    /// Dosya seçildi callback için
    pub pending_file: Option<PathBuf>,
    /// Kaydetme modu mu?
    pub save_mode: bool,
    /// Kaydetme için dosya adı
    pub save_filename: String,
    /// Filtre (uzantılar)
    pub filter: Vec<String>,
}

impl Default for TharvexalExplorer {
    fn default() -> Self {
        Self::new()
    }
}

impl TharvexalExplorer {
    pub fn new() -> Self {
        let current_dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/"));
        let mut explorer = Self {
            current_dir: current_dir.clone(),
            entries: Vec::new(),
            selected: None,
            error: None,
            visible: false,
            pending_file: None,
            save_mode: false,
            save_filename: String::new(),
            filter: vec!["turkcekod".to_string(), "tk".to_string()],
        };
        explorer.refresh();
        explorer
    }
    
    /// Dosya aç dialog göster
    pub fn show_open(&mut self) {
        self.visible = true;
        self.save_mode = false;
        self.pending_file = None;
        self.refresh();
    }
    
    /// Dosya kaydet dialog göster
    pub fn show_save(&mut self, default_name: &str) {
        self.visible = true;
        self.save_mode = true;
        self.save_filename = default_name.to_string();
        self.pending_file = None;
        self.refresh();
    }
    
    /// Dizin içeriğini yenile
    pub fn refresh(&mut self) {
        self.entries.clear();
        self.error = None;
        
        match fs::read_dir(&self.current_dir) {
            Ok(read_dir) => {
                let mut entries: Vec<FileEntry> = read_dir
                    .filter_map(|entry| entry.ok())
                    .filter_map(|entry| FileEntry::from_path(&entry.path()))
                    .filter(|entry| {
                        // Gizli dosyaları gösterme
                        !entry.name.starts_with('.')
                    })
                    .filter(|entry| {
                        // Filtre uygula
                        if entry.is_dir {
                            true
                        } else if self.filter.is_empty() {
                            true
                        } else {
                            entry.path.extension()
                                .map(|ext| self.filter.iter().any(|f| ext == f.as_str()))
                                .unwrap_or(false)
                        }
                    })
                    .collect();
                
                // Sırala: önce klasörler, sonra dosyalar, alfabetik
                entries.sort_by(|a, b| {
                    match (a.is_dir, b.is_dir) {
                        (true, false) => std::cmp::Ordering::Less,
                        (false, true) => std::cmp::Ordering::Greater,
                        _ => a.name.to_lowercase().cmp(&b.name.to_lowercase()),
                    }
                });
                
                self.entries = entries;
            }
            Err(e) => {
                self.error = Some(format!("Dizin okunamadı: {}", e));
            }
        }
    }
    
    /// Üst dizine git
    pub fn go_up(&mut self) {
        if let Some(parent) = self.current_dir.parent() {
            self.current_dir = parent.to_path_buf();
            self.refresh();
        }
    }
    
    /// Belirtilen dizine git
    pub fn navigate_to(&mut self, path: &Path) {
        if path.is_dir() {
            self.current_dir = path.to_path_buf();
            self.refresh();
        }
    }
    
    /// Dosya seç ve kapat
    fn select_file(&mut self, path: PathBuf) {
        self.pending_file = Some(path);
        self.visible = false;
    }
    
    /// UI çiz
    pub fn show(&mut self, ctx: &egui::Context) {
        if !self.visible {
            return;
        }
        
        let title = if self.save_mode { "📁 Dosya Kaydet" } else { "📂 Dosya Aç" };
        
        egui::Window::new(title)
            .collapsible(false)
            .resizable(true)
            .default_size([500.0, 400.0])
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                // Üst araç çubuğu
                ui.horizontal(|ui| {
                    if ui.button("⬆ Üst").clicked() {
                        self.go_up();
                    }
                    
                    if ui.button("🔄").clicked() {
                        self.refresh();
                    }
                    
                    ui.separator();
                    
                    // Mevcut yol
                    ui.label(format!("📍 {}", self.current_dir.display()));
                });
                
                ui.separator();
                
                // Hata mesajı
                if let Some(err) = &self.error {
                    ui.colored_label(egui::Color32::RED, err);
                    ui.separator();
                }
                
                // Dosya listesi
                egui::ScrollArea::vertical()
                    .max_height(280.0)
                    .show(ui, |ui| {
                        let entries = self.entries.clone();
                        
                        for entry in entries {
                            let icon = if entry.is_dir { "📁" } else { "📄" };
                            let label = format!("{} {}", icon, entry.name);
                            
                            let is_selected = self.selected.as_ref() == Some(&entry.path);
                            
                            let response = ui.selectable_label(is_selected, &label);
                            
                            if response.clicked() {
                                self.selected = Some(entry.path.clone());
                            }
                            
                            if response.double_clicked() {
                                if entry.is_dir {
                                    self.navigate_to(&entry.path);
                                } else if !self.save_mode {
                                    self.select_file(entry.path);
                                }
                            }
                        }
                    });
                
                ui.separator();
                
                // Kaydetme modu: dosya adı girişi
                if self.save_mode {
                    ui.horizontal(|ui| {
                        ui.label("Dosya adı:");
                        ui.text_edit_singleline(&mut self.save_filename);
                    });
                }
                
                // Alt düğmeler
                ui.horizontal(|ui| {
                    if self.save_mode {
                        if ui.button("💾 Kaydet").clicked() {
                            let mut filename = self.save_filename.clone();
                            if !filename.ends_with(".turkcekod") && !filename.ends_with(".tk") {
                                filename.push_str(".turkcekod");
                            }
                            let path = self.current_dir.join(&filename);
                            self.select_file(path);
                        }
                    } else {
                        let can_open = self.selected.as_ref().map(|p| !p.is_dir()).unwrap_or(false);
                        if ui.add_enabled(can_open, egui::Button::new("📂 Aç")).clicked() {
                            if let Some(path) = self.selected.clone() {
                                self.select_file(path);
                            }
                        }
                    }
                    
                    if ui.button("❌ İptal").clicked() {
                        self.visible = false;
                        self.pending_file = None;
                    }
                });
            });
    }
    
    /// Bekleyen dosyayı al ve sıfırla
    pub fn take_pending_file(&mut self) -> Option<PathBuf> {
        self.pending_file.take()
    }
}

/// Home dizinini bul
pub fn get_home_dir() -> PathBuf {
    #[cfg(target_os = "linux")]
    {
        std::env::var("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("/root"))
    }
    
    #[cfg(target_os = "windows")]
    {
        std::env::var("USERPROFILE")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("C:\\"))
    }
    
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        PathBuf::from("/")
    }
}

/// TharvexalOS varsayılan çalışma dizini
pub fn get_tharvexal_workspace() -> PathBuf {
    let home = get_home_dir();
    let workspace = home.join("TharvexalOS").join("projeler");
    
    // Yoksa oluştur
    if !workspace.exists() {
        let _ = fs::create_dir_all(&workspace);
    }
    
    workspace
}
