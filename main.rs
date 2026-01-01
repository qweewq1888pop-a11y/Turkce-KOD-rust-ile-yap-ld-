//! Türkçe Kod - GPU-Accelerated IDE
//! 
//! A modern IDE for the Turkish programming language using egui and WebGPU.
//! 
//! TharvexalOS Kiosk Mode:
//!   turkcekod --fullscreen --kiosk --gpu-mode
//! 
//! This boots the IDE in full-screen kiosk mode for embedded/Live USB deployments.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod error;
mod interpreter;
mod lexer;
mod parser;
mod value;
mod tensor;
mod backend;
mod explorer;
mod keyboard;
mod system_panel;

use clap::Parser;
use eframe::egui;
use std::path::PathBuf;

/// TürkçeKod - GPU-Accelerated Turkish Programming IDE
/// 
/// TharvexalOS için tam ekran kiosk modunda çalıştırılabilir.
#[derive(Parser, Debug)]
#[command(name = "turkcekod")]
#[command(version = "0.0.1-first-tests")]
#[command(author = "Efe Aydın")]
#[command(about = "Türkçe Programlama Dili - GPU Hızlandırmalı IDE")]
struct CliArgs {
    /// Tam ekran modunda başlat (TharvexalOS için)
    #[arg(short = 'f', long)]
    fullscreen: bool,
    
    /// Kiosk modunda başlat (dekorasyon yok, çıkış zor)
    #[arg(short = 'k', long)]
    kiosk: bool,
    
    /// GPU hesaplamayı zorla (varsayılan: otomatik)
    #[arg(long)]
    gpu_mode: bool,
    
    /// Başlangıçta açılacak dosya
    #[arg(short = 'o', long)]
    open: Option<PathBuf>,
    
    /// Başlangıçta kodu otomatik çalıştır
    #[arg(long)]
    autorun: bool,
}

use crate::interpreter::Interpreter;
use crate::lexer::Lexer;
use crate::parser::{Parser as TurkceParser, GuiWidgetType, Statement};
use crate::value::Value;

/// Main application state
struct TurkceKodApp {
    /// Code editor content
    code: String,
    /// Console output
    output: String,
    /// Current file path
    file_path: Option<PathBuf>,
    /// Interpreter instance
    interpreter: Interpreter,
    /// Is dark theme enabled
    dark_mode: bool,
    /// Show about dialog
    show_about: bool,
    /// Error message to display
    error_message: Option<String>,
    /// Font size
    font_size: f32,
    /// Built-in file explorer (TharvexalOS için)
    explorer: explorer::TharvexalExplorer,
    /// Kiosk modu (rfd yerine explorer kullan)
    kiosk_mode: bool,
    /// Sanal klavye (TharvexalOS için)
    keyboard: keyboard::VirtualKeyboard,
    /// Sistem ayarları paneli (TharvexalOS için)
    system_panel: system_panel::SystemPanel,
}

impl Default for TurkceKodApp {
    fn default() -> Self {
        Self {
            code: String::from(
                r#"# Türkçe Kod - Hoş geldiniz!
# Bu bir Türkçe programlama dilidir.

yaz "Merhaba Dünya!"

sayıv x = 10
sayıv y = 5
hesapla toplam = x + y
yaz toplam

eğer x > y {
    yaz "x, y'den büyük"
} değilse {
    yaz "y, x'den büyük veya eşit"
}

tekrar 3 {
    yaz "Merhaba!"
}
"#,
            ),
            output: String::new(),
            file_path: None,
            interpreter: Interpreter::new(),
            dark_mode: true,
            show_about: false,
            error_message: None,
            font_size: 16.0,
            explorer: explorer::TharvexalExplorer::new(),
            kiosk_mode: false,
            keyboard: keyboard::VirtualKeyboard::new(),
            system_panel: system_panel::SystemPanel::new(),
        }
    }
}

impl TurkceKodApp {
    /// Create a new application instance
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // Set initial theme
        cc.egui_ctx.set_visuals(egui::Visuals::dark());
        
        Self::default()
    }

    /// Run the code in the editor
    fn run_code(&mut self) {
        self.output.clear();
        self.error_message = None;
        self.interpreter.reset();

        // Tokenize
        let mut lexer = Lexer::new(&self.code);
        let tokens = match lexer.tokenize() {
            Ok(t) => t,
            Err(e) => {
                self.output = format!("[HATA] {}", e);
                self.error_message = Some(e.to_string());
                return;
            }
        };

        // Parse
        let mut parser = TurkceParser::new(tokens);
        let ast = match parser.parse() {
            Ok(a) => a,
            Err(e) => {
                self.output = format!("[HATA] {}", e);
                self.error_message = Some(e.to_string());
                return;
            }
        };

        // Execute
        match self.interpreter.execute(&ast) {
            Ok(_) => {
                self.output = self.interpreter.get_output().join("\n");
            }
            Err(e) => {
                let mut output = self.interpreter.get_output().join("\n");
                if !output.is_empty() {
                    output.push_str("\n\n");
                }
                output.push_str(&format!("[HATA] {}", e));
                self.output = output;
                self.error_message = Some(e.to_string());
            }
        }
    }

    /// Open a file
    fn open_file(&mut self) {
        if self.kiosk_mode {
            // TharvexalOS: Built-in explorer kullan
            self.explorer.show_open();
        } else {
            // Normal mod: rfd kullan
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("Türkçe Kod Dosyaları", &["turkcekod", "tk"])
                .pick_file()
            {
                self.load_file_content(&path);
            }
        }
    }
    
    /// Load file content from path
    fn load_file_content(&mut self, path: &std::path::Path) {
        match std::fs::read_to_string(path) {
            Ok(content) => {
                self.code = content;
                self.file_path = Some(path.to_path_buf());
                self.output.clear();
            }
            Err(e) => {
                self.error_message = Some(format!("Dosya açılamadı: {}", e));
            }
        }
    }

    /// Save the current file
    fn save_file(&mut self) {
        if let Some(ref path) = self.file_path {
            match std::fs::write(path, &self.code) {
                Ok(_) => {
                    self.output = format!("Dosya kaydedildi: {}", path.display());
                }
                Err(e) => {
                    self.error_message = Some(format!("Dosya kaydedilemedi: {}", e));
                }
            }
        } else {
            self.save_file_as();
        }
    }

    /// Save as a new file
    fn save_file_as(&mut self) {
        if self.kiosk_mode {
            // TharvexalOS: Built-in explorer kullan
            let filename = self.file_path
                .as_ref()
                .and_then(|p| p.file_name())
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| "yeni.turkcekod".to_string());
            self.explorer.show_save(&filename);
        } else {
            // Normal mod: rfd kullan
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("Türkçe Kod Dosyaları", &["turkcekod"])
                .set_file_name("yeni.turkcekod")
                .save_file()
            {
                self.save_to_path(&path);
            }
        }
    }
    
    /// Save content to specific path
    fn save_to_path(&mut self, path: &std::path::Path) {
        match std::fs::write(path, &self.code) {
            Ok(_) => {
                self.file_path = Some(path.to_path_buf());
                self.output = format!("Dosya kaydedildi: {}", path.display());
            }
            Err(e) => {
                self.error_message = Some(format!("Dosya kaydedilemedi: {}", e));
            }
        }
    }

    /// Create a new file
    fn new_file(&mut self) {
        self.code.clear();
        self.file_path = None;
        self.output.clear();
        self.error_message = None;
        self.interpreter.reset();
    }

    /// Toggle theme
    fn toggle_theme(&mut self, ctx: &egui::Context) {
        self.dark_mode = !self.dark_mode;
        if self.dark_mode {
            ctx.set_visuals(egui::Visuals::dark());
        } else {
            ctx.set_visuals(egui::Visuals::light());
        }
    }
    
    /// Render GUI widgets from the interpreter
    fn render_gui_widgets(&mut self, ui: &mut egui::Ui) {
        if self.interpreter.gui_widgets.is_empty() {
            ui.vertical_centered(|ui| {
                ui.add_space(20.0);
                ui.label(egui::RichText::new("Henüz bir arayüz tanımlanmadı.").color(ui.visuals().weak_text_color()));
                ui.label("Kodunuzda 'pencere', 'buton' gibi");
                ui.label("bileşenler tanımlayın.");
            });
            return;
        }

        let mut pending_callbacks: Vec<Vec<Statement>> = Vec::new();

        egui::ScrollArea::vertical().show(ui, |ui| {
            // Need to iterate with mutable access to update input values
            for widget in &mut self.interpreter.gui_widgets {
                match widget.widget_type {
                    GuiWidgetType::Window => {
                        ui.group(|ui| {
                            ui.heading(&widget.text);
                            ui.separator();
                        });
                    }
                    GuiWidgetType::Button => {
                        if ui.button(&widget.text).clicked() {
                            if !widget.callback.is_empty() {
                                pending_callbacks.push(widget.callback.clone());
                            }
                        }
                    }
                    GuiWidgetType::Label => {
                        // Check if text property is updated dynamically, otherwise use static text
                        let label_text = if let Some(Value::String(s)) = widget.properties.get("metin") {
                            s.clone()
                        } else {
                            widget.text.clone()
                        };
                        ui.label(label_text);
                    }
                    GuiWidgetType::Input => {
                        ui.horizontal(|ui| {
                            ui.label(&widget.text);
                            
                            // Get mutable reference to value
                            // We initialized "deger" property in interpreter.rs
                            if let Some(Value::String(val)) = widget.properties.get_mut("deger") {
                                ui.text_edit_singleline(val);
                            } else {
                                // Fallback if property missing (shouldn't happen with correct init)
                                let mut text = String::new(); 
                                ui.text_edit_singleline(&mut text);
                            }
                        });
                    }
                    GuiWidgetType::Layout => {
                        ui.separator();
                    }
                    GuiWidgetType::Chat => {
                        ui.group(|ui| {
                            ui.heading(&widget.text);
                            ui.separator();
                            
                            // Chat History
                            let messages = if let Some(Value::List(msgs)) = widget.properties.get("mesajlar") {
                                msgs.clone()
                            } else {
                                Vec::new()
                            };
                            
                            ui.push_id(format!("chat_hist_{}", widget.id), |ui| {
                                egui::ScrollArea::vertical()
                                    .max_height(150.0)
                                    .show(ui, |ui| {
                                        for msg in messages {
                                            ui.label(msg.as_string());
                                        }
                                    });
                            });
                            
                            ui.separator();
                            
                            // Input Area
                            ui.horizontal(|ui| {
                                // Get mutable reference to input value
                                if let Some(Value::String(val)) = widget.properties.get_mut("girdi") {
                                    ui.text_edit_singleline(val);
                                    
                                    if ui.button("Gönder").clicked() {
                                        // Set input data for 'gui_veri' command to read
                                        self.interpreter.gui_input_text = Some(val.clone());
                                        // Clear data to avoid tensor confusion
                                        self.interpreter.gui_input_data = None;
                                        
                                        if !widget.callback.is_empty() {
                                            pending_callbacks.push(widget.callback.clone());
                                        }
                                        
                                        // Clear input after sending (but need to do it after callback or separately?)
                                        // If we clear here, 'val' is cleared immediately in next frame maybe?
                                        // Let's clear it via script or let user do it?
                                        // For better UX, clear it here.
                                        *val = String::new();
                                    }
                                }
                            });
                        });
                    }
                }
                ui.add_space(4.0);
            }
        });
        
        // Execute callbacks
        for callback_block in pending_callbacks {
            if let Err(e) = self.interpreter.execute(&callback_block) {
                self.output.push_str(&format!("\n[HATA] Callback: {}\n", e));
            }
        }
    }
}

impl eframe::App for TurkceKodApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Handle keyboard shortcuts
        ctx.input(|i| {
            // F5 = Run
            if i.key_pressed(egui::Key::F5) {
                self.run_code();
            }
            // Ctrl+S = Save
            if i.modifiers.ctrl && i.key_pressed(egui::Key::S) {
                self.save_file();
            }
            // Ctrl+O = Open
            if i.modifiers.ctrl && i.key_pressed(egui::Key::O) {
                self.open_file();
            }
            // Ctrl+N = New
            if i.modifiers.ctrl && i.key_pressed(egui::Key::N) {
                self.new_file();
            }
        });
        
        // Top menu bar
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                // File menu
                ui.menu_button("📁 Dosya", |ui| {
                    if ui.button("📄 Yeni (Ctrl+N)").clicked() {
                        self.new_file();
                        ui.close_menu();
                    }
                    if ui.button("📂 Aç (Ctrl+O)").clicked() {
                        self.open_file();
                        ui.close_menu();
                    }
                    if ui.button("💾 Kaydet (Ctrl+S)").clicked() {
                        self.save_file();
                        ui.close_menu();
                    }
                    if ui.button("💾 Farklı Kaydet").clicked() {
                        self.save_file_as();
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("🚪 Çıkış").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });

                // Edit menu
                ui.menu_button("✏️ Düzenle", |ui| {
                    if ui.button("🔄 Temizle").clicked() {
                        self.output.clear();
                        self.error_message = None;
                        ui.close_menu();
                    }
                });

                // View menu
                ui.menu_button("👁 Görünüm", |ui| {
                    let theme_text = if self.dark_mode { "☀️ Aydınlık Tema" } else { "🌙 Koyu Tema" };
                    if ui.button(theme_text).clicked() {
                        self.toggle_theme(ctx);
                        ui.close_menu();
                    }
                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.label("Yazı Boyutu:");
                        ui.add(egui::DragValue::new(&mut self.font_size).range(10.0..=32.0).speed(0.5));
                    });
                });

                // Run menu
                ui.menu_button("▶️ Çalıştır", |ui| {
                    if ui.button("▶️ Çalıştır (F5)").clicked() {
                        self.run_code();
                        ui.close_menu();
                    }
                });

                // Help menu
                ui.menu_button("❓ Yardım", |ui| {
                    if ui.button("ℹ️ Hakkında").clicked() {
                        self.show_about = true;
                        ui.close_menu();
                    }
                });

                // File name in menu bar
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if let Some(ref path) = self.file_path {
                        ui.label(format!("📄 {}", path.file_name().unwrap_or_default().to_string_lossy()));
                    } else {
                        ui.label("📄 Yeni Dosya");
                    }
                });
            });
        });

        // Bottom panel with run button and status
        egui::TopBottomPanel::bottom("bottom_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                let run_button = egui::Button::new(egui::RichText::new("▶️ Çalıştır").size(16.0))
                    .fill(egui::Color32::from_rgb(0, 120, 212));
                
                if ui.add(run_button).clicked() {
                    self.run_code();
                }

                ui.separator();

                ui.label("Efe Aydın Türkçe Kod | TharvexalOS");

                // Error indicator
                if self.error_message.is_some() {
                    ui.colored_label(egui::Color32::RED, "⚠️");
                }
                
                // Right side: System buttons (kiosk mode or always show)
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    // Sistem Ayarları butonu
                    system_panel::system_button(ui, &mut self.system_panel);
                    
                    // Sanal Klavye butonu (sadece kiosk modunda göster)
                    if self.kiosk_mode {
                        keyboard::keyboard_toggle_button(ui, &mut self.keyboard);
                    }
                    
                    // Hardware monitor küçük gösterge
                    if let Some(temp) = self.system_panel.gpu_temp {
                        let color = if temp > 80.0 { egui::Color32::RED } 
                            else if temp > 60.0 { egui::Color32::YELLOW } 
                            else { egui::Color32::GREEN };
                        ui.colored_label(color, format!("🌡️{:.0}°", temp));
                    }
                    
                    let ram_percent = if self.system_panel.ram_total > 0 {
                        (self.system_panel.ram_used * 100 / self.system_panel.ram_total) as u8
                    } else { 0 };
                    ui.label(format!("💾{}%", ram_percent));
                });
            });
        });

        // Console panel at the bottom
        egui::TopBottomPanel::bottom("console_panel")
            .resizable(true)
            .min_height(100.0)
            .default_height(150.0)
            .show(ctx, |ui| {
                ui.heading("📟 Konsol");
                ui.separator();
                
                egui::ScrollArea::vertical().show(ui, |ui| {
                    let console_bg = if self.dark_mode {
                        egui::Color32::from_rgb(30, 30, 30)
                    } else {
                        egui::Color32::from_rgb(250, 250, 250)
                    };

                    let console_fg = if self.dark_mode {
                        egui::Color32::from_rgb(212, 212, 212)
                    } else {
                        egui::Color32::BLACK
                    };

                    egui::Frame::none()
                        .fill(console_bg)
                        .inner_margin(8.0)
                        .show(ui, |ui| {
                            ui.add(
                                egui::TextEdit::multiline(&mut self.output.as_str())
                                    .font(egui::TextStyle::Monospace)
                                    .desired_width(f32::INFINITY)
                                    .text_color(console_fg),
                            );
                        });
                });
            });

        // GUI App Output Panel (Right Side)
        egui::SidePanel::right("gui_panel")
            .resizable(true)
            .min_width(200.0)
            .default_width(300.0)
            .show(ctx, |ui| {
                ui.heading("📱 Uygulama");
                ui.separator();
                
                self.render_gui_widgets(ui);
            });

        // Main editor panel
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("📝 Kod Editörü");
            ui.separator();

            egui::ScrollArea::both().show(ui, |ui| {
                let editor_bg = if self.dark_mode {
                    egui::Color32::from_rgb(37, 37, 38)
                } else {
                    egui::Color32::from_rgb(255, 255, 255)
                };

                let editor_fg = if self.dark_mode {
                    egui::Color32::from_rgb(212, 212, 212)
                } else {
                    egui::Color32::BLACK
                };
                
                let line_num_color = if self.dark_mode {
                    egui::Color32::from_rgb(100, 100, 100)
                } else {
                    egui::Color32::from_rgb(150, 150, 150)
                };

                egui::Frame::none()
                    .fill(editor_bg)
                    .inner_margin(8.0)
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            // Line numbers column
                            let line_count = self.code.lines().count().max(1);
                            let line_numbers: String = (1..=line_count)
                                .map(|n| format!("{:4}", n))
                                .collect::<Vec<_>>()
                                .join("\n");
                            
                            ui.add(
                                egui::TextEdit::multiline(&mut line_numbers.as_str())
                                    .font(egui::FontId::monospace(self.font_size))
                                    .desired_width(50.0)
                                    .text_color(line_num_color)
                                    .interactive(false)
                            );
                            
                            ui.separator();
                            
                            // Code editor
                            let text_edit = egui::TextEdit::multiline(&mut self.code)
                                .font(egui::FontId::monospace(self.font_size))
                                .desired_width(f32::INFINITY)
                                .text_color(editor_fg)
                                .lock_focus(true);

                            ui.add(text_edit);
                        });
                    });
            });
        });

        // About dialog
        if self.show_about {
            egui::Window::new("Türkçe Kod Hakkında")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    ui.vertical_centered(|ui| {
                        ui.heading("🇹🇷 Türkçe Kod");
                        ui.add_space(10.0);
                        ui.label("Sürüm 0.0.1-first-tests");
                        ui.add_space(5.0);
                        ui.label("Türkçe Programlama Dili");
                        ui.add_space(10.0);
                        ui.label("© 2025 Efe Aydın");
                        ui.add_space(5.0);
                        ui.label("GPL v3 Lisansı");
                        ui.add_space(15.0);
                        ui.label("Rust + egui + WebGPU ile");
                        ui.label("GPU hızlandırmalı IDE");
                        ui.add_space(15.0);
                        if ui.button("Kapat").clicked() {
                            self.show_about = false;
                        }
                    });
                });
        }
        
        // Tharvexal Explorer (built-in file browser)
        self.explorer.show(ctx);
        
        // Handle pending file from explorer
        if let Some(path) = self.explorer.take_pending_file() {
            if self.explorer.save_mode {
                // Saving a file
                self.save_to_path(&path);
            } else {
                // Opening a file
                self.load_file_content(&path);
            }
        }
        
        // System Panel (settings, hardware monitor)
        self.system_panel.show(ctx);
        
        // Virtual Keyboard (kiosk mode)
        self.keyboard.show(ctx);
        
        // Handle keyboard input
        let output = self.keyboard.take_output();
        if !output.is_empty() {
            // Append to code editor (simple implementation)
            self.code.push_str(&output);
        }
        if self.keyboard.take_backspace() {
            self.code.pop();
        }

        // Handle keyboard shortcuts
        ctx.input(|i| {
            if i.modifiers.ctrl && i.key_pressed(egui::Key::N) {
                self.new_file();
            }
            if i.modifiers.ctrl && i.key_pressed(egui::Key::O) {
                self.open_file();
            }
            if i.modifiers.ctrl && i.key_pressed(egui::Key::S) {
                self.save_file();
            }
            if i.key_pressed(egui::Key::F5) {
                self.run_code();
            }
        });
    }
}

fn main() -> eframe::Result<()> {
    env_logger::init();
    
    // Parse CLI arguments
    let args = CliArgs::parse();
    
    // Configure viewport based on CLI args
    let mut viewport = egui::ViewportBuilder::default()
        .with_title("Türkçe Kod IDE - TharvexalOS");
    
    if args.fullscreen {
        // TharvexalOS fullscreen mode
        viewport = viewport
            .with_fullscreen(true)
            .with_maximized(true);
    } else {
        viewport = viewport
            .with_inner_size([1024.0, 768.0])
            .with_min_inner_size([800.0, 600.0]);
    }
    
    if args.kiosk {
        // Kiosk mode: no decorations, no close button
        viewport = viewport
            .with_decorations(false)
            .with_resizable(false);
    }
    
    // Set GPU mode if requested
    if args.gpu_mode {
        backend::set_execution_mode(backend::ExecutionMode::Gpu);
    }

    let native_options = eframe::NativeOptions {
        viewport,
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };
    
    // Clone args for closure
    let open_file = args.open.clone();
    let autorun = args.autorun;
    let kiosk_mode = args.kiosk;

    eframe::run_native(
        "Türkçe Kod IDE",
        native_options,
        Box::new(move |cc| {
            let mut app = TurkceKodApp::new(cc);
            
            // Set kiosk mode (uses built-in explorer instead of rfd)
            app.kiosk_mode = kiosk_mode;
            
            // Open file if specified
            if let Some(path) = &open_file {
                if let Ok(content) = std::fs::read_to_string(path) {
                    app.code = content;
                    app.file_path = Some(path.clone());
                }
            }
            
            // Auto-run if specified
            if autorun {
                app.run_code();
            }
            
            Ok(Box::new(app))
        }),
    )
}
