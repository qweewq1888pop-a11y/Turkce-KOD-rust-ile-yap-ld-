//! Tharvexal Keyboard - On-Screen Virtual Keyboard
//!
//! TharvexalOS için dokunmatik ekran desteği.
//! Kiosk modunda fiziksel klavye olmadığında kullanılır.

use eframe::egui;

/// Klavye düzeni türleri
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KeyboardLayout {
    TurkishLower,
    TurkishUpper,
    Numbers,
    Symbols,
}

/// Sanal klavye durumu
pub struct VirtualKeyboard {
    /// Klavye görünür mü?
    pub visible: bool,
    /// Aktif düzen
    layout: KeyboardLayout,
    /// Shift tuşu aktif mi?
    shift_active: bool,
    /// Çıktı buffer (basılan tuşlar)
    pub output_buffer: String,
    /// Backspace basıldı mı?
    pub backspace_pressed: bool,
    /// Enter basıldı mı?
    pub enter_pressed: bool,
}

impl Default for VirtualKeyboard {
    fn default() -> Self {
        Self::new()
    }
}

impl VirtualKeyboard {
    pub fn new() -> Self {
        Self {
            visible: false,
            layout: KeyboardLayout::TurkishLower,
            shift_active: false,
            output_buffer: String::new(),
            backspace_pressed: false,
            enter_pressed: false,
        }
    }
    
    /// Klavyeyi göster
    pub fn show_keyboard(&mut self) {
        self.visible = true;
        self.output_buffer.clear();
        self.backspace_pressed = false;
        self.enter_pressed = false;
    }
    
    /// Klavyeyi gizle
    pub fn hide_keyboard(&mut self) {
        self.visible = false;
    }
    
    /// Çıktıyı al ve temizle
    pub fn take_output(&mut self) -> String {
        std::mem::take(&mut self.output_buffer)
    }
    
    /// Backspace durumunu kontrol et ve sıfırla
    pub fn take_backspace(&mut self) -> bool {
        let val = self.backspace_pressed;
        self.backspace_pressed = false;
        val
    }
    
    /// Enter durumunu kontrol et ve sıfırla
    pub fn take_enter(&mut self) -> bool {
        let val = self.enter_pressed;
        self.enter_pressed = false;
        val
    }
    
    /// Türkçe küçük harf düzeni
    fn turkish_lower() -> Vec<Vec<&'static str>> {
        vec![
            vec!["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"],
            vec!["q", "w", "e", "r", "t", "y", "u", "ı", "o", "p", "ğ", "ü"],
            vec!["a", "s", "d", "f", "g", "h", "j", "k", "l", "ş", "i"],
            vec!["⇧", "z", "x", "c", "v", "b", "n", "m", "ö", "ç", "⌫"],
            vec!["123", "🌐", "␣ Boşluk", ".", "↵"],
        ]
    }
    
    /// Türkçe büyük harf düzeni
    fn turkish_upper() -> Vec<Vec<&'static str>> {
        vec![
            vec!["!", "@", "#", "₺", "%", "&", "*", "(", ")", "="],
            vec!["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", "Ğ", "Ü"],
            vec!["A", "S", "D", "F", "G", "H", "J", "K", "L", "Ş", "İ"],
            vec!["⇧", "Z", "X", "C", "V", "B", "N", "M", "Ö", "Ç", "⌫"],
            vec!["123", "🌐", "␣ Boşluk", ",", "↵"],
        ]
    }
    
    /// Sayı düzeni
    fn numbers() -> Vec<Vec<&'static str>> {
        vec![
            vec!["1", "2", "3", "+", "-"],
            vec!["4", "5", "6", "*", "/"],
            vec!["7", "8", "9", "(", ")"],
            vec!["#@!", "0", ".", "⌫", "↵"],
            vec!["ABC", "🌐", "␣ Boşluk", "=", ","],
        ]
    }
    
    /// Sembol düzeni
    fn symbols() -> Vec<Vec<&'static str>> {
        vec![
            vec!["!", "@", "#", "$", "%", "^", "&", "*"],
            vec!["(", ")", "[", "]", "{", "}", "<", ">"],
            vec!["-", "_", "=", "+", "/", "\\", "|", "~"],
            vec!["'", "\"", ";", ":", ",", ".", "?", "⌫"],
            vec!["123", "ABC", "␣ Boşluk", "↵"],
        ]
    }
    
    fn get_current_layout(&self) -> Vec<Vec<&'static str>> {
        match self.layout {
            KeyboardLayout::TurkishLower => Self::turkish_lower(),
            KeyboardLayout::TurkishUpper => Self::turkish_upper(),
            KeyboardLayout::Numbers => Self::numbers(),
            KeyboardLayout::Symbols => Self::symbols(),
        }
    }
    
    /// Tuş basıldığında
    fn handle_key(&mut self, key: &str) {
        match key {
            "⇧" => {
                self.shift_active = !self.shift_active;
                self.layout = if self.shift_active {
                    KeyboardLayout::TurkishUpper
                } else {
                    KeyboardLayout::TurkishLower
                };
            }
            "⌫" => {
                self.backspace_pressed = true;
            }
            "↵" => {
                self.enter_pressed = true;
            }
            "␣ Boşluk" => {
                self.output_buffer.push(' ');
            }
            "123" => {
                self.layout = KeyboardLayout::Numbers;
            }
            "#@!" => {
                self.layout = KeyboardLayout::Symbols;
            }
            "ABC" => {
                self.layout = KeyboardLayout::TurkishLower;
                self.shift_active = false;
            }
            "🌐" => {
                // Dil değiştirme - şimdilik sadece Türkçe
            }
            _ => {
                self.output_buffer.push_str(key);
                // Shift'i otomatik kapat
                if self.shift_active && self.layout == KeyboardLayout::TurkishUpper {
                    self.shift_active = false;
                    self.layout = KeyboardLayout::TurkishLower;
                }
            }
        }
    }
    
    /// Klavye UI'ını çiz
    pub fn show(&mut self, ctx: &egui::Context) {
        if !self.visible {
            return;
        }
        
        egui::TopBottomPanel::bottom("virtual_keyboard")
            .resizable(false)
            .min_height(200.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.heading("⌨️ Klavye");
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.small_button("✕ Kapat").clicked() {
                            self.visible = false;
                        }
                    });
                });
                
                ui.separator();
                
                let layout = self.get_current_layout();
                
                for row in layout {
                    ui.horizontal(|ui| {
                        ui.add_space(10.0);
                        for key in row {
                            let button_width = if key.contains("Boşluk") {
                                150.0
                            } else if key.len() > 2 {
                                50.0
                            } else {
                                35.0
                            };
                            
                            let button = egui::Button::new(
                                egui::RichText::new(*key).size(18.0)
                            )
                            .min_size(egui::vec2(button_width, 40.0));
                            
                            if ui.add(button).clicked() {
                                self.handle_key(key);
                            }
                        }
                    });
                    ui.add_space(2.0);
                }
            });
    }
}

/// Klavye toggle butonu (IDE'nin herhangi bir yerinde gösterilebilir)
pub fn keyboard_toggle_button(ui: &mut egui::Ui, keyboard: &mut VirtualKeyboard) {
    if ui.button("⌨️").on_hover_text("Sanal Klavye").clicked() {
        if keyboard.visible {
            keyboard.hide_keyboard();
        } else {
            keyboard.show_keyboard();
        }
    }
}
