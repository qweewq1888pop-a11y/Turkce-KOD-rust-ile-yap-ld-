//! Tharvexal System Panel - Linux System Controls
//!
//! TharvexalOS için sistem ayarları paneli.
//! WiFi, ses, parlaklık ve güç yönetimi kontrolleri.

use eframe::egui;
use std::process::Command;

/// WiFi ağı bilgisi
#[derive(Clone, Debug)]
pub struct WifiNetwork {
    pub ssid: String,
    pub signal: u8,
    pub connected: bool,
    pub secured: bool,
}

/// Sistem paneli durumu
pub struct SystemPanel {
    /// Panel görünür mü?
    pub visible: bool,
    /// Mevcut ses seviyesi (0-100)
    volume: u8,
    /// Parlaklık seviyesi (0-100)
    brightness: u8,
    /// WiFi etkin mi?
    wifi_enabled: bool,
    /// Bulunan WiFi ağları
    wifi_networks: Vec<WifiNetwork>,
    /// Bağlı WiFi adı
    connected_wifi: Option<String>,
    /// WiFi şifre girişi
    wifi_password: String,
    /// Seçili WiFi
    selected_wifi: Option<String>,
    /// Hata mesajı
    error: Option<String>,
    /// Bilgi mesajı
    info: Option<String>,
    /// Son tarama zamanı
    last_scan: std::time::Instant,
    // === Hardware Monitor ===
    /// GPU sıcaklığı (°C)
    pub gpu_temp: Option<f32>,
    /// GPU kullanımı (%)
    pub gpu_usage: Option<u8>,
    /// Toplam RAM (MB)
    pub ram_total: u64,
    /// Kullanılan RAM (MB)
    pub ram_used: u64,
    /// CPU kullanımı (%)
    pub cpu_usage: u8,
}

impl Default for SystemPanel {
    fn default() -> Self {
        Self::new()
    }
}

impl SystemPanel {
    pub fn new() -> Self {
        let mut panel = Self {
            visible: false,
            volume: 50,
            brightness: 80,
            wifi_enabled: true,
            wifi_networks: Vec::new(),
            connected_wifi: None,
            wifi_password: String::new(),
            selected_wifi: None,
            error: None,
            info: None,
            last_scan: std::time::Instant::now(),
            // Hardware monitor
            gpu_temp: None,
            gpu_usage: None,
            ram_total: 0,
            ram_used: 0,
            cpu_usage: 0,
        };
        panel.refresh_system_info();
        panel
    }
    
    /// Sistem bilgilerini yenile
    pub fn refresh_system_info(&mut self) {
        self.get_current_volume();
        self.get_current_brightness();
        self.get_connected_wifi();
        self.get_hardware_info();
    }
    
    /// Donanım bilgilerini al (GPU, RAM, CPU)
    pub fn get_hardware_info(&mut self) {
        self.get_gpu_info();
        self.get_ram_info();
        self.get_cpu_usage();
    }
    
    /// GPU sıcaklığını al (NVIDIA: nvidia-smi, AMD: sensors)
    fn get_gpu_info(&mut self) {
        // NVIDIA GPU
        if let Ok(output) = Self::run_command("nvidia-smi", &["--query-gpu=temperature.gpu,utilization.gpu", "--format=csv,noheader,nounits"]) {
            let parts: Vec<&str> = output.split(',').map(|s| s.trim()).collect();
            if parts.len() >= 2 {
                self.gpu_temp = parts[0].parse().ok();
                self.gpu_usage = parts[1].parse().ok();
                return;
            }
        }
        
        // AMD GPU (sensors)
        if let Ok(output) = Self::run_command("sensors", &["-u"]) {
            for line in output.lines() {
                if line.contains("edge") || line.contains("junction") {
                    if let Some(temp_line) = output.lines().find(|l| l.contains("temp1_input")) {
                        if let Some(val) = temp_line.split(':').last() {
                            self.gpu_temp = val.trim().parse().ok();
                        }
                    }
                }
            }
        }
    }
    
    /// RAM bilgisini al (/proc/meminfo)
    fn get_ram_info(&mut self) {
        if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
            let mut total: u64 = 0;
            let mut available: u64 = 0;
            
            for line in content.lines() {
                if line.starts_with("MemTotal:") {
                    if let Some(val) = line.split_whitespace().nth(1) {
                        total = val.parse().unwrap_or(0) / 1024; // KB -> MB
                    }
                } else if line.starts_with("MemAvailable:") {
                    if let Some(val) = line.split_whitespace().nth(1) {
                        available = val.parse().unwrap_or(0) / 1024;
                    }
                }
            }
            
            self.ram_total = total;
            self.ram_used = total.saturating_sub(available);
        }
    }
    
    /// CPU kullanımını al (/proc/stat)
    fn get_cpu_usage(&mut self) {
        if let Ok(output) = Self::run_command("top", &["-bn1", "-p0"]) {
            for line in output.lines() {
                if line.contains("Cpu(s)") || line.starts_with("%Cpu") {
                    // Idle değerini bul ve 100'den çıkar
                    if let Some(idle_pos) = line.find("id") {
                        let before = &line[..idle_pos];
                        if let Some(val) = before.split_whitespace().last() {
                            if let Ok(idle) = val.replace(',', ".").parse::<f32>() {
                                self.cpu_usage = (100.0 - idle) as u8;
                            }
                        }
                    }
                }
            }
        }
    }
    
    /// Paneli göster
    pub fn show_panel(&mut self) {
        self.visible = true;
        self.refresh_system_info();
    }
    
    /// Linux komutu çalıştır
    fn run_command(cmd: &str, args: &[&str]) -> Result<String, String> {
        Command::new(cmd)
            .args(args)
            .output()
            .map(|output| {
                String::from_utf8_lossy(&output.stdout).trim().to_string()
            })
            .map_err(|e| format!("Komut hatası: {}", e))
    }
    
    /// Mevcut ses seviyesini al
    fn get_current_volume(&mut self) {
        // pactl veya amixer ile ses seviyesi
        if let Ok(output) = Self::run_command("pactl", &["get-sink-volume", "@DEFAULT_SINK@"]) {
            // Parse "Volume: front-left: 65536 / 100%"
            if let Some(percent_pos) = output.find('%') {
                if let Some(start) = output[..percent_pos].rfind(' ') {
                    if let Ok(vol) = output[start+1..percent_pos].parse::<u8>() {
                        self.volume = vol.min(100);
                    }
                }
            }
        }
    }
    
    /// Ses seviyesini ayarla
    fn set_volume(&mut self, volume: u8) {
        self.volume = volume.min(100);
        let vol_str = format!("{}%", self.volume);
        let _ = Self::run_command("pactl", &["set-sink-volume", "@DEFAULT_SINK@", &vol_str]);
    }
    
    /// Sesi kapat/aç
    fn toggle_mute(&mut self) {
        let _ = Self::run_command("pactl", &["set-sink-mute", "@DEFAULT_SINK@", "toggle"]);
    }
    
    /// Mevcut parlaklığı al
    fn get_current_brightness(&mut self) {
        // brightnessctl ile parlaklık
        if let Ok(output) = Self::run_command("brightnessctl", &["get"]) {
            if let Ok(current) = output.parse::<u32>() {
                if let Ok(max_output) = Self::run_command("brightnessctl", &["max"]) {
                    if let Ok(max) = max_output.parse::<u32>() {
                        self.brightness = ((current * 100) / max) as u8;
                    }
                }
            }
        }
    }
    
    /// Parlaklığı ayarla
    fn set_brightness(&mut self, brightness: u8) {
        self.brightness = brightness.min(100).max(5); // min %5
        let bright_str = format!("{}%", self.brightness);
        let _ = Self::run_command("brightnessctl", &["set", &bright_str]);
    }
    
    /// Bağlı WiFi'ı al
    fn get_connected_wifi(&mut self) {
        if let Ok(output) = Self::run_command("nmcli", &["-t", "-f", "NAME", "con", "show", "--active"]) {
            let connections: Vec<&str> = output.lines().collect();
            // İlk WiFi bağlantısını al
            self.connected_wifi = connections.first().map(|s| s.to_string());
        }
    }
    
    /// WiFi ağlarını tara
    fn scan_wifi(&mut self) {
        // Çok sık tarama yapma
        if self.last_scan.elapsed().as_secs() < 5 {
            return;
        }
        self.last_scan = std::time::Instant::now();
        
        self.wifi_networks.clear();
        
        // nmcli ile WiFi tarama
        if let Ok(output) = Self::run_command("nmcli", &["-t", "-f", "SSID,SIGNAL,SECURITY,ACTIVE", "dev", "wifi"]) {
            for line in output.lines() {
                let parts: Vec<&str> = line.split(':').collect();
                if parts.len() >= 4 {
                    let ssid = parts[0].to_string();
                    if ssid.is_empty() { continue; }
                    
                    let signal = parts[1].parse().unwrap_or(0);
                    let secured = !parts[2].is_empty() && parts[2] != "--";
                    let connected = parts[3] == "yes";
                    
                    self.wifi_networks.push(WifiNetwork {
                        ssid,
                        signal,
                        secured,
                        connected,
                    });
                }
            }
        }
    }
    
    /// WiFi'a bağlan
    fn connect_wifi(&mut self, ssid: &str, password: &str) {
        self.error = None;
        self.info = Some(format!("{}' ağına bağlanılıyor...", ssid));
        
        let result = if password.is_empty() {
            Self::run_command("nmcli", &["dev", "wifi", "connect", ssid])
        } else {
            Self::run_command("nmcli", &["dev", "wifi", "connect", ssid, "password", password])
        };
        
        match result {
            Ok(_) => {
                self.info = Some(format!("'{}' ağına bağlandı!", ssid));
                self.connected_wifi = Some(ssid.to_string());
                self.wifi_password.clear();
            }
            Err(e) => {
                self.error = Some(format!("Bağlantı hatası: {}", e));
            }
        }
    }
    
    /// WiFi bağlantısını kes
    fn disconnect_wifi(&mut self) {
        if let Some(ssid) = &self.connected_wifi.clone() {
            let _ = Self::run_command("nmcli", &["con", "down", ssid]);
            self.connected_wifi = None;
            self.info = Some("WiFi bağlantısı kesildi".to_string());
        }
    }
    
    /// Sistemi kapat
    fn shutdown(&self) {
        let _ = Self::run_command("systemctl", &["poweroff"]);
    }
    
    /// Sistemi yeniden başlat
    fn reboot(&self) {
        let _ = Self::run_command("systemctl", &["reboot"]);
    }
    
    /// Uyku moduna geç
    fn suspend(&self) {
        let _ = Self::run_command("systemctl", &["suspend"]);
    }
    
    /// Panel UI'ını çiz
    pub fn show(&mut self, ctx: &egui::Context) {
        if !self.visible {
            return;
        }
        
        egui::Window::new("⚙️ Sistem Ayarları")
            .collapsible(false)
            .resizable(true)
            .default_size([400.0, 500.0])
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                // Hata/bilgi mesajları
                if let Some(err) = &self.error {
                    ui.colored_label(egui::Color32::RED, format!("⚠️ {}", err));
                }
                if let Some(info) = &self.info {
                    ui.colored_label(egui::Color32::GREEN, format!("✓ {}", info));
                }
                
                ui.separator();
                
                // === SES ===
                ui.heading("🔊 Ses");
                ui.horizontal(|ui| {
                    ui.label("Ses Seviyesi:");
                    let mut vol = self.volume as f32;
                    if ui.add(egui::Slider::new(&mut vol, 0.0..=100.0).suffix("%")).changed() {
                        self.set_volume(vol as u8);
                    }
                    if ui.button("🔇").clicked() {
                        self.toggle_mute();
                    }
                });
                
                ui.separator();
                
                // === PARLAKLIK ===
                ui.heading("☀️ Parlaklık");
                ui.horizontal(|ui| {
                    ui.label("Ekran Parlaklığı:");
                    let mut bright = self.brightness as f32;
                    if ui.add(egui::Slider::new(&mut bright, 5.0..=100.0).suffix("%")).changed() {
                        self.set_brightness(bright as u8);
                    }
                });
                
                ui.separator();
                
                // === WIFI ===
                ui.heading("📶 WiFi");
                
                ui.horizontal(|ui| {
                    if let Some(ssid) = &self.connected_wifi {
                        ui.label(format!("Bağlı: {}", ssid));
                        if ui.button("Bağlantıyı Kes").clicked() {
                            self.disconnect_wifi();
                        }
                    } else {
                        ui.label("Bağlı değil");
                    }
                    
                    if ui.button("🔄 Tara").clicked() {
                        self.scan_wifi();
                    }
                });
                
                // WiFi ağ listesi
                egui::ScrollArea::vertical()
                    .max_height(120.0)
                    .show(ui, |ui| {
                        for network in self.wifi_networks.clone() {
                            let icon = if network.connected { "✓" } 
                                else if network.secured { "🔒" } 
                                else { "📶" };
                            
                            let signal_bars = match network.signal {
                                0..=25 => "▂",
                                26..=50 => "▂▄",
                                51..=75 => "▂▄▆",
                                _ => "▂▄▆█",
                            };
                            
                            let label = format!("{} {} {} ({}%)", icon, network.ssid, signal_bars, network.signal);
                            
                            if ui.selectable_label(
                                self.selected_wifi.as_ref() == Some(&network.ssid),
                                &label
                            ).clicked() {
                                self.selected_wifi = Some(network.ssid.clone());
                            }
                        }
                    });
                
                // WiFi şifre girişi
                if let Some(ssid) = &self.selected_wifi.clone() {
                    ui.horizontal(|ui| {
                        ui.label("Şifre:");
                        ui.add(egui::TextEdit::singleline(&mut self.wifi_password).password(true));
                        if ui.button("Bağlan").clicked() {
                            let pass = self.wifi_password.clone();
                            self.connect_wifi(ssid, &pass);
                        }
                    });
                }
                
                ui.separator();
                
                // === DONANIM MONİTÖRÜ ===
                ui.heading("🖥️ Donanım");
                
                // RAM
                ui.horizontal(|ui| {
                    ui.label("RAM:");
                    let ram_percent = if self.ram_total > 0 {
                        (self.ram_used as f32 / self.ram_total as f32 * 100.0) as u8
                    } else { 0 };
                    
                    let ram_bar = egui::ProgressBar::new(ram_percent as f32 / 100.0)
                        .text(format!("{} / {} MB ({}%)", self.ram_used, self.ram_total, ram_percent));
                    ui.add(ram_bar);
                });
                
                // CPU
                ui.horizontal(|ui| {
                    ui.label("CPU:");
                    let cpu_bar = egui::ProgressBar::new(self.cpu_usage as f32 / 100.0)
                        .text(format!("{}%", self.cpu_usage));
                    ui.add(cpu_bar);
                });
                
                // GPU
                ui.horizontal(|ui| {
                    ui.label("GPU:");
                    if let Some(temp) = self.gpu_temp {
                        let temp_color = if temp > 80.0 {
                            egui::Color32::RED
                        } else if temp > 60.0 {
                            egui::Color32::YELLOW
                        } else {
                            egui::Color32::GREEN
                        };
                        ui.colored_label(temp_color, format!("🌡️ {:.0}°C", temp));
                    }
                    if let Some(usage) = self.gpu_usage {
                        let gpu_bar = egui::ProgressBar::new(usage as f32 / 100.0)
                            .text(format!("{}%", usage));
                        ui.add(gpu_bar);
                    } else {
                        ui.label("Bilgi yok");
                    }
                });
                
                if ui.small_button("🔄 Yenile").clicked() {
                    self.get_hardware_info();
                }
                
                ui.separator();
                
                // === GÜÇ ===
                ui.heading("⏻ Güç");
                ui.horizontal(|ui| {
                    if ui.button("😴 Uyku").clicked() {
                        self.suspend();
                    }
                    if ui.button("🔄 Yeniden Başlat").clicked() {
                        self.reboot();
                    }
                    if ui.button("⏻ Kapat").clicked() {
                        self.shutdown();
                    }
                });
                
                ui.separator();
                
                // Kapat butonu
                ui.horizontal(|ui| {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("❌ Kapat").clicked() {
                            self.visible = false;
                        }
                    });
                });
            });
    }
}

/// Sistem paneli toggle butonu (durum çubuğu için)
pub fn system_button(ui: &mut egui::Ui, panel: &mut SystemPanel) {
    if ui.button("⚙️").on_hover_text("Sistem Ayarları").clicked() {
        if panel.visible {
            panel.visible = false;
        } else {
            panel.show_panel();
        }
    }
}
