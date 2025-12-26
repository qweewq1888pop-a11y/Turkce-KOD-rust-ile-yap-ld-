# Türkçe Kod

**GPU-Accelerated Turkish Programming Language with Built-in AI/ML Capabilities**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Overview

Türkçe Kod is a programming language designed with **native Turkish syntax**, making programming accessible to Turkish speakers. It features:

- **Turkish Keywords**: `eğer` (if), `işlev` (function), `yaz` (print), etc.
- **GPU Acceleration**: Built-in tensor operations powered by `wgpu` for AI/ML workloads.
- **Autograd Engine**: Automatic differentiation for neural network training.
- **Integrated GUI**: Declarative UI components using `egui`.

## Features

| Feature | Description |
| :--- | :--- |
| **Türkçe Sözdizimi** | All keywords and error messages are in Turkish. |
| **Matris İşlemleri** | GPU-accelerated matrix multiplication, activations (ReLU, GeLU, Softmax). |
| **AI Eğitimi** | Built-in backpropagation (`geri_yayilim`) and optimizers. |
| **Dosya İşlemleri** | Native file I/O (`oku_dosya`, `yaz_dosya`). |

## Building

Requires Rust 1.70+ and a GPU with Vulkan/Metal/DX12 support.

```bash
cargo build --release
```

The executable will be in `target/release/turkcekod.exe` (Windows) or `target/release/turkcekod` (Linux/macOS).

## Example

```turkcekod
# Merhaba Dünya
yaz "Merhaba, Dünya!"

# Değişken tanımlama
sayıv x = 10
metinv isim = "Ali"

# Koşul
eğer x > 5 {
    yaz "x beşten büyük"
}

# Fonksiyon
işlev topla(a, b) {
    dön a + b
}

yaz topla(3, 4)  # 7
```

## License

This project is licensed under the **GNU General Public License v3.0** - see the [LICENSE](LICENSE) file for details.

## Author

Efe Aydın
