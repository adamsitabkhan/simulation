use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::thread;

use crossbeam_channel::{unbounded, Receiver, Sender};
use eframe::egui;

const TILE_SIZE: usize = 256;
const MAX_ITER: u32 = 1000;

const PALETTE: [(f64, f64, f64); 16] = [
    (0.051, 0.027, 0.106),
    (0.098, 0.027, 0.275),
    (0.141, 0.039, 0.490),
    (0.098, 0.110, 0.667),
    (0.067, 0.216, 0.741),
    (0.039, 0.376, 0.745),
    (0.098, 0.545, 0.667),
    (0.224, 0.698, 0.494),
    (0.475, 0.824, 0.314),
    (0.741, 0.906, 0.224),
    (0.929, 0.933, 0.231),
    (0.996, 0.847, 0.220),
    (0.996, 0.682, 0.204),
    (0.949, 0.471, 0.180),
    (0.824, 0.259, 0.153),
    (0.620, 0.098, 0.129),
];

fn palette_color(t: f64) -> [u8; 4] {
    let n = PALETTE.len() as f64;
    let t = t % n;
    let idx = t.floor() as usize % PALETTE.len();
    let frac = t - t.floor();

    let next = (idx + 1) % PALETTE.len();
    let (r0, g0, b0) = PALETTE[idx];
    let (r1, g1, b1) = PALETTE[next];

    let f = (1.0 - (frac * std::f64::consts::PI).cos()) * 0.5;

    let r = r0 + (r1 - r0) * f;
    let g = g0 + (g1 - g0) * f;
    let b = b0 + (b1 - b0) * f;

    [
        (r * 255.0).min(255.0) as u8,
        (g * 255.0).min(255.0) as u8,
        (b * 255.0).min(255.0) as u8,
        255,
    ]
}

#[inline(always)]
fn mandelbrot(cr: f64, ci: f64) -> Option<f64> {
    let mut zr = 0.0_f64;
    let mut zi = 0.0_f64;

    let q = (cr - 0.25) * (cr - 0.25) + ci * ci;
    if q * (q + (cr - 0.25)) <= 0.25 * ci * ci {
        return None;
    }
    if (cr + 1.0) * (cr + 1.0) + ci * ci <= 0.0625 {
        return None;
    }

    for i in 0..MAX_ITER {
        let zr2 = zr * zr;
        let zi2 = zi * zi;
        if zr2 + zi2 > 256.0 {
            let log_zn = (zr2 + zi2).ln() / 2.0;
            let nu = log_zn.ln() / std::f64::consts::LN_2;
            return Some(i as f64 + 1.0 - nu);
        }
        zi = 2.0 * zr * zi + ci;
        zr = zr2 - zi2 + cr;
    }
    None
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct TileKey {
    x: i64,
    y: i64,
    level: i32,
}

struct TileData {
    key: TileKey,
    pixels: Vec<u8>,
}

fn compute_tile(key: TileKey) -> TileData {
    let scale = 2.0_f64.powi(-key.level);
    let tile_w_coord = 2.0;
    
    let x_min = key.x as f64 * tile_w_coord * scale;
    let y_max = key.y as f64 * tile_w_coord * scale; 
    
    let dx = (tile_w_coord * scale) / TILE_SIZE as f64;
    let dy = (tile_w_coord * scale) / TILE_SIZE as f64;

    let mut pixels = Vec::with_capacity(TILE_SIZE * TILE_SIZE * 4);

    for py in 0..TILE_SIZE {
        let ci = y_max - (py as f64 * dy);
        for px in 0..TILE_SIZE {
            let cr = x_min + (px as f64 * dx);
            let color = match mandelbrot(cr, ci) {
                Some(val) => palette_color(val * 0.15),
                None => [0, 0, 0, 255],
            };
            pixels.extend_from_slice(&color);
        }
    }

    TileData { key, pixels }
}

struct RenderTile {
    texture: egui::TextureHandle,
    insertion_time: f64,
}

struct MandelbrotApp {
    cache: HashMap<TileKey, RenderTile>,
    pending: HashSet<TileKey>,
    tx_req: Sender<TileKey>,
    rx_res: Receiver<TileData>,
    
    center_x: f64,
    center_y: f64,
    level: i32,     // Base index, discrete powers of 2 (0 is minimum)
    fractional_zoom: f64, // Extra continuous zoom on top of the level (1.0 to 1.999)
}

impl Default for MandelbrotApp {
    fn default() -> Self {
        let (tx_req, rx_req) = unbounded();
        let (tx_res, rx_res) = unbounded();
        
        let num_threads = std::thread::available_parallelism().map_or(4, |n| n.get());
        for _ in 0..num_threads {
            let rx = rx_req.clone();
            let tx = tx_res.clone();
            thread::spawn(move || {
                for key in rx {
                    let data = compute_tile(key);
                    let _ = tx.send(data);
                }
            });
        }

        Self {
            cache: HashMap::new(),
            pending: HashSet::new(),
            tx_req,
            rx_res,
            center_x: -0.5,
            center_y: 0.0,
            level: 0,
            fractional_zoom: 1.0,
        }
    }
}

impl eframe::App for MandelbrotApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Collect finished tiles
        let current_time = ctx.input(|i| i.time);
        while let Ok(data) = self.rx_res.try_recv() {
            self.pending.remove(&data.key);
            let image = egui::ColorImage::from_rgba_unmultiplied(
                [TILE_SIZE, TILE_SIZE],
                &data.pixels,
            );
            let handle = ctx.load_texture(
                format!("tile_{}_{}_{}", data.key.level, data.key.x, data.key.y),
                image,
                egui::TextureOptions::LINEAR,
            );
            self.cache.insert(data.key, RenderTile {
                texture: handle,
                insertion_time: current_time,
            });
            ctx.request_repaint(); // Important to notify the UI we have new data
        }

        // Animate fades
        ctx.request_repaint(); // continuously repaint to animate the fades

        egui::CentralPanel::default().frame(egui::Frame::none()).show(ctx, |ui| {
            let (rect, response) = ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());
            
            // Physical pixel scaling for High-DPI screens
            let ppp = ctx.pixels_per_point();
            
            // Handle Panning
            if response.dragged() {
                let drag_delta = response.drag_delta();
                // Screen pixels (logical) to math coordinates
                let scale = 2.0_f64.powi(-self.level) / self.fractional_zoom;
                let logical_screen_tile_size = TILE_SIZE as f64 / ppp as f64;
                let pixels_to_coords = (2.0 * scale) / logical_screen_tile_size;
                
                self.center_x -= (drag_delta.x as f64) * pixels_to_coords;
                self.center_y += (drag_delta.y as f64) * pixels_to_coords; 
            }

            // Handle Zooming inside the rect
            if response.hovered() {
                let scroll = ctx.input(|i| i.smooth_scroll_delta.y);
                if scroll != 0.0 {
                    // 1. Math coordinates of the mouse BEFORE zoom
                    let pointer_pos = response.hover_pos().unwrap_or(rect.center());
                    let dx = pointer_pos.x - rect.center().x; // Logical UI points offset
                    let dy = pointer_pos.y - rect.center().y;
                    
                    let scale_old = 2.0_f64.powi(-self.level) / self.fractional_zoom;
                    let logical_screen_tile_size = TILE_SIZE as f64 / ppp as f64;
                    let pixels_to_coords_old = (2.0 * scale_old) / logical_screen_tile_size;
                    
                    let mouse_math_x = self.center_x + (dx as f64) * pixels_to_coords_old;
                    let mouse_math_y = self.center_y - (dy as f64) * pixels_to_coords_old;

                    // 2. Apply zoom factor
                    let zoom_factor = (scroll * 0.005).exp();
                    self.fractional_zoom *= zoom_factor as f64;
                    
                    // Normalize zoom into integer `level` and remaining `fractional_zoom`
                    while self.fractional_zoom >= 2.0 {
                        self.level += 1;
                        self.fractional_zoom /= 2.0;
                    }
                    while self.fractional_zoom < 1.0 {
                        self.level -= 1;
                        self.fractional_zoom *= 2.0;
                    }
                    
                    // 3. Clamp minimum zoom to level 0 (initial view)
                    if self.level < 0 {
                        self.level = 0;
                        self.fractional_zoom = 1.0;
                    }

                    // 4. Adjust center_x/center_y so that the mouse_math_x/y remains exactly under the screen's dx/dy cursor position
                    let scale_new = 2.0_f64.powi(-self.level) / self.fractional_zoom;
                    let pixels_to_coords_new = (2.0 * scale_new) / logical_screen_tile_size;
                    
                    self.center_x = mouse_math_x - (dx as f64) * pixels_to_coords_new;
                    self.center_y = mouse_math_y + (dy as f64) * pixels_to_coords_new;
                }
            }

            // Draw tiles
            let painter = ui.painter_at(rect);
            
            // Adjust screen tile size according to the exact screen DPI to produce 1 physical pixel per Math calculated mandelbrot point.
            let physical_screen_tile_size = TILE_SIZE as f32; // The generated 256 physical pixel image size
            let logical_screen_tile_size = physical_screen_tile_size / ppp; // Logical points to take up
            
            let scale_factor = self.fractional_zoom as f32; // visually scale tiles up
            let render_tile_size = logical_screen_tile_size * scale_factor;
            
            // Current math scale for tile sizing
            let math_scale = 2.0_f64.powi(-self.level);
            let tile_w_coord = 2.0 * math_scale;

            // Figure out which tile indices we need
            // Math coordinates spanned by viewport
            let pixels_to_coords = (2.0 * math_scale / self.fractional_zoom) / (TILE_SIZE as f64 / ppp as f64);
            
            let x_min = self.center_x - (rect.width() as f64 * pixels_to_coords * 0.5);
            let x_max = self.center_x + (rect.width() as f64 * pixels_to_coords * 0.5);
            
            let y_min = self.center_y - (rect.height() as f64 * pixels_to_coords * 0.5);
            let y_max = self.center_y + (rect.height() as f64 * pixels_to_coords * 0.5);

            let t_x_min = (x_min / tile_w_coord).floor() as i64;
            let t_x_max = (x_max / tile_w_coord).ceil() as i64;
            
            // Note: y increases upwards natively for math
            let t_y_min = (y_min / tile_w_coord).floor() as i64;
            let t_y_max = (y_max / tile_w_coord).ceil() as i64;

            for tx in t_x_min..=t_x_max {
                for ty in t_y_min..=t_y_max {
                    let key = TileKey { x: tx, y: ty, level: self.level };
                    
                    // Draw if available, otherwise draw lower-res fallback
                    if let Some(render_tile) = self.cache.get(&key) {
                        // Calculate screen rectangle for this tile
                        let math_x = tx as f64 * tile_w_coord;
                        let math_y = ty as f64 * tile_w_coord;

                        let screen_x_center = rect.center().x + ((math_x - self.center_x) / pixels_to_coords) as f32;
                        let screen_y_center = rect.center().y - ((math_y - self.center_y) / pixels_to_coords) as f32; // Y goes down

                        let width = render_tile_size;
                        let dest_rect = egui::Rect::from_min_max(
                            egui::pos2(screen_x_center, screen_y_center),
                            egui::pos2(screen_x_center + width, screen_y_center + width),
                        );
                        
                        // Calculate fade-in opacity
                        let age = (current_time - render_tile.insertion_time) as f32;
                        let alpha = (age * 3.0).clamp(0.0, 1.0); // 333ms fade-in
                        let tint = egui::Color32::from_rgba_unmultiplied(255, 255, 255, (alpha * 255.0) as u8);
                        
                        painter.image(render_tile.texture.id(), dest_rect, egui::Rect::from_min_max(egui::pos2(0.0,0.0), egui::pos2(1.0,1.0)), tint);
                    } else {
                        // Request generation since we don't have it
                        if !self.pending.contains(&key) {
                            self.pending.insert(key);
                            let _ = self.tx_req.send(key);
                        }
                    }
                }
            }

            // Simple UI HUD
            let total_bytes = self.cache.len() * (TILE_SIZE * TILE_SIZE * 4); // 4 bytes per pixel (RGBA)
            let cache_mb = total_bytes as f64 / 1_048_576.0;
            
            // Calculate absolute magnification scale 
            // Default level = 0, fractional_zoom = 1.0 (scale factor = 1.0)
            let mag = 2.0_f64.powi(self.level) * self.fractional_zoom;

            painter.rect_filled(
                egui::Rect::from_min_max(rect.min + egui::vec2(5.0, 5.0), rect.min + egui::vec2(320.0, 95.0)),
                5.0,
                egui::Color32::from_black_alpha(150),
            );

            painter.text(
                rect.min + egui::vec2(15.0, 15.0),
                egui::Align2::LEFT_TOP,
                format!("Location:  {:.8}, {:.8}", self.center_x, self.center_y),
                egui::FontId::monospace(14.0),
                egui::Color32::WHITE,
            );
            painter.text(
                rect.min + egui::vec2(15.0, 35.0),
                egui::Align2::LEFT_TOP,
                format!("Zoom Rank: {:.2e}x (Lvl {})", mag, self.level),
                egui::FontId::monospace(14.0),
                egui::Color32::WHITE,
            );
            painter.text(
                rect.min + egui::vec2(15.0, 55.0),
                egui::Align2::LEFT_TOP,
                format!("Memory:    {} tiles ({:.1} MB)", self.cache.len(), cache_mb),
                egui::FontId::monospace(14.0),
                egui::Color32::WHITE,
            );
            painter.text(
                rect.min + egui::vec2(15.0, 75.0),
                egui::Align2::LEFT_TOP,
                format!("Pending:   {} threads working", self.pending.len()),
                egui::FontId::monospace(14.0),
                egui::Color32::from_rgb(200, 200, 200),
            );
        });
    }
}

fn main() -> eframe::Result<()> {
    // Optional check: Limit Cache size to prevent OOM
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([800.0, 600.0])
            .with_title("Mandelbrot Explorer"),
        ..Default::default()
    };
    
    eframe::run_native(
        "Mandelbrot Explorer",
        options,
        Box::new(|_cc| Box::<MandelbrotApp>::default()),
    )
}
