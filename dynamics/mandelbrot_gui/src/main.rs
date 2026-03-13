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
        for px in 0..TILE_SIZE {
            let mut r_sum = 0.0;
            let mut g_sum = 0.0;
            let mut b_sum = 0.0;

            // 2x2 Supersampling
            for sy in 0..2 {
                for sx in 0..2 {
                    let sub_x = px as f64 + (sx as f64 + 0.5) / 2.0;
                    let sub_y = py as f64 + (sy as f64 + 0.5) / 2.0;
                    
                    let cr = x_min + (sub_x * dx);
                    let ci = y_max - (sub_y * dy);
                    
                    let color = match mandelbrot(cr, ci) {
                        Some(val) => palette_color(val * 0.15),
                        None => [0, 0, 0, 255],
                    };
                    
                    r_sum += color[0] as f64;
                    g_sum += color[1] as f64;
                    b_sum += color[2] as f64;
                }
            }
            
            pixels.push((r_sum / 4.0) as u8);
            pixels.push((g_sum / 4.0) as u8);
            pixels.push((b_sum / 4.0) as u8);
            pixels.push(255);
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
    
    // Screenshot telemetry
    rx_progress: Receiver<f32>,
    capture_progress: Option<f32>,
    capture_finished_time: Option<f64>,

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

        let (tx_progress, rx_progress) = unbounded();

        Self {
            cache: HashMap::new(),
            pending: HashSet::new(),
            tx_req,
            rx_res,
            rx_progress,
            capture_progress: None,
            capture_finished_time: None,
            center_x: -0.5,
            center_y: 0.0,
            level: 1,
            fractional_zoom: 1.0,
        }
    }
}

fn format_scientific(val: f64) -> String {
    if val >= 0.001 && val < 1_000_000.0 {
        // Standard formatting for intermediate numbers
        if val == val.floor() {
            return format!("{:.0}", val);
        } else {
            return format!("{:.2}", val);
        }
    }

    // Mathematical zero protection
    if val == 0.0 {
        return "0".to_string();
    }

    let exponent = val.abs().log10().floor() as i32;
    let mantissa = val / 10.0_f64.powi(exponent);

    let superscript_digits = ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹'];
    
    let mut exp_str = String::new();
    if exponent < 0 {
        exp_str.push('⁻');
    }
    
    for c in exponent.abs().to_string().chars() {
        if let Some(digit) = c.to_digit(10) {
            exp_str.push(superscript_digits[digit as usize]);
        }
    }

    format!("{:.2} × 10{}", mantissa, exp_str)
}

impl MandelbrotApp {
    fn get_fallback_tile(&self, key: TileKey) -> Option<(egui::TextureId, egui::Rect)> {
        let mut parent_key = key;
        for diff in 1..=8 {
            parent_key.level -= 1;
            parent_key.x = (parent_key.x as f64 / 2.0).floor() as i64;
            parent_key.y = (parent_key.y as f64 / 2.0).ceil() as i64;

            if let Some(parent) = self.cache.get(&parent_key) {
                // Find where the child is inside the parent
                // The parent covers twice the area per level difference.
                // Child width in parent UV space is 1.0 / 2^diff
                let uv_width = 1.0 / (1 << diff) as f32;

                // How many units is the child away from the parent's top-left corner?
                // Notice that y increases upwards, so the parent top-left is actually (parent.x, parent.y)
                let child_rel_x = key.x - (parent_key.x << diff);
                
                // In math coords y is up, but in UV y is down.
                // UV(0,0) corresponds to parent max Y
                // parent_key.y represents the top edge of the parent tile.
                let child_rel_y = (parent_key.y << diff) - key.y;

                let u_min = child_rel_x as f32 * uv_width;
                let v_min = child_rel_y as f32 * uv_width;
                
                let uv_rect = egui::Rect::from_min_max(
                    egui::pos2(u_min, v_min),
                    egui::pos2(u_min + uv_width, v_min + uv_width),
                );

                return Some((parent.texture.id(), uv_rect));
            }
            if parent_key.level <= 0 {
                break;
            }
        }
        None
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

        // Drain progress updates from background thread
        let mut progress_updated = false;
        while let Ok(progress) = self.rx_progress.try_recv() {
            if progress >= 1.0 {
                self.capture_progress = None;
                self.capture_finished_time = Some(current_time);
            } else {
                self.capture_progress = Some(progress);
            }
            progress_updated = true;
        }
        
        // Ensure UI repaints continuously if rendering is active or finishing animation
        if progress_updated || self.capture_progress.is_some() {
            ctx.request_repaint();
        }
        if let Some(finish_time) = self.capture_finished_time {
            if current_time - finish_time < 1.0 {
                ctx.request_repaint(); // Keep animating fade-out
            } else {
                self.capture_finished_time = None;
            }
        }

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
                    
                    let original_level = self.level;
                    let original_fractional_zoom = self.fractional_zoom;

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
                        self.level = original_level;
                        self.fractional_zoom = original_fractional_zoom;
                    } else {
                        // 4. Adjust center_x/center_y so that the mouse_math_x/y remains exactly under the screen's dx/dy cursor position
                        let scale_new = 2.0_f64.powi(-self.level) / self.fractional_zoom;
                        let pixels_to_coords_new = (2.0 * scale_new) / logical_screen_tile_size;
                        
                        self.center_x = mouse_math_x - (dx as f64) * pixels_to_coords_new;
                        self.center_y = mouse_math_y + (dy as f64) * pixels_to_coords_new;
                    }

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

                    let mut drawn_full_opacity = false;

                    // Draw if available, otherwise draw lower-res fallback
                    if let Some(render_tile) = self.cache.get(&key) {
                        painter.image(render_tile.texture.id(), dest_rect, egui::Rect::from_min_max(egui::pos2(0.0,0.0), egui::pos2(1.0,1.0)), egui::Color32::WHITE);
                    } else {
                        // Request generation since we don't have it
                        if !self.pending.contains(&key) {
                            self.pending.insert(key);
                            let _ = self.tx_req.send(key);
                        }
                        
                        // Draw fallback if available natively (no crossfade)
                        if let Some((tex_id, uv_rect)) = self.get_fallback_tile(key) {
                            painter.image(tex_id, dest_rect, uv_rect, egui::Color32::WHITE);
                        }
                    }
                }
            }

            // Simple UI HUD
            let total_bytes = self.cache.len() * (TILE_SIZE * TILE_SIZE * 4); // 4 bytes per pixel (RGBA)
            let cache_mb = total_bytes as f64 / 1_048_576.0;
            
            // Calculate absolute magnification scale 
            // Default level = 1, fractional_zoom = 1.0 (scale factor = 2.0 based on level 0)
            let mag_linear = 2.0_f64.powi(self.level) * self.fractional_zoom;
            // Area magnification is roughly the square of the linear scaling
            let mag_area = mag_linear.powi(2) / 4.0; // Normalized so level 1 is 1x Area

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
                format!("Zoom: {}", format_scientific(mag_area)),
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

            // Draw Scale Bar (100 physical pixels wide equivalent) bottom right
            let scale_bar_width_px = 100.0;
            let screen_pixels_to_coords = pixels_to_coords; 
            let math_width = (scale_bar_width_px as f64) * screen_pixels_to_coords;

            let bar_y = rect.max.y - 30.0;
            let bar_x_end = rect.max.x - 20.0;
            let bar_x_start = bar_x_end - scale_bar_width_px;

            // Draw horizontal bar
            painter.line_segment(
                [egui::pos2(bar_x_start, bar_y), egui::pos2(bar_x_end, bar_y)],
                egui::Stroke::new(2.0, egui::Color32::WHITE),
            );
            
            // Draw end ticks
            painter.line_segment(
                [egui::pos2(bar_x_start, bar_y - 4.0), egui::pos2(bar_x_start, bar_y + 4.0)],
                egui::Stroke::new(2.0, egui::Color32::WHITE),
            );
            painter.line_segment(
                [egui::pos2(bar_x_end, bar_y - 4.0), egui::pos2(bar_x_end, bar_y + 4.0)],
                egui::Stroke::new(2.0, egui::Color32::WHITE),
            );
            
            // Draw scale text right above it
            painter.text(
                egui::pos2(bar_x_start + 50.0, bar_y - 8.0),
                egui::Align2::CENTER_BOTTOM,
                format_scientific(math_width),
                egui::FontId::monospace(12.0),
                egui::Color32::WHITE,
            );

            // Screenshot Button (Top Right)
            let capture_center = egui::pos2(rect.max.x - 30.0, rect.min.y + 30.0);
            let capture_radius = 20.0;
            let capture_rect = egui::Rect::from_center_size(capture_center, egui::vec2(capture_radius * 2.0, capture_radius * 2.0));
            
            let interact = ui.interact(capture_rect, ui.id().with("capture_btn"), egui::Sense::click());
            
            // Interaction colors (glow and press)
            let mut bg_color = egui::Color32::from_black_alpha(150);
            if self.capture_progress.is_some() {
                // Dim out while rendering
                bg_color = egui::Color32::from_black_alpha(200);
            } else if interact.is_pointer_button_down_on() {
                // Pressed flash
                bg_color = egui::Color32::from_white_alpha(150);
            } else if interact.hovered() {
                // Hover glow
                bg_color = egui::Color32::from_black_alpha(80);
            }

            // Draw button background and text
            painter.circle_filled(capture_center, capture_radius, bg_color);
            painter.text(
                capture_center,
                egui::Align2::CENTER_CENTER,
                "📷",
                egui::FontId::proportional(22.0),
                if interact.is_pointer_button_down_on() { egui::Color32::BLACK } else { egui::Color32::WHITE },
            );

            // Draw active circular progress bar
            if let Some(progress) = self.capture_progress {
                // Let's manually draw a polyline arc
                let mut points = Vec::new();
                let num_segments = 32;
                // -PI/2 is the "top" (12 o'clock)
                let start_angle = -std::f32::consts::PI / 2.0;
                let end_angle = start_angle + (progress * 2.0 * std::f32::consts::PI);
                
                for i in 0..=num_segments {
                    let t = i as f32 / num_segments as f32;
                    let angle = start_angle + (end_angle - start_angle) * t;
                    let p = capture_center + egui::vec2(angle.cos() * capture_radius, angle.sin() * capture_radius);
                    points.push(p);
                }
                
                painter.add(egui::Shape::line(
                    points,
                    egui::Stroke::new(3.0, egui::Color32::GREEN),
                ));
            }

            // Draw post-completion flash animation
            if let Some(finish_time) = self.capture_finished_time {
                let age = (current_time - finish_time) as f32;
                if age < 1.0 {
                    let alpha = (1.0 - age).clamp(0.0, 1.0);
                    painter.circle_stroke(
                        capture_center,
                        capture_radius,
                        egui::Stroke::new(3.0, egui::Color32::from_rgba_unmultiplied(255, 255, 255, (255.0 * alpha) as u8)),
                    );
                }
            }

            if interact.clicked() && self.capture_progress.is_none() {
                // Initialize progress to render UI instantly
                self.capture_progress = Some(0.0);
                let tx_prog = self.rx_progress.clone(); // Can't clone receiver directly, we need a sender copy
                
                // Oops, we need the sender handle specifically.
                // We'll pass `self.rx_progress` into a structure or just pass the sender during initialization.
                // Since `tx_progress` is dropped inside `default()`, we must refactor `MandelbrotApp` to store `tx_progress` 
                // temporarily, or just create a new one every capture click!
                            
                let (tx_job_prog, rx_job_prog) = unbounded();
                self.rx_progress = rx_job_prog; // Swap the active receiver for this job

                // Calculate parameters for 4K capture
                // 3840x2160, preserving the current view center and physical zoom ratio
                let capture_w: u32 = 3840;
                let capture_h: u32 = 2160;
                
                // Real width in math coords 
                let math_w = rect.width() as f64 * pixels_to_coords;
                // Ratio to convert 1080p width to 4k width math representation
                let capture_pixels_to_coords = math_w / (capture_w as f64);

                let cx = self.center_x;
                let cy = self.center_y;

                std::thread::spawn(move || {
                    let mut img = image::RgbaImage::new(capture_w, capture_h);
                    
                    let x_min = cx - (capture_w as f64 * capture_pixels_to_coords * 0.5);
                    let y_max = cy + (capture_h as f64 * capture_pixels_to_coords * 0.5);
                    let dx = capture_pixels_to_coords;
                    let dy = capture_pixels_to_coords;

                    // Compute 4K
                    let report_interval = capture_h / 100; // Report 1% increments
                    for py in 0..capture_h {
                        if py % report_interval == 0 {
                            let _ = tx_job_prog.send(py as f32 / capture_h as f32);
                        }
                        
                        for px in 0..capture_w {
                            let mut r_sum = 0.0;
                            let mut g_sum = 0.0;
                            let mut b_sum = 0.0;

                            // Include 2x2 MSAA on the 4K render itself for ultra high fidelity
                            for sy in 0..2 {
                                for sx in 0..2 {
                                    let sub_x = px as f64 + (sx as f64 + 0.5) / 2.0;
                                    let sub_y = py as f64 + (sy as f64 + 0.5) / 2.0;
                                    
                                    let cr = x_min + (sub_x * dx);
                                    let ci = y_max - (sub_y * dy);
                                    
                                    let color = match mandelbrot(cr, ci) {
                                        Some(val) => palette_color(val * 0.15),
                                        None => [0, 0, 0, 255],
                                    };
                                    r_sum += color[0] as f64;
                                    g_sum += color[1] as f64;
                                    b_sum += color[2] as f64;
                                }
                            }

                            img.put_pixel(px, py, image::Rgba([
                                (r_sum / 4.0) as u8, 
                                (g_sum / 4.0) as u8, 
                                (b_sum / 4.0) as u8, 
                                255
                            ]));
                        }
                    }

                    // Save to artifacts
                    std::fs::create_dir_all("artifacts").unwrap_or_default();
                    let timestamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
                    let path = format!("artifacts/capture_{}.png", timestamp);
                    if let Err(e) = img.save(&path) {
                        eprintln!("Failed to save screenshot: {}", e);
                    } else {
                        println!("Saved 4K screenshot to {}", path);
                    }
                    
                    let _ = tx_job_prog.send(1.0); // 100% complete
                });
            }
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
