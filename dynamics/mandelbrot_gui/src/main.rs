use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::thread;

use crossbeam_channel::{unbounded, Receiver, Sender};
use eframe::egui;
use rand::seq::SliceRandom;

const TILE_SIZE: usize = 256;
const MAX_ITER: u32 = 1000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ColorScheme {
    Rainbow,
    Grayscale,
    BlueOrange,
    Inferno,
    Ocean,
    Neon,
    Pastel,
    Forest,
    Cyberpunk,
    Sunset,
}

impl ColorScheme {
    fn name(&self) -> &'static str {
        match self {
            ColorScheme::Rainbow => "Rainbow",
            ColorScheme::Grayscale => "Grayscale",
            ColorScheme::BlueOrange => "Blue & Orange",
            ColorScheme::Inferno => "Inferno",
            ColorScheme::Ocean => "Ocean",
            ColorScheme::Neon => "Neon",
            ColorScheme::Pastel => "Pastel",
            ColorScheme::Forest => "Forest",
            ColorScheme::Cyberpunk => "Cyberpunk",
            ColorScheme::Sunset => "Sunset",
        }
    }
}

// Rainbow (Original High Contrast)
const PALETTE_RAINBOW: [(f64, f64, f64); 16] = [
    (0.051, 0.027, 0.106), (0.098, 0.027, 0.275), (0.141, 0.039, 0.490), (0.098, 0.110, 0.667),
    (0.067, 0.216, 0.741), (0.039, 0.376, 0.745), (0.098, 0.545, 0.667), (0.224, 0.698, 0.494),
    (0.475, 0.824, 0.314), (0.741, 0.906, 0.224), (0.929, 0.933, 0.231), (0.996, 0.847, 0.220),
    (0.996, 0.682, 0.204), (0.949, 0.471, 0.180), (0.824, 0.259, 0.153), (0.620, 0.098, 0.129),
];

// Grayscale (High Contrast Bands)
const PALETTE_GRAYSCALE: [(f64, f64, f64); 16] = [
    (0.0, 0.0, 0.0), (0.1, 0.1, 0.1), (0.25, 0.25, 0.25), (0.4, 0.4, 0.4),
    (0.6, 0.6, 0.6), (0.8, 0.8, 0.8), (0.95, 0.95, 0.95), (1.0, 1.0, 1.0),
    (0.85, 0.85, 0.85), (0.7, 0.7, 0.7), (0.55, 0.55, 0.55), (0.45, 0.45, 0.45),
    (0.35, 0.35, 0.35), (0.2, 0.2, 0.2), (0.1, 0.1, 0.1), (0.05, 0.05, 0.05),
];

// Blue & Orange
const PALETTE_BLUE_ORANGE: [(f64, f64, f64); 16] = [
    (0.0, 0.05, 0.2), (0.0, 0.1, 0.4), (0.0, 0.2, 0.6), (0.1, 0.3, 0.8),
    (0.2, 0.4, 0.9), (0.4, 0.6, 1.0), (0.6, 0.4, 0.1), (0.8, 0.5, 0.0),
    (1.0, 0.6, 0.0), (1.0, 0.8, 0.2), (1.0, 0.9, 0.5), (0.8, 0.4, 0.0),
    (0.6, 0.2, 0.0), (0.4, 0.1, 0.0), (0.2, 0.05, 0.0), (0.1, 0.02, 0.0),
];

// Inferno
const PALETTE_INFERNO: [(f64, f64, f64); 16] = [
    (0.05, 0.0, 0.0), (0.15, 0.0, 0.05), (0.3, 0.0, 0.1), (0.45, 0.0, 0.2),
    (0.6, 0.1, 0.2), (0.8, 0.2, 0.1), (0.9, 0.4, 0.1), (1.0, 0.6, 0.0),
    (1.0, 0.8, 0.2), (1.0, 0.95, 0.6), (1.0, 1.0, 0.8), (0.9, 0.7, 0.3),
    (0.7, 0.4, 0.1), (0.5, 0.1, 0.0), (0.3, 0.0, 0.0), (0.1, 0.0, 0.0),
];

// Ocean
const PALETTE_OCEAN: [(f64, f64, f64); 16] = [
    (0.0, 0.05, 0.15), (0.0, 0.15, 0.3), (0.0, 0.25, 0.45), (0.0, 0.35, 0.6),
    (0.0, 0.5, 0.75), (0.1, 0.65, 0.85), (0.2, 0.8, 0.95), (0.4, 0.9, 1.0),
    (0.6, 0.95, 1.0), (0.8, 1.0, 1.0), (0.6, 0.85, 0.9), (0.3, 0.7, 0.8),
    (0.1, 0.5, 0.65), (0.0, 0.3, 0.5), (0.0, 0.15, 0.3), (0.0, 0.05, 0.1),
];

// Neon
const PALETTE_NEON: [(f64, f64, f64); 16] = [
    (0.0, 0.0, 0.0), (0.1, 0.0, 0.2), (0.3, 0.0, 0.5), (0.6, 0.0, 0.8),
    (1.0, 0.0, 1.0), (1.0, 0.2, 0.8), (1.0, 0.4, 0.4), (1.0, 0.8, 0.0),
    (0.8, 1.0, 0.0), (0.4, 1.0, 0.4), (0.0, 1.0, 0.8), (0.0, 0.8, 1.0),
    (0.0, 0.4, 1.0), (0.0, 0.2, 0.8), (0.0, 0.1, 0.4), (0.0, 0.0, 0.1),
];

// Pastel
const PALETTE_PASTEL: [(f64, f64, f64); 16] = [
    (0.9, 0.8, 0.9), (0.8, 0.7, 0.9), (0.7, 0.7, 0.95), (0.6, 0.75, 0.95),
    (0.6, 0.85, 0.95), (0.6, 0.95, 0.9), (0.7, 0.95, 0.8), (0.8, 0.95, 0.7),
    (0.95, 0.95, 0.7), (0.95, 0.85, 0.7), (0.95, 0.75, 0.7), (0.95, 0.6, 0.7),
    (0.9, 0.6, 0.8), (0.85, 0.65, 0.85), (0.85, 0.7, 0.9), (0.9, 0.75, 0.9),
];

// Forest
const PALETTE_FOREST: [(f64, f64, f64); 16] = [
    (0.0, 0.1, 0.0), (0.0, 0.2, 0.05), (0.05, 0.35, 0.1), (0.1, 0.5, 0.15),
    (0.2, 0.6, 0.2), (0.3, 0.7, 0.3), (0.4, 0.8, 0.4), (0.6, 0.9, 0.5),
    (0.8, 1.0, 0.6), (0.9, 0.95, 0.4), (0.7, 0.8, 0.3), (0.5, 0.6, 0.2),
    (0.3, 0.4, 0.15), (0.2, 0.3, 0.1), (0.1, 0.2, 0.05), (0.05, 0.1, 0.0),
];

// Cyberpunk
const PALETTE_CYBERPUNK: [(f64, f64, f64); 16] = [
    (0.0, 0.0, 0.1), (0.1, 0.0, 0.3), (0.4, 0.0, 0.6), (0.8, 0.0, 0.8),
    (1.0, 0.0, 1.0), (1.0, 0.0, 0.6), (1.0, 0.0, 0.2), (1.0, 0.2, 0.0),
    (1.0, 0.6, 0.0), (1.0, 1.0, 0.0), (0.5, 1.0, 0.5), (0.0, 1.0, 1.0),
    (0.0, 0.8, 1.0), (0.0, 0.4, 0.8), (0.0, 0.2, 0.5), (0.0, 0.0, 0.3),
];

// Sunset
const PALETTE_SUNSET: [(f64, f64, f64); 16] = [
    (0.1, 0.0, 0.2), (0.2, 0.0, 0.3), (0.4, 0.1, 0.4), (0.6, 0.1, 0.4),
    (0.8, 0.2, 0.3), (0.9, 0.3, 0.2), (1.0, 0.4, 0.1), (1.0, 0.6, 0.1),
    (1.0, 0.8, 0.2), (1.0, 0.9, 0.4), (1.0, 0.95, 0.6), (0.9, 0.7, 0.5),
    (0.7, 0.5, 0.4), (0.5, 0.3, 0.3), (0.3, 0.1, 0.2), (0.2, 0.05, 0.2),
];

fn interpolate_palette(t: f64, palette: &[(f64, f64, f64)]) -> [u8; 4] {
    let n = palette.len() as f64;
    let t = t % n;
    let idx = t.floor() as usize % palette.len();
    let frac = t - t.floor();

    let next = (idx + 1) % palette.len();
    let (r0, g0, b0) = palette[idx];
    let (r1, g1, b1) = palette[next];

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

fn palette_color(t: f64, scheme: ColorScheme) -> [u8; 4] {
    match scheme {
        ColorScheme::Rainbow => interpolate_palette(t, &PALETTE_RAINBOW),
        ColorScheme::Grayscale => interpolate_palette(t, &PALETTE_GRAYSCALE),
        ColorScheme::BlueOrange => interpolate_palette(t, &PALETTE_BLUE_ORANGE),
        ColorScheme::Inferno => interpolate_palette(t, &PALETTE_INFERNO),
        ColorScheme::Ocean => interpolate_palette(t, &PALETTE_OCEAN),
        ColorScheme::Neon => interpolate_palette(t, &PALETTE_NEON),
        ColorScheme::Pastel => interpolate_palette(t, &PALETTE_PASTEL),
        ColorScheme::Forest => interpolate_palette(t, &PALETTE_FOREST),
        ColorScheme::Cyberpunk => interpolate_palette(t, &PALETTE_CYBERPUNK),
        ColorScheme::Sunset => interpolate_palette(t, &PALETTE_SUNSET),
    }
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

fn find_period(cr: f64, ci: f64) -> Option<u32> {
    // Cardioid: period 1
    let q = (cr - 0.25) * (cr - 0.25) + ci * ci;
    if q * (q + (cr - 0.25)) <= 0.25 * ci * ci {
        return Some(1);
    }
    // Period-2 bulb
    if (cr + 1.0) * (cr + 1.0) + ci * ci <= 0.0625 {
        return Some(2);
    }
    // Check if it escapes
    let mut zr = 0.0_f64;
    let mut zi = 0.0_f64;
    for _ in 0..MAX_ITER {
        let zr2 = zr * zr;
        let zi2 = zi * zi;
        if zr2 + zi2 > 256.0 {
            return None; // escapes, not in set
        }
        zi = 2.0 * zr * zi + ci;
        zr = zr2 - zi2 + cr;
    }
    // Floyd's cycle detection
    let mut tort_r = 0.0_f64;
    let mut tort_i = 0.0_f64;
    let mut hare_r = 0.0_f64;
    let mut hare_i = 0.0_f64;
    for _ in 0..MAX_ITER {
        // Tortoise: 1 step
        let tr2 = tort_r * tort_r - tort_i * tort_i + cr;
        let ti2 = 2.0 * tort_r * tort_i + ci;
        tort_r = tr2;
        tort_i = ti2;
        // Hare: 2 steps
        for _ in 0..2 {
            let hr2 = hare_r * hare_r - hare_i * hare_i + cr;
            let hi2 = 2.0 * hare_r * hare_i + ci;
            hare_r = hr2;
            hare_i = hi2;
        }
        if (tort_r - hare_r).abs() < 1e-12 && (tort_i - hare_i).abs() < 1e-12 {
            // Find the period length
            let mut period = 1u32;
            let mut zr2 = tort_r * tort_r - tort_i * tort_i + cr;
            let mut zi2 = 2.0 * tort_r * tort_i + ci;
            while (zr2 - tort_r).abs() > 1e-12 || (zi2 - tort_i).abs() > 1e-12 {
                let nr = zr2 * zr2 - zi2 * zi2 + cr;
                let ni = 2.0 * zr2 * zi2 + ci;
                zr2 = nr;
                zi2 = ni;
                period += 1;
                if period > MAX_ITER {
                    return Some(0); // could not determine
                }
            }
            return Some(period);
        }
    }
    Some(0)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct TileKey {
    x: i64,
    y: i64,
    level: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum RenderPass {
    Res16,
    Res4,
    Full,
    SuperSample,
}

impl RenderPass {
    fn next(self, super_sample: bool) -> Option<RenderPass> {
        match self {
            RenderPass::Res16 => Some(RenderPass::Res4),
            RenderPass::Res4 => Some(RenderPass::Full),
            RenderPass::Full => if super_sample { Some(RenderPass::SuperSample) } else { None },
            RenderPass::SuperSample => None,
        }
    }
}

struct TileRequest {
    key: TileKey,
    pass: RenderPass,
    scheme: ColorScheme,
}

struct TileResult {
    key: TileKey,
    pass: RenderPass,
    pixels: Vec<u8>,
}

fn compute_tile_pass(req: TileRequest) -> TileResult {
    let scale = 2.0_f64.powi(-req.key.level);
    let tile_w_coord = 2.0;
    
    let x_min = req.key.x as f64 * tile_w_coord * scale;
    let y_max = req.key.y as f64 * tile_w_coord * scale; 
    
    let dx = (tile_w_coord * scale) / TILE_SIZE as f64;
    let dy = (tile_w_coord * scale) / TILE_SIZE as f64;

    let mut pixels = vec![0; TILE_SIZE * TILE_SIZE * 4];
    
    let step = match req.pass {
        RenderPass::Res16 => 16,
        RenderPass::Res4 => 4,
        RenderPass::Full | RenderPass::SuperSample => 1,
    };

    for py in (0..TILE_SIZE).step_by(step) {
        for px in (0..TILE_SIZE).step_by(step) {
            let color = if req.pass == RenderPass::SuperSample {
                let mut r_sum = 0.0;
                let mut g_sum = 0.0;
                let mut b_sum = 0.0;
                for sy in 0..2 {
                    for sx in 0..2 {
                        let sub_x = px as f64 + (sx as f64 + 0.5) / 2.0;
                        let sub_y = py as f64 + (sy as f64 + 0.5) / 2.0;
                        
                        let cr = x_min + (sub_x * dx);
                        let ci = y_max - (sub_y * dy);
                        
                        let color = match mandelbrot(cr, ci) {
                            Some(val) => palette_color(val * 0.15, req.scheme),
                            None => [0, 0, 0, 255],
                        };
                        
                        r_sum += color[0] as f64;
                        g_sum += color[1] as f64;
                        b_sum += color[2] as f64;
                    }
                }
                [
                    (r_sum / 4.0) as u8,
                    (g_sum / 4.0) as u8,
                    (b_sum / 4.0) as u8,
                    255,
                ]
            } else {
                // Multi-sample even for coarse passes to prevent brightness popping
                let mut r_sum = 0.0;
                let mut g_sum = 0.0;
                let mut b_sum = 0.0;
                let offsets = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)];
                for &(ox, oy) in &offsets {
                    let sub_x = px as f64 + ox * step as f64;
                    let sub_y = py as f64 + oy * step as f64;
                    let cr = x_min + (sub_x * dx);
                    let ci = y_max - (sub_y * dy);
                    let color = match mandelbrot(cr, ci) {
                        Some(val) => palette_color(val * 0.15, req.scheme),
                        None => [0, 0, 0, 255],
                    };
                    r_sum += color[0] as f64;
                    g_sum += color[1] as f64;
                    b_sum += color[2] as f64;
                }
                [
                    (r_sum / 4.0) as u8,
                    (g_sum / 4.0) as u8,
                    (b_sum / 4.0) as u8,
                    255,
                ]
            };

            for sy in 0..step {
                for sx in 0..step {
                    if py + sy < TILE_SIZE && px + sx < TILE_SIZE {
                        let idx = ((py + sy) * TILE_SIZE + (px + sx)) * 4;
                        pixels[idx] = color[0];
                        pixels[idx + 1] = color[1];
                        pixels[idx + 2] = color[2];
                        pixels[idx + 3] = color[3];
                    }
                }
            }
        }
    }

    TileResult { key: req.key, pass: req.pass, pixels }
}

struct RenderTile {
    texture: egui::TextureHandle,
    pass: RenderPass,
}

struct CaptureRequest {
    cx: f64,
    cy: f64,
    rect_width: f64,
    pixels_to_coords: f64,
    capture_w: u32,
    capture_h: u32,
    scheme: ColorScheme,
}

struct MandelbrotApp {
    cache: HashMap<TileKey, RenderTile>,
    pending: HashSet<TileKey>,
    visible_tiles: Arc<Mutex<HashSet<TileKey>>>,
    tx_req: Sender<TileRequest>,
    rx_res: Receiver<TileResult>,
    super_sampling: bool,
    color_scheme: ColorScheme,

    tx_capture: Sender<CaptureRequest>,
    rx_capture_progress: Receiver<f32>,
    capture_progress: Option<f32>,
    capture_finished_time: Option<f64>,
    queued_captures: Arc<Mutex<usize>>,

    center_x: f64,
    center_y: f64,
    level: i32,
    fractional_zoom: f64,
    hover_period: Option<u32>,
}

impl Default for MandelbrotApp {
    fn default() -> Self {
        let (tx_req, rx_req): (Sender<TileRequest>, Receiver<TileRequest>) = unbounded();
        let (tx_res, rx_res): (Sender<TileResult>, Receiver<TileResult>) = unbounded();
        let visible_tiles = Arc::new(Mutex::new(HashSet::new()));
        
        let num_threads = std::thread::available_parallelism().map_or(4, |n| n.get());
        for _ in 0..num_threads {
            let rx = rx_req.clone();
            let tx = tx_res.clone();
            let visible = visible_tiles.clone();
            thread::spawn(move || {
                for req in rx {
                    {
                        let guard = visible.lock().unwrap();
                        if !guard.contains(&req.key) {
                            continue;
                        }
                    }
                    let data = compute_tile_pass(req);
                    let _ = tx.send(data);
                }
            });
        }

        let (tx_capture, rx_capture) = unbounded::<CaptureRequest>();
        let (tx_capture_progress, rx_capture_progress) = unbounded();
        let queued_captures = Arc::new(Mutex::new(0));

        let queued_captures_clone = queued_captures.clone();
        thread::spawn(move || {
            for req in rx_capture {
                let capture_w: u32 = req.capture_w;
                let capture_h: u32 = req.capture_h;
                
                let math_w = req.rect_width * req.pixels_to_coords;
                let capture_pixels_to_coords = math_w / (capture_w as f64);

                let x_min = req.cx - (capture_w as f64 * capture_pixels_to_coords * 0.5);
                let y_max = req.cy + (capture_h as f64 * capture_pixels_to_coords * 0.5);
                let dx = capture_pixels_to_coords;
                let dy = capture_pixels_to_coords;

                let mut img = image::RgbaImage::new(capture_w, capture_h);
                let report_interval = capture_h / 100;
                
                let _ = tx_capture_progress.send(0.0);

                for py in 0..capture_h {
                    if py > 0 && py % report_interval == 0 {
                        let _ = tx_capture_progress.send(py as f32 / capture_h as f32);
                    }
                    
                    for px in 0..capture_w {
                        let mut r_sum = 0.0;
                        let mut g_sum = 0.0;
                        let mut b_sum = 0.0;

                        for sy in 0..2 {
                            for sx in 0..2 {
                                let sub_x = px as f64 + (sx as f64 + 0.5) / 2.0;
                                let sub_y = py as f64 + (sy as f64 + 0.5) / 2.0;
                                
                                let cr = x_min + (sub_x * dx);
                                let ci = y_max - (sub_y * dy);
                                
                                let color = match mandelbrot(cr, ci) {
                                    Some(val) => palette_color(val * 0.15, req.scheme),
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

                std::fs::create_dir_all("artifacts").unwrap_or_default();
                let timestamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
                let path = format!("artifacts/capture_{}.png", timestamp);
                if let Err(e) = img.save(&path) {
                    eprintln!("Failed to save screenshot: {}", e);
                } else {
                    println!("Saved HD screenshot to {}", path);
                }
                
                let _ = tx_capture_progress.send(1.0);
                {
                    let mut count = queued_captures_clone.lock().unwrap();
                    if *count > 0 {
                        *count -= 1;
                    }
                }
            }
        });

        Self {
            cache: HashMap::new(),
            pending: HashSet::new(),
            visible_tiles,
            super_sampling: false,
            color_scheme: ColorScheme::Rainbow,
            tx_req,
            rx_res,
            tx_capture,
            rx_capture_progress,
            capture_progress: None,
            capture_finished_time: None,
            queued_captures,
            center_x: -0.5,
            center_y: 0.0,
            level: 1,
            fractional_zoom: 1.0,
            hover_period: None,
        }
    }
}

fn format_scientific(val: f64) -> String {
    if val == 0.0 {
        return "0.00".to_string();
    }
    let exponent = val.abs().log10().floor() as i32;
    let mantissa = val / 10.0_f64.powi(exponent);
    if exponent == 0 {
        format!("{:.2}", mantissa)
    } else {
        format!("{:.2}e{}", mantissa, exponent)
    }
}

impl MandelbrotApp {
    fn get_fallback_tile(&self, key: TileKey) -> Option<(egui::TextureId, egui::Rect)> {
        let mut parent_key = key;
        for diff in 1..=8 {
            parent_key.level -= 1;
            parent_key.x = (parent_key.x as f64 / 2.0).floor() as i64;
            parent_key.y = (parent_key.y as f64 / 2.0).ceil() as i64;

            if let Some(parent) = self.cache.get(&parent_key) {
                let uv_width = 1.0 / (1 << diff) as f32;
                let child_rel_x = key.x - (parent_key.x << diff);
                let child_rel_y = (parent_key.y << diff) - key.y;

                let u_min = child_rel_x as f32 * uv_width;
                let v_min = child_rel_y as f32 * uv_width;
                
                let uv_rect = egui::Rect::from_min_max(
                    egui::pos2(u_min, v_min),
                    egui::pos2(u_min + uv_width, v_min + uv_width),
                );

                return Some((parent.texture.id(), uv_rect));
            }
            if parent_key.level <= 0 { break; }
        }
        None
    }
}

impl eframe::App for MandelbrotApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let dt = ctx.input(|i| i.stable_dt);
        let current_time = ctx.input(|i| i.time);
        while let Ok(data) = self.rx_res.try_recv() {
            let image = egui::ColorImage::from_rgba_unmultiplied(
                [TILE_SIZE, TILE_SIZE],
                &data.pixels,
            );
            let handle = ctx.load_texture(
                format!("tile_{}_{}_{}", data.key.level, data.key.x, data.key.y),
                image,
                egui::TextureOptions::LINEAR,
            );
            
            // Only insert into cache if it's the requested pass or higher
            let mut pass_ok = true;
            if let Some(existing) = self.cache.get(&data.key) {
                if existing.pass >= data.pass {
                    pass_ok = false;
                }
            }
            if pass_ok {
                self.cache.insert(data.key, RenderTile {
                    texture: handle,
                    pass: data.pass,
                });
            }

            if let Some(next_pass) = data.pass.next(self.super_sampling) {
                let _ = self.tx_req.send(TileRequest { key: data.key, pass: next_pass, scheme: self.color_scheme });
            } else {
                self.pending.remove(&data.key);
            }
            ctx.request_repaint();
        }

        let mut queue_count = 0;
        if let Ok(count) = self.queued_captures.lock() {
            queue_count = *count;
        }

        let mut progress_updated = false;
        while let Ok(progress) = self.rx_capture_progress.try_recv() {
            if progress >= 1.0 {
                self.capture_progress = None;
                self.capture_finished_time = Some(current_time);
            } else {
                self.capture_progress = Some(progress);
            }
            progress_updated = true;
        }
        
        if progress_updated || self.capture_progress.is_some() || !self.pending.is_empty() {
            ctx.request_repaint();
        }
        if let Some(finish_time) = self.capture_finished_time {
            if current_time - finish_time < 1.0 {
                ctx.request_repaint();
            } else {
                self.capture_finished_time = None;
            }
        }

        egui::CentralPanel::default().frame(egui::Frame::none()).show(ctx, |ui| {
            let (rect, response) = ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());
            let ppp = ctx.pixels_per_point();
            
            if response.dragged() {
                let drag_delta = response.drag_delta();
                let scale = 2.0_f64.powi(-self.level) / self.fractional_zoom;
                let logical_screen_tile_size = TILE_SIZE as f64 / ppp as f64;
                let pixels_to_coords = (2.0 * scale) / logical_screen_tile_size;
                
                self.center_x -= (drag_delta.x as f64) * pixels_to_coords;
                self.center_y += (drag_delta.y as f64) * pixels_to_coords; 
            }

            if response.hovered() {
                let scroll = ctx.input(|i| i.smooth_scroll_delta.y);
                if scroll != 0.0 {
                    let pointer_pos = response.hover_pos().unwrap_or(rect.center());
                    let dx = pointer_pos.x - rect.center().x;
                    let dy = pointer_pos.y - rect.center().y;
                    
                    let original_level = self.level;
                    let original_fractional_zoom = self.fractional_zoom;

                    let scale_old = 2.0_f64.powi(-self.level) / self.fractional_zoom;
                    let logical_screen_tile_size = TILE_SIZE as f64 / ppp as f64;
                    let pixels_to_coords_old = (2.0 * scale_old) / logical_screen_tile_size;
                    
                    let mouse_math_x = self.center_x + (dx as f64) * pixels_to_coords_old;
                    let mouse_math_y = self.center_y - (dy as f64) * pixels_to_coords_old;

                    let mut zoom_factor = (scroll * 0.005).exp();
                    
                    // Limit zoom speed
                    let max_log2_per_sec = 8.0;
                    let max_log2_delta = max_log2_per_sec * dt;
                    let actual_log2_delta = zoom_factor.log2();
                    if actual_log2_delta > max_log2_delta {
                        zoom_factor = max_log2_delta.exp2();
                    } else if actual_log2_delta < -max_log2_delta {
                        zoom_factor = (-max_log2_delta).exp2();
                    }

                    self.fractional_zoom *= zoom_factor as f64;
                    
                    while self.fractional_zoom >= 2.0 {
                        self.level += 1;
                        self.fractional_zoom /= 2.0;
                    }
                    while self.fractional_zoom < 1.0 {
                        self.level -= 1;
                        self.fractional_zoom *= 2.0;
                    }
                    
                    if self.level < 0 {
                        self.level = original_level;
                        self.fractional_zoom = original_fractional_zoom;
                    } else {
                        let scale_new = 2.0_f64.powi(-self.level) / self.fractional_zoom;
                        let pixels_to_coords_new = (2.0 * scale_new) / logical_screen_tile_size;
                        
                        self.center_x = mouse_math_x - (dx as f64) * pixels_to_coords_new;
                        self.center_y = mouse_math_y + (dy as f64) * pixels_to_coords_new;
                    }
                }
            }

            let painter = ui.painter_at(rect);
            
            let settings_rect = egui::Rect::from_min_max(rect.min + egui::vec2(285.0, 15.0), rect.min + egui::vec2(520.0, 55.0));
            painter.rect_filled(
                settings_rect,
                12.0, // macOS rounding
                egui::Color32::from_black_alpha(110), // frosted glass
            );
            painter.rect_stroke(
                settings_rect,
                12.0,
                egui::Stroke::new(1.0, egui::Color32::from_white_alpha(30)),
            );

            egui::Window::new("Settings")
                .fixed_pos(settings_rect.min + egui::vec2(10.0, 10.0))
                .collapsible(false)
                .resizable(false)
                .title_bar(false)
                .frame(egui::Frame::none()) // Transparent, we drew our own frosted rect!
                .show(ctx, |ui| {
                    let mut changed_rendering = false;
                    ui.horizontal(|ui| {
                        let schemes = [
                            ColorScheme::Rainbow, ColorScheme::Grayscale, ColorScheme::BlueOrange,
                            ColorScheme::Inferno, ColorScheme::Ocean, ColorScheme::Neon,
                            ColorScheme::Pastel, ColorScheme::Forest, ColorScheme::Cyberpunk,
                            ColorScheme::Sunset,
                        ];
                        
                        // Force modern proportional font for UI body
                        ui.style_mut().text_styles.insert(
                            egui::TextStyle::Button,
                            egui::FontId::proportional(14.0),
                        );
                        ui.style_mut().text_styles.insert(
                            egui::TextStyle::Body,
                            egui::FontId::proportional(14.0),
                        );

                        egui::ComboBox::from_id_source("ColorSchemeCombo")
                            .selected_text(self.color_scheme.name())
                            .show_ui(ui, |ui| {
                                for scheme in schemes {
                                    ui.horizontal(|ui| {
                                        // Draw gradient
                                        let (rect, _response) = ui.allocate_exact_size(egui::vec2(60.0, 16.0), egui::Sense::hover());
                                        let n_steps = 10;
                                        for i in 0..n_steps {
                                            let t = (i as f32 / n_steps as f32) * 16.0; // t range for colors
                                            let c = palette_color(t as f64, scheme);
                                            let color32 = egui::Color32::from_rgba_unmultiplied(c[0], c[1], c[2], c[3]);
                                            let w = rect.width() / n_steps as f32;
                                            ui.painter_at(rect).rect_filled(
                                                egui::Rect::from_min_max(
                                                    rect.min + egui::vec2(w * i as f32, 0.0),
                                                    rect.min + egui::vec2(w * (i + 1) as f32, rect.height())
                                                ),
                                                egui::Rounding::ZERO,
                                                color32
                                            );
                                        }

                                        if ui.selectable_value(&mut self.color_scheme, scheme, scheme.name()).changed() {
                                            changed_rendering = true;
                                        }
                                    });
                                }
                            });
                    });

                    if ui.checkbox(&mut self.super_sampling, "Super-sampling").changed() {
                        changed_rendering = true;
                    }

                    if changed_rendering {
                        self.cache.clear();
                        self.pending.clear();
                    }
                });
            
            let physical_screen_tile_size = TILE_SIZE as f32;
            let logical_screen_tile_size = physical_screen_tile_size / ppp;
            let scale_factor = self.fractional_zoom as f32;
            let render_tile_size = logical_screen_tile_size * scale_factor;
            
            let math_scale = 2.0_f64.powi(-self.level);
            let tile_w_coord = 2.0 * math_scale;

            let pixels_to_coords = (2.0 * math_scale / self.fractional_zoom) / (TILE_SIZE as f64 / ppp as f64);
            
            let x_min = self.center_x - (rect.width() as f64 * pixels_to_coords * 0.5);
            let x_max = self.center_x + (rect.width() as f64 * pixels_to_coords * 0.5);
            let y_min = self.center_y - (rect.height() as f64 * pixels_to_coords * 0.5);
            let y_max = self.center_y + (rect.height() as f64 * pixels_to_coords * 0.5);

            let t_x_min = (x_min / tile_w_coord).floor() as i64;
            let t_x_max = (x_max / tile_w_coord).ceil() as i64;
            let t_y_min = (y_min / tile_w_coord).floor() as i64;
            let t_y_max = (y_max / tile_w_coord).ceil() as i64;

            let mut new_visible = HashSet::new();
            for tx in t_x_min..=t_x_max {
                for ty in t_y_min..=t_y_max {
                    new_visible.insert(TileKey { x: tx, y: ty, level: self.level });
                }
            }
            {
                let mut guard = self.visible_tiles.lock().unwrap();
                *guard = new_visible.clone();
            }
            
            self.pending.retain(|k| new_visible.contains(k));

            let mut tile_requests = Vec::new();

            for tx in t_x_min..=t_x_max {
                for ty in t_y_min..=t_y_max {
                    let key = TileKey { x: tx, y: ty, level: self.level };
                    let math_x = tx as f64 * tile_w_coord;
                    let math_y = ty as f64 * tile_w_coord;

                    let screen_x_center = rect.center().x + ((math_x - self.center_x) / pixels_to_coords) as f32;
                    let screen_y_center = rect.center().y - ((math_y - self.center_y) / pixels_to_coords) as f32;

                    let width = render_tile_size;
                    let dest_rect = egui::Rect::from_min_max(
                        egui::pos2(screen_x_center, screen_y_center),
                        egui::pos2(screen_x_center + width, screen_y_center + width),
                    );

                    if let Some(render_tile) = self.cache.get(&key) {
                        painter.image(render_tile.texture.id(), dest_rect, egui::Rect::from_min_max(egui::pos2(0.0,0.0), egui::pos2(1.0,1.0)), egui::Color32::WHITE);
                        // Request next pass if not done
                        if !self.pending.contains(&key) {
                            if let Some(next_pass) = render_tile.pass.next(self.super_sampling) {
                                self.pending.insert(key);
                                tile_requests.push(TileRequest { key, pass: next_pass, scheme: self.color_scheme });
                            }
                        }
                    } else {
                        let has_fallback = if let Some((tex_id, uv_rect)) = self.get_fallback_tile(key) {
                            painter.image(tex_id, dest_rect, uv_rect, egui::Color32::WHITE);
                            true
                        } else {
                            false
                        };

                        if !self.pending.contains(&key) {
                            self.pending.insert(key);
                            // If we have a fallback, skip coarse passes to avoid momentary quality loss
                            let start_pass = if has_fallback { RenderPass::Full } else { RenderPass::Res16 };
                            tile_requests.push(TileRequest { key, pass: start_pass, scheme: self.color_scheme });
                        }
                    }
                }
            }
            
            if !tile_requests.is_empty() {
                let mut rng = rand::rng();
                tile_requests.shuffle(&mut rng);
                tile_requests.sort_by_key(|req| req.pass);
                for req in tile_requests {
                    let _ = self.tx_req.send(req);
                }
            }

            // Periodicity calculation from hover position
            let pixels_to_coords_cur = pixels_to_coords;
            if let Some(hover_pos) = response.hover_pos() {
                let dx_hover = hover_pos.x - rect.center().x;
                let dy_hover = hover_pos.y - rect.center().y;
                let cr = self.center_x + (dx_hover as f64) * pixels_to_coords_cur;
                let ci = self.center_y - (dy_hover as f64) * pixels_to_coords_cur;
                self.hover_period = find_period(cr, ci);
            } else {
                self.hover_period = None;
            }

            let total_bytes = self.cache.len() * (TILE_SIZE * TILE_SIZE * 4);
            let cache_mb = total_bytes as f64 / 1_048_576.0;
            let mag_linear = 2.0_f64.powi(self.level) * self.fractional_zoom;
            let mag_area = mag_linear.powi(2) / 4.0;
            let mag_log2 = mag_area.log2();

            // macOS HUD Style
            let hud_height = if self.hover_period.is_some() { 135.0 } else { 115.0 };
            let popup_rect = egui::Rect::from_min_max(rect.min + egui::vec2(15.0, 15.0), rect.min + egui::vec2(270.0, 15.0 + hud_height));
            painter.rect_filled(
                popup_rect,
                12.0,
                egui::Color32::from_black_alpha(110),
            );
            painter.rect_stroke(
                popup_rect,
                12.0,
                egui::Stroke::new(1.0, egui::Color32::from_white_alpha(30)),
            );

            let font = egui::FontId::proportional(14.0);
            painter.text(
                popup_rect.min + egui::vec2(15.0, 15.0),
                egui::Align2::LEFT_TOP,
                format!("Location:  {:.8}, {:.8}", self.center_x, self.center_y),
                font.clone(),
                egui::Color32::WHITE,
            );
            painter.text(
                popup_rect.min + egui::vec2(15.0, 35.0),
                egui::Align2::LEFT_TOP,
                format!("Zoom (log2): {:.2}", mag_log2),
                font.clone(),
                egui::Color32::WHITE,
            );
            painter.text(
                popup_rect.min + egui::vec2(15.0, 55.0),
                egui::Align2::LEFT_TOP,
                format!("Memory:   {} tiles ({:.1} MB)", self.cache.len(), cache_mb),
                font.clone(),
                egui::Color32::WHITE,
            );
            painter.text(
                popup_rect.min + egui::vec2(15.0, 75.0),
                egui::Align2::LEFT_TOP,
                format!("Engines:    {} threads processing", self.pending.len()),
                font.clone(),
                egui::Color32::from_rgb(200, 200, 200),
            );
            if let Some(period) = self.hover_period {
                let period_text = if period == 0 { "Period: unknown".to_string() } else { format!("Period: {}", period) };
                painter.text(
                    popup_rect.min + egui::vec2(15.0, 95.0),
                    egui::Align2::LEFT_TOP,
                    period_text,
                    font.clone(),
                    egui::Color32::from_rgb(100, 255, 100),
                );
            }

            // Scale Bar Wrapper (macOS Overlay Style)
            let scale_bar_width_px = 100.0;
            let math_width = (scale_bar_width_px as f64) * pixels_to_coords;
            let bar_y = rect.max.y - 30.0;
            let bar_x_end = rect.max.x - 20.0;
            let bar_x_start = bar_x_end - scale_bar_width_px;

            let scale_rect = egui::Rect::from_min_max(
                egui::pos2(bar_x_start - 10.0, bar_y - 25.0),
                egui::pos2(bar_x_end + 10.0, bar_y + 10.0)
            );
            painter.rect_filled(
                scale_rect,
                10.0,
                egui::Color32::from_black_alpha(110),
            );
            painter.rect_stroke(
                scale_rect,
                10.0,
                egui::Stroke::new(1.0, egui::Color32::from_white_alpha(30)),
            );

            painter.line_segment(
                [egui::pos2(bar_x_start, bar_y), egui::pos2(bar_x_end, bar_y)],
                egui::Stroke::new(2.0, egui::Color32::WHITE),
            );
            painter.line_segment(
                [egui::pos2(bar_x_start, bar_y - 4.0), egui::pos2(bar_x_start, bar_y + 4.0)],
                egui::Stroke::new(2.0, egui::Color32::WHITE),
            );
            painter.line_segment(
                [egui::pos2(bar_x_end, bar_y - 4.0), egui::pos2(bar_x_end, bar_y + 4.0)],
                egui::Stroke::new(2.0, egui::Color32::WHITE),
            );
            painter.text(
                egui::pos2(bar_x_start + 50.0, bar_y - 8.0),
                egui::Align2::CENTER_BOTTOM,
                format_scientific(math_width),
                egui::FontId::proportional(12.0),
                egui::Color32::WHITE,
            );

            // Screenshot Button (macOS Squircle Toolbar Style)
            let capture_center = egui::pos2(rect.max.x - 35.0, rect.min.y + 35.0);
            let capture_radius = 20.0;
            let capture_rect = egui::Rect::from_center_size(capture_center, egui::vec2(capture_radius * 2.0, capture_radius * 2.0));
            
            let interact = ui.interact(capture_rect, ui.id().with("capture_btn"), egui::Sense::click());
            
            let mut bg_color = egui::Color32::from_black_alpha(110);
            if self.capture_progress.is_some() {
                bg_color = egui::Color32::from_black_alpha(200);
            } else if interact.is_pointer_button_down_on() {
                bg_color = egui::Color32::from_white_alpha(50);
            } else if interact.hovered() {
                bg_color = egui::Color32::from_white_alpha(20);
            }

            // Draw Squircle
            painter.rect_filled(capture_rect, 10.0, bg_color);
            painter.rect_stroke(
                capture_rect,
                10.0,
                egui::Stroke::new(1.0, egui::Color32::from_white_alpha(30)),
            );
            
            painter.text(
                capture_center,
                egui::Align2::CENTER_CENTER,
                "📷", // macOS doesn't support SF Symbols in raw text rendering here natively unfortunately, but the emoji suffices
                egui::FontId::proportional(22.0),
                egui::Color32::WHITE,
            );

            if let Some(progress) = self.capture_progress {
                // Squircle fill from bottom
                let fill_height = capture_rect.height() * progress;
                let fill_rect = egui::Rect::from_min_max(
                    egui::pos2(capture_rect.min.x, capture_rect.max.y - fill_height),
                    capture_rect.max,
                );
                painter.rect_filled(
                    fill_rect,
                    10.0,
                    egui::Color32::from_rgba_unmultiplied(100, 255, 100, 80),
                );
            }

            if let Some(finish_time) = self.capture_finished_time {
                let age = (current_time - finish_time) as f32;
                if age < 1.0 {
                    let alpha = (1.0 - age).clamp(0.0, 1.0);
                    painter.rect_stroke(
                        capture_rect.expand(age * 5.0),
                        10.0 + age * 5.0,
                        egui::Stroke::new(2.0, egui::Color32::from_rgba_unmultiplied(255, 255, 255, (255.0 * alpha) as u8)),
                    );
                }
            }

            if interact.clicked() {
                if self.capture_progress.is_none() && queue_count == 0 {
                    self.capture_progress = Some(0.0);
                }
                
                {
                    let mut count = self.queued_captures.lock().unwrap();
                    *count += 1;
                }
                
                let cx = self.center_x;
                let cy = self.center_y;
                let rect_width = rect.width() as f64;

                let _ = self.tx_capture.send(CaptureRequest {
                    cx,
                    cy,
                    rect_width,
                    pixels_to_coords,
                    capture_w: 4536,
                    capture_h: 2946,
                    scheme: self.color_scheme,
                });
            }

            if queue_count > 0 {
                let text = if queue_count == 1 { "1 pending screenshot" } else { &format!("{} pending screenshots", queue_count) };
                
                let pill_rect = egui::Rect::from_min_max(
                    capture_center + egui::vec2(-190.0, -10.0),
                    capture_center + egui::vec2(-35.0, 10.0)
                );
                painter.rect_filled(
                    pill_rect,
                    10.0,
                    egui::Color32::from_black_alpha(110),
                );
                painter.rect_stroke(
                    pill_rect,
                    10.0,
                    egui::Stroke::new(1.0, egui::Color32::from_white_alpha(30)),
                );

                painter.text(
                    pill_rect.center(),
                    egui::Align2::CENTER_CENTER,
                    text,
                    egui::FontId::proportional(13.0),
                    egui::Color32::WHITE,
                );
            }
        });
    }
}

// macOS Sequoia overall egui theme setup
fn setup_custom_styles(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();
    
    // Smooth macOS corners
    style.visuals.window_rounding = egui::Rounding::same(12.0);
    style.visuals.menu_rounding = egui::Rounding::same(8.0);
    style.visuals.widgets.noninteractive.rounding = egui::Rounding::same(8.0);
    style.visuals.widgets.inactive.rounding = egui::Rounding::same(8.0);
    style.visuals.widgets.hovered.rounding = egui::Rounding::same(8.0);
    style.visuals.widgets.active.rounding = egui::Rounding::same(8.0);
    style.visuals.widgets.open.rounding = egui::Rounding::same(8.0);
    
    // Window appearance
    style.visuals.window_shadow.blur = 24.0;
    style.visuals.window_shadow.spread = 2.0;
    style.visuals.window_shadow.color = egui::Color32::from_black_alpha(60);
    
    // Make popups borderless mimicking frosted glass overlays
    style.visuals.window_stroke = egui::Stroke::new(1.0, egui::Color32::from_white_alpha(30));
    style.visuals.window_fill = egui::Color32::from_black_alpha(150);
    
    ctx.set_style(style);
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([800.0, 600.0])
            .with_title("Mandelbrot Explorer"),
        ..Default::default()
    };
    
    eframe::run_native(
        "Mandelbrot Explorer",
        options,
        Box::new(|cc| {
            setup_custom_styles(&cc.egui_ctx);
            Box::<MandelbrotApp>::default()
        }),
    )
}
