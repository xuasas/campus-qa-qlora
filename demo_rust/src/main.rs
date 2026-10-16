#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod backend;
mod knowledge;

use eframe::egui;
use std::path::PathBuf;

fn main() -> eframe::Result {
    if std::env::args().any(|a| a == "--verify-backend") {
        if let Err(error) = backend::verify_backend() {
            eprintln!("{error}");
            std::process::exit(1);
        }
        return Ok(());
    }
    let options = eframe::NativeOptions {
        renderer: eframe::Renderer::Glow,
        viewport: egui::ViewportBuilder::default()
            .with_title("校园资料问答 · Q4")
            .with_inner_size([1440.0, 920.0])
            .with_min_inner_size([940.0, 720.0]),
        ..Default::default()
    };
    eframe::run_native(
        "campus-qa",
        options,
        Box::new(|cc| Ok(Box::new(app::CampusApp::new(cc)))),
    )
}

fn root() -> PathBuf {
    std::env::var_os("QLORA_ROOT").map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().expect("repository parent").to_path_buf())
}
