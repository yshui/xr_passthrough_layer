pub mod api_layer;
pub mod camera;
pub mod config;
pub mod pipeline;
pub mod steam;
pub mod utils;

use anyhow::{Context, Result, anyhow};

/// Camera image will be (size * 2, size)
pub const CAMERA_SIZE: u32 = 960;
use glam::UVec2;
#[allow(unused_imports)]
use log::info;
pub struct FrameInfo {
    pub frame: Vec<u8>,
    pub frame_time: std::time::Instant,
    pub size: UVec2,
    pub needs_postprocess: bool,
}

pub fn find_index_camera() -> Result<std::path::PathBuf> {
    let mut it = udev::Enumerator::new()?;
    it.match_subsystem("video4linux")?;
    it.match_property("ID_VENDOR_ID", "28de")?;

    let dev = it
        .scan_devices()?
        .find(|d| {
            d.properties()
                .find(|p| p.name() == "ID_MODEL_ID")
                .is_some_and(|p| p.value() == "2400")
        })
        .with_context(|| anyhow!("Index camera not found"))?;
    let devnode = dev
        .devnode()
        .with_context(|| anyhow!("Index camera cannot be accessed"))?;
    log::info!("Index camera is {}, {}", devnode.display(), dev.syspath().display());
    Ok(devnode.to_owned())
}
