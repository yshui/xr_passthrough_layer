use ed25519_dalek::Signer;
use std::{io::Write, sync::Arc};

use log::warn;
use serde::{Deserialize, Serialize};

/// Because your eye and the camera is at different physical locations, it is impossible
/// to project camera view into VR space perfectly. There are trade offs approximating
/// this projection. (viewing range means things too close to you will give you double vision).
#[derive(Eq, PartialEq, Debug, Serialize, Deserialize, Clone, Copy, PartialOrd, Ord, Default)]
pub enum ProjectionMode {
    /// in this mode, we assume your eyes are at the cameras' physical location. this mode
    /// has larger viewing range, but everything will smaller to you.
    #[default]
    FromCamera,
    /// in this mode, we assume your cameras are at your eyes' physical location. everything will
    /// have the right scale in this mode, but the viewing range is smaller.
    FromEye,
}
#[derive(Debug, Serialize, Deserialize, Eq, PartialEq, Clone, Copy, PartialOrd, Ord)]
pub enum Eye {
    Left,
    Right,
}

pub const fn default_display_eye() -> Eye {
    Eye::Left
}

/// Index camera passthrough
#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    /// camera device to use. auto detect if not set
    #[serde(default)]
    pub camera_device: String,
    /// enable debug option, including:
    ///   - use trigger button to do renderdoc capture
    #[serde(default)]
    pub debug: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            camera_device: "".to_owned(),
            debug: false,
        }
    }
}

use anyhow::{Context, Result};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferToImageInfo,
        PrimaryCommandBufferAbstract as _, allocator::CommandBufferAllocator,
    },
    device::Queue,
    image::{ImageCreateInfo, ImageUsage},
    memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter},
    pipeline::cache::{PipelineCache, PipelineCacheCreateInfo, PipelineCacheData},
    sync::GpuFuture as _,
};
use xdg::BaseDirectories;

use crate::utils::DeviceExt as _;

pub fn load_config(xdg: &BaseDirectories) -> Result<Config> {
    if let Some(f) = xdg.find_config_file("index_camera_passthrough.toml") {
        let cfg = std::fs::read_to_string(f)?;
        Ok(toml::from_str(&cfg)?)
    } else {
        Ok(Default::default())
    }
}

pub struct AutoSavingPipelineCache(Arc<PipelineCache>);
impl std::ops::Deref for AutoSavingPipelineCache {
    type Target = Arc<PipelineCache>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl From<Arc<PipelineCache>> for AutoSavingPipelineCache {
    fn from(value: Arc<PipelineCache>) -> Self {
        Self(value)
    }
}

impl AutoSavingPipelineCache {
    fn save(&self) -> Result<()> {
        let data = self.0.get_data().context("get pipeline cache data")?;
        let xdg = xdg::BaseDirectories::new();

        let mut f = std::fs::OpenOptions::new()
            .truncate(true)
            .write(true)
            .create(true)
            .open(
                xdg.place_cache_file(std::path::Path::new("xr_passthrough").join("pipeline_cache"))
                    .context("create pipeline cache file")?,
            )
            .context("open pipeline cache file")?;
        let key = ed25519_dalek::SigningKey::generate(&mut rand::rng());
        let signature = key.sign(&data);
        let verifying_key = key.verifying_key();
        f.write_all(&verifying_key.as_bytes()[..])?;
        f.write_all(&signature.to_bytes()[..])?;
        f.write_all(&data)?;

        Ok(())
    }
}

impl Drop for AutoSavingPipelineCache {
    fn drop(&mut self) {
        match self.save() {
            Ok(()) => (),
            Err(e) => warn!("Failed to save pipeline cache {e:#}"),
        }
    }
}

/// Load pipeline cache from file, if file not found or fails validation, create empty
/// PipelineCache.
pub fn load_pipeline_cache(
    device: &Arc<vulkano::device::Device>,
    xdg: &BaseDirectories,
) -> Result<Arc<PipelineCache>> {
    if let Some(data) = xdg
        .find_cache_file(std::path::Path::new("xr_passthrough").join("pipeline_cache"))
        .and_then(|f| std::fs::read(f).ok())
        .and_then(|mut data| {
            let buf = &data[..];
            let verifying_key = ed25519_dalek::VerifyingKey::from_bytes(
                &buf[..ed25519_dalek::PUBLIC_KEY_LENGTH].try_into().unwrap(),
            )
            .ok()?;
            let buf = &buf[ed25519_dalek::PUBLIC_KEY_LENGTH..];
            let signature = &ed25519_dalek::Signature::from_bytes(
                buf[..ed25519_dalek::SIGNATURE_LENGTH].try_into().unwrap(),
            );
            let buf = &buf[ed25519_dalek::SIGNATURE_LENGTH..];

            verifying_key.verify_strict(buf, signature).ok()?;
            data.drain(..ed25519_dalek::PUBLIC_KEY_LENGTH + ed25519_dalek::SIGNATURE_LENGTH);
            Some(data)
        })
    {
        PipelineCache::new(
            device,
            &PipelineCacheCreateInfo {
                // SAFETY: we validated the signature
                initial_data: Some(unsafe { PipelineCacheData::new(&data) }),
                ..Default::default()
            },
        )
    } else {
        PipelineCache::new(device, &PipelineCacheCreateInfo::default())
    }
    .map_err(Into::into)
}
pub fn load_splash(
    device: &Arc<vulkano::device::Device>,
    allocator: &Arc<dyn MemoryAllocator>,
    cmdbuf_allocator: &Arc<dyn CommandBufferAllocator>,
    queue: &Arc<Queue>,
    data: &[u8],
) -> Result<Arc<vulkano::image::Image>> {
    log::debug!("loading splash");
    let img = image::load_from_memory_with_format(data, image::ImageFormat::Png)?.into_rgba8();
    let extent = [img.width(), img.height()];
    let img = img.into_raw();

    log::debug!("splash loaded");
    let vkimg = device.new_image(
        &ImageCreateInfo {
            format: vulkano::format::Format::R8G8B8A8_UNORM,
            extent: [extent[0], extent[1], 1],
            usage: ImageUsage::TRANSFER_DST | ImageUsage::TRANSFER_SRC | ImageUsage::SAMPLED,
            ..Default::default()
        },
        MemoryTypeFilter::PREFER_DEVICE,
    )?;
    let mut cmdbuf = AutoCommandBufferBuilder::primary(
        cmdbuf_allocator.clone(),
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )?;
    let buffer = Buffer::new_unsized::<[u8]>(
        allocator,
        &BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        &AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE
                | MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
        img.len() as _,
    )?;
    buffer.write()?.copy_from_slice(&img);
    cmdbuf.copy_buffer_to_image(CopyBufferToImageInfo::new(buffer, vkimg.clone()))?;
    cmdbuf
        .build()?
        .execute(queue.clone())?
        .then_signal_fence()
        .wait(None)?;

    Ok(vkimg)
}
