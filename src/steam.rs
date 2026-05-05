use std::io::prelude::*;
use std::os::fd::AsRawFd;
use std::time::Duration;
use std::{ffi::OsStr, fs::File};

use anyhow::{Context, Result, anyhow, bail};
use flate2::read::ZlibDecoder;
use nix::{errno::Errno, fcntl::OFlag, ioctl_readwrite_buf, sys::stat::Mode};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug)]
pub struct Extrinsics {
    /// Offset of the camera from Hmd
    pub position: [f64; 3],
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug)]
pub struct Distort {
    pub coeffs: [f64; 4],
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug)]
pub struct Intrinsics {
    /// Optical center X
    pub center_x: f64,
    /// Optical center Y
    pub center_y: f64,
    /// X focal length in device pixels
    pub focal_x: f64,
    /// Y focal length in device pixels
    pub focal_y: f64,
    /// Height of the camera output in pixels
    pub height: f64,
    /// Width of the camera output in pixels
    pub width: f64,
    pub distort: Distort,
}
#[derive(Serialize, Deserialize, Eq, PartialEq, Clone, Copy, Debug)]
#[serde(rename_all = "lowercase")]
pub enum Camera {
    Left,
    Right,
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug)]
pub struct TrackedCamera {
    pub extrinsics: Extrinsics,
    pub intrinsics: Intrinsics,
    pub name: Camera,
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug)]
pub struct StereoCamera {
    pub left: TrackedCamera,
    pub right: TrackedCamera,
}
impl StereoCamera {
    fn new(cfg: &LighthouseConfig) -> Result<Self> {
        log::debug!("Trying to find left camera");
        let left = cfg
            .tracked_cameras
            .iter()
            .copied()
            .find(|p| p.name == Camera::Left);
        log::debug!("Trying to find right camera");
        let right = cfg
            .tracked_cameras
            .iter()
            .copied()
            .find(|p| p.name == Camera::Right);
        if left.is_none() {
            bail!("Failed to find left camera");
        }
        if right.is_none() {
            bail!("Failed to find right camera");
        }
        Ok(StereoCamera {
            left: left.unwrap(),
            right: right.unwrap(),
        })
    }
}
/// Extract relevant bits of information from steam config files
#[derive(Serialize, Deserialize)]
pub struct LighthouseConfig {
    pub tracked_cameras: Vec<TrackedCamera>,
    pub device_serial_number: String,
}

ioctl_readwrite_buf!(hidiocgfeature, 'H', 7, u8);

fn hidiocgfeature_with_retry(
    fd: &impl AsRawFd,
    report_id: u8,
    buf: &mut [u8],
) -> Result<i32, Errno> {
    const MAX_RETRIES: u32 = 50;
    let mut retries = 0;
    loop {
        retries += 1;
        buf[0] = report_id;
        match unsafe { hidiocgfeature(fd.as_raw_fd(), buf) } {
            Ok(n) => break Ok(n),
            Err(e) => {
                if retries == MAX_RETRIES {
                    break Err(e);
                }
            }
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

pub fn load_config_file_from_headset(dev: &udev::Device) -> Option<StereoCamera> {
    let devnode = dev.devnode().unwrap();
    log::info!("Loading config file from headset: {}", devnode.display());
    let mut buf: [u8; 0x41] = [0; 0x41];
    let fd = match nix::fcntl::open(devnode, OFlag::O_RDWR, Mode::S_IRWXU) {
        Ok(fd) => fd,
        Err(e) => {
            log::error!("Couldn't open {}: {e}", devnode.display());
            return None;
        }
    };

    match hidiocgfeature_with_retry(&fd, 0x10, &mut buf) {
        Err(e) => {
            log::error!(
                "[{}] Failed to request start of config: {e:#}",
                devnode.display()
            );
            return None;
        }
        Ok(0) => {
            log::error!("[{}] Request start of config got 0", devnode.display());
            return None;
        }
        Ok(n) => n,
    };

    let mut config_data = vec![];
    loop {
        let n = match hidiocgfeature_with_retry(&fd, 0x11, &mut buf) {
            Err(e) => {
                log::error!(
                    "[{}] Failed to retrieve usb config data: {e:#}",
                    devnode.display()
                );
                return None;
            }
            Ok(n) => n,
        };
        if n == 0 || buf[1] == 0 {
            break;
        }
        config_data.extend_from_slice(&buf[2..][..buf[1] as usize]);
    }

    let mut config = String::new();
    let mut z = ZlibDecoder::new(&config_data[..]);
    let res = z.read_to_string(&mut config);
    if let Err(e) = res {
        log::error!(
            "[{}] Failed to decompress usb data: {e:#}",
            devnode.display()
        );
        return None;
    }

    log::debug!("{:?}", config);
    log::debug!("Trying to parse config");
    let lhconfig: LighthouseConfig = match serde_json::from_str(&config) {
        Ok(cfg) => cfg,
        Err(e) => {
            log::error!("Failed to parse config json: {e}");
            return None;
        }
    };

    let camera_config = match StereoCamera::new(&lhconfig) {
        Ok(cfg) => cfg,
        Err(e) => {
            log::error!("Couldn't load camera correction from headset config: {e}");
            return None;
        }
    };

    log::debug!("Serial number: {:?}", lhconfig.device_serial_number);
    let xdg = xdg::BaseDirectories::new();
    let cache = xdg
        .create_cache_directory("xr_passthrough_layer")
        .expect("Failed to create cache dir");
    let config_file = cache.join(lhconfig.device_serial_number.clone() + ".json");
    let file = File::create(config_file).expect("Failed to open cache file");
    if let Err(e) = serde_json::to_writer(file, &lhconfig) {
        log::warn!("Failed to write headset config to disk: {e}")
    }
    Some(camera_config)
}

fn find_steam_config_steam(serial: &OsStr) -> Option<StereoCamera> {
    let serial = serial.to_ascii_lowercase();
    let xdg = xdg::BaseDirectories::new();
    log::debug!("Base directories: {:?}", xdg);
    let steam = xdg
        .find_data_file("steam")
        .or_else(|| xdg.find_data_file("Steam"))?;
    log::debug!("Steam directory: {:?}", steam);
    let steam_config = steam
        .join("config")
        .join("lighthouse")
        .join(serial)
        .join("config.json");
    log::debug!("Steam lighthouse config path {}", steam_config.display());
    let json = std::fs::read_to_string(&steam_config).ok()?;
    log::debug!("Trying to parse config");
    let lhconfig: LighthouseConfig = serde_json::from_str(&json).ok()?;
    log::info!(
        "Found headset config in steam lighthouse config: {}",
        steam_config.display()
    );
    match StereoCamera::new(&lhconfig) {
        Err(e) => {
            log::error!(
                "Failed to get StereoCamera for device {e:#}: {}",
                lhconfig.device_serial_number
            );
            None
        }
        Ok(cfg) => Some(cfg),
    }
}

fn find_steam_config_cache(serial: &OsStr) -> Option<StereoCamera> {
    let xdg = xdg::BaseDirectories::new();
    let cache = match xdg.create_cache_directory("xr_passthrough_layer") {
        Ok(cache) => cache,
        Err(e) => {
            log::error!("Failed to create cache dir: {e}");
            return None;
        }
    };
    let mut cache = cache.join(serial).into_os_string();
    cache.push(".json");
    log::info!(
        "Didn't find config file for {} in steam config dir, trying {}",
        serial.display(),
        cache.display()
    );
    let json = std::fs::read_to_string(&cache).ok()?;
    log::debug!("Trying to parse config");
    let lhconfig: LighthouseConfig = serde_json::from_str(&json).ok()?;
    log::info!("Found headset config in cache: {}", cache.display());
    match StereoCamera::new(&lhconfig) {
        Err(e) => {
            log::error!(
                "Failed to get StereoCamera for device {e:#}: {}",
                lhconfig.device_serial_number
            );
            None
        }
        Ok(cfg) => Some(cfg),
    }
}

/// Try to find the config file for index
pub fn find_steam_config() -> Option<StereoCamera> {
    let mut it = match udev::Enumerator::new() {
        Ok(it) => it,
        Err(e) => {
            log::error!("Couldn't create udev enumerator: {e}");
            return None;
        }
    };
    if let Err(e) = it.match_property("ID_VENDOR_ID", "28de") {
        log::error!("Couldn't add match property: {e}");
        return None;
    }
    if let Err(e) = it.match_subsystem("hidraw") {
        log::error!("Couldn't add match subsystem: {e}");
        return None;
    }
    let mut devs = match it.scan_devices() {
        Ok(devs) => devs,
        Err(e) => {
            log::error!("Couldn't start device scan: {e}");
            return None;
        }
    };
    let dev = devs.find(|device| {
        let mut model = None;
        let mut interface_num = None;
        for prop in device.properties() {
            if prop.name() == "ID_MODEL_ID" {
                model = Some(prop)
            } else if prop.name() == "ID_USB_INTERFACE_NUM" {
                interface_num = Some(prop)
            }
        }
        model.is_some_and(|m| m.value() == "2300")
            && interface_num.is_some_and(|i| i.value() == "00")
            && device.devnode().is_some()
    });
    if dev.is_none() {
        log::error!("No headset found");
    }
    let dev = dev?;
    let serial = dev
        .properties()
        .find(|prop| prop.name() == "ID_SERIAL_SHORT");
    if serial.is_none() {
        log::error!("Headset has no serial number");
    }
    let serial = serial?;
    find_steam_config_steam(serial.value())
        .or_else(|| find_steam_config_cache(serial.value()))
        .or_else(|| load_config_file_from_headset(&dev))
}
// seems to not be used????
pub fn load_steam_config(hmd_serial: &str) -> Result<StereoCamera> {
    let xdg = xdg::BaseDirectories::new();
    let steam = xdg
        .find_data_file("steam")
        .or_else(|| xdg.find_data_file("Steam"))
        .with_context(|| anyhow!("Cannot find steam directory"))?;
    let lhconfig = std::fs::read_to_string(
        steam
            .join("config")
            .join("lighthouse")
            .join(hmd_serial.to_lowercase())
            .join("config.json"),
    )?;
    let lhconfig: LighthouseConfig = serde_json::from_str(&lhconfig)?;
    StereoCamera::new(&lhconfig)
}
