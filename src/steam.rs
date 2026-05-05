use std::fs::File;
use std::io::prelude::*;
use std::os::fd::AsRawFd;

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

pub fn load_config_file_from_headset() -> Option<StereoCamera> {
    log::debug!("Loading config file from headset");
    let mut it = udev::Enumerator::new().expect("Failed to create udev enumerator");
    it.match_property("ID_VENDOR_ID", "28de")
        .expect("Failed to add property filter");
    it.match_property("ID_MODEL_ID", "2300")
        .expect("Failed to add property filter");
    it.match_subsystem("hidraw")
        .expect("Failed to add subsystem filter");
    //it.match_attribute("bInterfaceNumber", "00")?;
    let mut buf: [u8; 0x41] = [0; 0x41];
    it.scan_devices()
        .expect("Failed to scan devices")
        .find_map(|device| {
            log::debug!("Found device {:?}", device.devnode());
            let fd = nix::fcntl::open(
                device.devnode().expect("failed to get devnode"),
                OFlag::O_RDWR,
                Mode::S_IRWXU,
            )
            .expect("Failed to open device");

            let mut res: Result<i32, Errno> = Errno::result(-1);
            let mut retries = 0;
            while res.is_err() && retries < 50 {
                retries += 1;
                buf[0] = 0x10;
                unsafe {
                    res = hidiocgfeature(fd.as_raw_fd(), &mut buf);
                }
            }
            if let Err(e) = res {
                log::error!("Failed to request start of config: {e:#}");
                return None;
            }

            let mut config_data: Vec<u8> = vec![];

            while res.unwrap() != 0 && buf[1] != 0 {
                retries = 0;
                res = Errno::result(-1);
                while res.is_err() && retries < 50 {
                    retries += 1;
                    buf[0] = 0x11;
                    unsafe {
                        res = hidiocgfeature(fd.as_raw_fd(), &mut buf);
                    }
                }
                if let Err(e) = res {
                    log::error!("Failed to retrieve usb config data: {e:#}");
                    return None;
                }
                config_data.extend_from_slice(&buf[2..buf.len() - 1]);
            }

            let mut config = String::new();
            let mut z = ZlibDecoder::new(&config_data[..]);
            let res = z.read_to_string(&mut config);
            if let Err(e) = res {
                log::error!("Failed to decompress usb data: {e:#}");
                return None;
            }

            log::debug!("{:?}", config);
            log::debug!("Trying to parse config");
            let lhconfig: LighthouseConfig = serde_json::from_str(&config).ok()?;
            log::debug!("Serial number: {:?}", lhconfig.device_serial_number);
            let xdg = xdg::BaseDirectories::new();
            let cache = xdg
                .create_cache_directory("xr_passthrough_layer")
                .expect("Failed to create cache dir");
            let config_file = cache.join(lhconfig.device_serial_number.clone() + ".json");
            let mut file = File::create(config_file).expect("Failed to open cache file");
            file.write_all(config.as_bytes())
                .expect("Failed to write headset config");

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
        })
}

fn find_steam_config_steam() -> Option<StereoCamera> {
    let xdg = xdg::BaseDirectories::new();
    log::debug!("Base directories: {:?}", xdg);
    let steam = xdg
        .find_data_file("steam")
        .or_else(|| xdg.find_data_file("Steam"))?;
    log::debug!("Steam directory: {:?}", steam);
    let steam_config = steam.join("config").join("lighthouse");
    log::debug!("Enumerating steam config dir {:?}", steam_config);
    let mut files = steam_config.read_dir().ok()?;
    files.find_map(|dir| {
        log::debug!("Trying to find config in {:?}", dir);
        let dir = dir.ok()?;
        let config = dir.path().join("config.json");
        log::debug!("Trying to read config from {:?}", config);
        let json = std::fs::read_to_string(config).ok()?;
        log::debug!("Trying to parse config");
        let lhconfig: LighthouseConfig = serde_json::from_str(&json).ok()?;
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
    })
}

fn find_steam_config_cache() -> Option<StereoCamera> {
    let xdg = xdg::BaseDirectories::new();
    let cache = xdg
        .create_cache_directory("xr_passthrough_layer")
        .expect("Failed to create cache dir");
    log::debug!(
        "Didn't find config file in steam config dir, trying {:?}",
        cache
    );
    let mut cache_files = cache.read_dir().ok()?;
    cache_files.find_map(|config| {
        log::debug!("Trying to read config from {:?}", config);
        let config = config.ok()?;
        let json = std::fs::read_to_string(config.path()).ok()?;
        log::debug!("Trying to parse config");
        let lhconfig: LighthouseConfig = serde_json::from_str(&json).ok()?;
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
    })
}

/// Try to find the config file for index
pub fn find_steam_config() -> Option<StereoCamera> {
    find_steam_config_steam()
        .or_else(find_steam_config_cache)
        .or_else(|| {
            Some(load_config_file_from_headset().expect("Failed to load config file from headset"))
        })
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
