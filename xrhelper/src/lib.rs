use anyhow::{Context, Result, anyhow};
use glam::UVec2;
use itertools::Itertools;
use nalgebra::{Affine3, Matrix3, UnitQuaternion};
use openxr::{
    ApplicationInfo, FrameStream, FrameWaiter, ReferenceSpaceType, ViewConfigurationType,
    sys::Handle as _,
};
use std::{
    collections::HashSet,
    sync::{Arc, OnceLock},
};
use vulkano::{
    Handle as _, VulkanObject,
    device::{
        DefaultQueueMutex, Device, DeviceExtensions, DeviceFeatures, DeviceQueueInfo, Queue,
        QueueFlags,
    },
    image::Image,
    instance::{Instance, InstanceExtensions as VkInstanceExtensions},
};

use vulkano::{
    device::QueueCreateInfo,
    image::{ImageCreateInfo, ImageUsage},
};

static VULKAN_LIBRARY: OnceLock<Arc<vulkano::VulkanLibrary>> = OnceLock::new();

fn get_vulkan_library() -> &'static Arc<vulkano::VulkanLibrary> {
    VULKAN_LIBRARY.get_or_init(|| unsafe { vulkano::VulkanLibrary::new() }.unwrap())
}

struct VulkanKeepAlive {
    _instance: Arc<Instance>,
    _device: Arc<Device>,
    _queue: Arc<Queue>,
}

pub struct OpenXr {
    device: Arc<Device>,
    queue: Arc<Queue>,
    vk_instance: Arc<Instance>,

    space: openxr::Space,
    swapchain_images: Vec<Arc<Image>>,
    swapchain: openxr::Swapchain<openxr::Vulkan>,
    depth_swapchain_images: Option<Vec<Arc<Image>>>,
    depth_swapchain: Option<openxr::Swapchain<openxr::Vulkan>>,
    session: openxr::Session<openxr::Vulkan>,
    instance: openxr::Instance,

    /// Render size of a single eye.
    render_size: UVec2,
}
pub fn affine_to_posef(t: Affine3<f32>) -> openxr::Posef {
    let m = t.to_homogeneous();
    let r: Matrix3<f32> = m.fixed_columns::<3>(0).fixed_rows::<3>(0).into();
    let rotation = nalgebra::geometry::Rotation3::from_matrix(&r);
    let quaternion = UnitQuaternion::from_rotation_matrix(&rotation);
    let quaternion = &quaternion.as_ref().coords;
    let translation: nalgebra::Vector3<f32> =
        [m.data.0[3][0], m.data.0[3][1], m.data.0[3][2]].into();
    openxr::Posef {
        orientation: openxr::Quaternionf {
            x: quaternion.x,
            y: quaternion.y,
            z: quaternion.z,
            w: quaternion.w,
        },
        position: openxr::Vector3f {
            x: translation.x,
            y: translation.y,
            z: translation.z,
        },
    }
}

pub fn posef_to_nalgebra(posef: openxr::Posef) -> (UnitQuaternion<f32>, nalgebra::Vector3<f32>) {
    let quaternion = UnitQuaternion::new_normalize(nalgebra::Quaternion::new(
        posef.orientation.w,
        posef.orientation.x,
        posef.orientation.y,
        posef.orientation.z,
    ));
    let translation: nalgebra::Vector3<f32> =
        [posef.position.x, posef.position.y, posef.position.z].into();
    (quaternion, translation)
}

pub struct RenderInfo<'a> {
    pub session: &'a openxr::Session<openxr::Vulkan>,
    pub swapchain: &'a mut openxr::Swapchain<openxr::Vulkan>,
    pub swapchain_images: &'a Vec<Arc<Image>>,
    pub depth_swapchain: Option<&'a mut openxr::Swapchain<openxr::Vulkan>>,
    pub depth_swapchain_images: Option<&'a Vec<Arc<Image>>>,
    pub space: &'a openxr::Space,
    pub render_size: UVec2,
}

impl OpenXr {
    fn create_vk_device(
        xr_instance: &openxr::Instance,
        xr_system: openxr::SystemId,
        instance: &Arc<Instance>,
    ) -> Result<(Arc<Device>, Arc<Queue>)> {
        let vk_requirements = xr_instance.graphics_requirements::<openxr::Vulkan>(xr_system)?;
        let physical_device = unsafe {
            let physical_device =
                xr_instance.vulkan_graphics_device(xr_system, instance.handle().as_raw() as _)?;
            vulkano::device::physical::PhysicalDevice::from_handle(
                instance,
                ash::vk::PhysicalDevice::from_raw(physical_device as _),
            )
        }?;
        let min_version = vulkano::Version::major_minor(
            vk_requirements.min_api_version_supported.major() as u32,
            vk_requirements.min_api_version_supported.minor() as u32,
        );
        if physical_device.api_version() < min_version {
            return Err(anyhow!(
                "Vulkan API version not supported {}",
                physical_device.api_version(),
            ));
        }
        let ext = DeviceExtensions {
            khr_swapchain: true,
            khr_multiview: true,
            ..Default::default()
        };
        let raw_extensions = ext
            .into_iter()
            .filter(|&(_, enabled)| enabled)
            .map(|(name, _)| std::ffi::CString::new(name).unwrap())
            .collect::<Vec<_>>();
        let raw_extensions = raw_extensions
            .iter()
            .map(|s| s.as_ptr())
            .collect::<Vec<_>>();
        let queue_family = physical_device
            .queue_family_properties()
            .iter()
            .position(|qf| qf.queue_flags.contains(QueueFlags::GRAPHICS))
            .context("No graphics queue found")?;
        log::debug!("queue family: {queue_family}");
        let queue_create_info = ash::vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family as u32)
            .queue_priorities(std::slice::from_ref(&1.0));
        let mut multiview_features =
            ash::vk::PhysicalDeviceMultiviewFeatures::default().multiview(true);
        let features = ash::vk::PhysicalDeviceFeatures2 {
            p_next: &mut multiview_features as *mut _ as _,
            ..Default::default()
        };
        let mut create_info = ash::vk::DeviceCreateInfo::default()
            .enabled_extension_names(&raw_extensions)
            .queue_create_infos(std::slice::from_ref(&queue_create_info));
        create_info.p_next = &features as *const _ as _;

        let vulkano_create_info = vulkano::device::DeviceCreateInfo {
            queue_create_infos: &[QueueCreateInfo {
                queue_family_index: queue_family as u32,
                queues: &[1.0],
                ..Default::default()
            }],
            enabled_extensions: &ext,
            enabled_features: &DeviceFeatures {
                multiview: true,
                ..Default::default()
            },
            physical_devices: &[&physical_device],
            ..Default::default()
        };
        let device = unsafe {
            vulkano::device::Device::from_handle(
                &physical_device,
                ash::vk::Device::from_raw(
                    xr_instance
                        .create_vulkan_device(
                            xr_system,
                            get_instance_proc_addr,
                            physical_device.handle().as_raw() as _,
                            (&create_info) as *const _ as _,
                        )?
                        .map_err(ash::vk::Result::from_raw)? as _,
                ),
                &vulkano_create_info,
            )
        };
        let queue = unsafe {
            let mut queue = ash::vk::Queue::null();
            (device.fns().v1_0.get_device_queue)(
                device.handle(),
                queue_family as u32,
                0,
                &mut queue,
            );
            vulkano::device::Queue::from_handle(
                &device,
                queue,
                &DeviceQueueInfo {
                    queue_family_index: queue_family as u32,
                    queue_index: 0,
                    ..Default::default()
                },
                Arc::new(DefaultQueueMutex::new()),
            )
        };
        Ok((device, queue))
    }

    fn create_vk_instance(
        mut vk_instance_extensions: VkInstanceExtensions,
        xr_instance: &openxr::Instance,
        xr_system: openxr::SystemId,
    ) -> Result<Arc<Instance>> {
        let vk_requirements = xr_instance.graphics_requirements::<openxr::Vulkan>(xr_system)?;
        log::info!("Vulkan requirements: {vk_requirements:?}");
        let extensions = *get_vulkan_library().supported_extensions();
        vk_instance_extensions.khr_surface = true;
        if let Some(unsupported) = vk_instance_extensions
            .difference(&extensions)
            .into_iter()
            .find_map(|(name, enabled)| enabled.then_some(name))
        {
            return Err(anyhow!(
                "Required instance extension {unsupported} not supported"
            ));
        }
        let vk_version = vulkano::Version::major_minor(
            vk_requirements.max_api_version_supported.major() as u32,
            vk_requirements.max_api_version_supported.minor() as u32,
        );
        let vk_version = vk_version.min(get_vulkan_library().api_version());

        let vulkano_create_info = vulkano::instance::InstanceCreateInfo {
            max_api_version: Some(vk_version),
            enabled_extensions: &vk_instance_extensions,
            enabled_layers: &[
                "VK_LAYER_KHRONOS_validation",
                //"VK_LAYER_LUNARG_api_dump".to_owned(),
                //"VK_LAYER_LUNARG_gfxreconstruct".to_owned(),
            ],
            ..Default::default()
        };
        let extensions = vk_instance_extensions
            .into_iter()
            .filter(|(_, enabled)| *enabled)
            .map(|(ext, _)| std::ffi::CString::new(ext).unwrap())
            .collect::<Vec<_>>();
        let extensions = extensions
            .iter()
            .map(|s| s.as_c_str().as_ptr())
            .collect::<Vec<_>>();
        let application_info =
            ash::vk::ApplicationInfo::default().api_version(vk_version.try_into().unwrap());
        let mut buf = [0u8; 1024];
        let mut buflen = 0;
        unsafe {
            (xr_instance
                .exts()
                .khr_vulkan_enable
                .unwrap()
                .get_vulkan_instance_extensions)(
                xr_instance.as_raw(),
                xr_system,
                buf.len() as _,
                &mut buflen,
                buf.as_mut_ptr() as _,
            )
        };
        println!(
            "{:?}",
            std::ffi::CStr::from_bytes_until_nul(&buf).unwrap().to_str()
        );
        let instance = unsafe {
            xr_instance.create_vulkan_instance(
                xr_system,
                get_instance_proc_addr,
                (&ash::vk::InstanceCreateInfo::default()
                    .enabled_extension_names(&extensions)
                    .application_info(&application_info)
                    .enabled_layer_names(&[
                        c"VK_LAYER_KHRONOS_validation".as_ptr(),
                        //c"VK_LAYER_LUNARG_api_dump".as_ptr(),
                        //c"VK_LAYER_LUNARG_gfxreconstruct".as_ptr(),
                    ])) as *const _ as _,
            )?
        }
        .map_err(ash::vk::Result::from_raw)?;
        let instance = ash::vk::Instance::from_raw(instance as _);
        Ok(unsafe { Instance::from_handle(get_vulkan_library(), instance, &vulkano_create_info) })
    }

    /// render_size: Resolution of the swapchain image for a *single* eye.
    pub fn new(
        vk_instance_extensions: VkInstanceExtensions,
        xr_instance_extensions: &openxr::ExtensionSet,
        api_layers: &[&str],
        app_name: &str,
        app_version: u32,
    ) -> Result<(Self, FrameWaiter, FrameStream<openxr::Vulkan>)> {
        let entry = unsafe { openxr::Entry::load()? };

        let mut xr_instance_extensions = xr_instance_extensions.clone();
        xr_instance_extensions.khr_vulkan_enable = true;
        xr_instance_extensions.khr_vulkan_enable2 = true;
        xr_instance_extensions.khr_convert_timespec_time = true;
        let instance = entry.create_instance(
            &ApplicationInfo {
                application_name: app_name,
                application_version: app_version,
                api_version: openxr::Version::new(1, 0, 0),
                engine_name: "engine",
                engine_version: 0,
            },
            &xr_instance_extensions,
            api_layers,
        )?;
        let system = instance.system(openxr::FormFactor::HEAD_MOUNTED_DISPLAY)?;
        let blend_modes = instance.enumerate_environment_blend_modes(
            system,
            openxr::ViewConfigurationType::PRIMARY_STEREO,
        )?;
        log::info!("{:?}", blend_modes);
        if !blend_modes.contains(&openxr::EnvironmentBlendMode::OPAQUE) {
            return Err(anyhow!("OpenXR runtime doesn't support opaque blend mode"));
        }
        let vk_instance = Self::create_vk_instance(vk_instance_extensions, &instance, system)?;
        let (device, queue) = Self::create_vk_device(&instance, system, &vk_instance)?;
        let binding = openxr::sys::GraphicsBindingVulkanKHR {
            ty: openxr::sys::GraphicsBindingVulkanKHR::TYPE,
            next: std::ptr::null(),
            instance: vk_instance.handle().as_raw() as _,
            physical_device: device.physical_device().handle().as_raw() as _,
            device: device.handle().as_raw() as _,
            queue_family_index: queue.queue_family_index(),
            queue_index: queue.queue_index(),
        };
        let info = openxr::sys::SessionCreateInfo {
            ty: openxr::sys::SessionCreateInfo::TYPE,
            next: &binding as *const _ as *const _,
            create_flags: Default::default(),
            system_id: system,
        };
        let mut out = openxr::sys::Session::NULL;
        let ret = unsafe { (instance.fp().create_session)(instance.as_raw(), &info, &mut out) };
        if ret.into_raw() < 0 {
            log::warn!("Cannot create session {ret}");
            return Err(ret.into());
        }
        let cfgs = instance
            .enumerate_view_configuration_views(system, ViewConfigurationType::PRIMARY_STEREO)?;
        if cfgs.len() != 2 {
            return Err(anyhow!("Stereo view has unexpected number of configs"));
        }
        if cfgs[0].recommended_image_rect_height != cfgs[1].recommended_image_rect_height
            || cfgs[0].recommended_image_rect_width != cfgs[1].recommended_image_rect_width
        {
            log::warn!("Stereo view has different recommended image rect sizes");
        }
        log::info!(
            "Recommended image rect sizes: {:?} {:?}",
            cfgs[0].recommended_image_rect_width,
            cfgs[0].recommended_image_rect_height
        );
        let width = cfgs[0]
            .recommended_image_rect_width
            .max(cfgs[1].recommended_image_rect_width);
        let height = cfgs[0]
            .recommended_image_rect_height
            .max(cfgs[1].recommended_image_rect_height);
        let sample_count = cfgs[0]
            .recommended_swapchain_sample_count
            .max(cfgs[1].recommended_swapchain_sample_count);

        let (session, frame_waiter, frame_stream) = unsafe {
            openxr::Session::<openxr::Vulkan>::from_raw(
                instance.clone(),
                out,
                Box::new(VulkanKeepAlive {
                    // Since the XR session uses the Vulkan device, these
                    // vulkan objects must not be dropped until the session
                    // is destroyed.
                    _instance: vk_instance.clone(),
                    _device: device.clone(),
                    _queue: queue.clone(),
                }),
            )
        };
        let formats = session.enumerate_swapchain_formats()?;
        let formats = formats
            .into_iter()
            .map(|f| vulkano::format::Format::try_from(ash::vk::Format::from_raw(f as _)).unwrap())
            .collect::<HashSet<_>>();
        log::info!("swapchain formats: {:?}", formats);
        const PREFERRED_FORMATS: [vulkano::format::Format; 4] = [
            vulkano::format::Format::R8G8B8A8_UNORM,
            vulkano::format::Format::B8G8R8A8_UNORM,
            vulkano::format::Format::R8G8B8A8_SRGB,
            vulkano::format::Format::B8G8R8A8_SRGB,
        ];
        let Some(format) = PREFERRED_FORMATS
            .iter()
            .find(|f| formats.contains(f))
            .copied()
        else {
            return Err(anyhow!("No suitable format found for swapchain"));
        };
        let swapchain = session.create_swapchain(&openxr::SwapchainCreateInfo {
            array_size: 2,
            face_count: 1,
            create_flags: Default::default(),
            usage_flags: openxr::SwapchainUsageFlags::COLOR_ATTACHMENT
                | openxr::SwapchainUsageFlags::TRANSFER_DST,
            format: format as u32,
            sample_count,
            width,
            height,
            mip_count: 1,
        })?;
        log::debug!("created swapchain");
        let swapchain_images: Vec<_> = swapchain
            .enumerate_images()?
            .into_iter()
            .map(|handle| {
                let handle = ash::vk::Image::from_raw(handle);
                let raw_image = unsafe {
                    vulkano::image::sys::RawImage::from_handle_borrowed(
                        &device,
                        handle,
                        &ImageCreateInfo {
                            format,
                            array_layers: 2,
                            extent: [width, height, 1],
                            usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_DST,
                            ..Default::default()
                        },
                    )?
                };
                // SAFETY: OpenXR guarantees that the image is a swapchain image, thus has memory backing it.
                let image = unsafe { raw_image.assume_bound() };
                Ok::<_, anyhow::Error>(Arc::new(image))
            })
            .try_collect()?;
        let (depth_swapchain, depth_swapchain_images) =
            if instance.exts().khr_composition_layer_depth.is_some() {
                const PREFERRED_DEPTH_FORMATS: [vulkano::format::Format; 5] = [
                    vulkano::format::Format::D32_SFLOAT,
                    vulkano::format::Format::D16_UNORM,
                    vulkano::format::Format::D32_SFLOAT_S8_UINT,
                    vulkano::format::Format::D24_UNORM_S8_UINT,
                    vulkano::format::Format::D16_UNORM_S8_UINT,
                ];
                if let Some(format) = PREFERRED_DEPTH_FORMATS
                    .iter()
                    .find(|f| formats.contains(f))
                    .copied()
                {
                    let depth_swapchain =
                        session.create_swapchain(&openxr::SwapchainCreateInfo {
                            array_size: 2,
                            face_count: 1,
                            create_flags: Default::default(),
                            usage_flags: openxr::SwapchainUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                            format: format as u32,
                            sample_count,
                            width,
                            height,
                            mip_count: 1,
                        })?;
                    log::debug!("created depth swapchain");
                    let depth_swapchain_images: Vec<_> = depth_swapchain
                        .enumerate_images()?
                        .into_iter()
                        .map(|handle| {
                            let handle = ash::vk::Image::from_raw(handle);
                            let raw_image = unsafe {
                                vulkano::image::sys::RawImage::from_handle_borrowed(
                                    &device,
                                    handle,
                                    &ImageCreateInfo {
                                        format,
                                        array_layers: 2,
                                        extent: [width, height, 1],
                                        usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT,
                                        ..Default::default()
                                    },
                                )?
                            };
                            // SAFETY: OpenXR guarantees that the image is a swapchain image, thus has memory backing it.
                            let image = unsafe { raw_image.assume_bound() };
                            Ok::<_, anyhow::Error>(Arc::new(image))
                        })
                        .try_collect()?;
                    assert_eq!(depth_swapchain_images.len(), swapchain_images.len());
                    (Some(depth_swapchain), Some(depth_swapchain_images))
                } else {
                    log::warn!("No suitable depth format found for swapchain");
                    (None, None)
                }
            } else {
                (None, None)
            };
        log::debug!("got swapchain images");
        let space =
            session.create_reference_space(ReferenceSpaceType::STAGE, openxr::Posef::IDENTITY)?;
        log::debug!("created actions");
        Ok((
            Self {
                instance,
                session,
                swapchain,
                swapchain_images,
                depth_swapchain,
                depth_swapchain_images,
                space,

                vk_instance,
                device,
                queue,

                render_size: UVec2::new(width, height),
            },
            frame_waiter,
            frame_stream,
        ))
    }
    pub fn render_info(&mut self) -> RenderInfo<'_> {
        RenderInfo {
            session: &self.session,
            swapchain: &mut self.swapchain,
            swapchain_images: &self.swapchain_images,
            depth_swapchain: self.depth_swapchain.as_mut(),
            depth_swapchain_images: self.depth_swapchain_images.as_ref(),
            space: &self.space,
            render_size: self.render_size,
        }
    }
    pub fn vk_device(&self) -> (Arc<Device>, Arc<Queue>) {
        (self.device.clone(), self.queue.clone())
    }
    pub fn vk_instance(&self) -> Arc<Instance> {
        self.vk_instance.clone()
    }
    pub fn xr_instance(&self) -> &openxr::Instance {
        &self.instance
    }
    pub fn xr_session(&self) -> &openxr::Session<openxr::Vulkan> {
        &self.session
    }
    pub fn render_size(&self) -> UVec2 {
        self.render_size
    }
}

unsafe extern "system" fn get_instance_proc_addr(
    instance: openxr::sys::platform::VkInstance,
    name: *const std::ffi::c_char,
) -> Option<unsafe extern "system" fn()> {
    let instance = ash::vk::Instance::from_raw(instance as _);
    let library = get_vulkan_library();
    unsafe { library.get_instance_proc_addr(instance, name) }
}

pub trait XrContext {
    fn context<C>(self, ctx: C) -> anyhow::Result<()>
    where
        C: std::fmt::Display + Send + Sync + 'static;
    fn with_context<C>(self, f: impl FnOnce(&Self) -> C) -> anyhow::Result<()>
    where
        C: std::fmt::Display + Send + Sync + 'static;
}

impl XrContext for openxr::sys::Result {
    fn context<C>(self, ctx: C) -> anyhow::Result<()>
    where
        C: std::fmt::Display + Send + Sync + 'static,
    {
        if self == openxr::sys::Result::SUCCESS {
            Ok(())
        } else {
            Err(self).context(ctx)
        }
    }

    fn with_context<C>(self, f: impl FnOnce(&Self) -> C) -> anyhow::Result<()>
    where
        C: std::fmt::Display + Send + Sync + 'static,
    {
        if self == openxr::sys::Result::SUCCESS {
            Ok(())
        } else {
            let save = self;
            Err(self).with_context(|| f(&save))
        }
    }
}
