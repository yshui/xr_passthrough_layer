use glam::UVec2;
use log::{debug, error, trace, warn};
use openxr::{
    AsHandle, CompositionLayerFlags, EnvironmentBlendMode, Extent2Di, Offset2Di, Rect2Di,
    ReferenceSpaceType, SwapchainCreateFlags, SwapchainCreateInfo, SwapchainSubImage,
    SwapchainUsageFlags, ViewStateFlags, sys::Handle as _,
};
use quark::{Hooked as _, Low as _, prelude::XrResult, try_xr, types::AnySession};
use smallvec::smallvec;
use std::{collections::HashSet, sync::Arc};
use vulkano::{
    Handle as _, VulkanObject,
    command_buffer::{
        AutoCommandBufferBuilder, BlitImageInfo, CommandBufferUsage, ImageBlit,
        allocator::{CommandBufferAllocator, StandardCommandBufferAllocator},
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::{DefaultQueueMutex, DeviceExtensions, DeviceQueueInfo, QueueCreateInfo},
    image::{ImageCreateInfo, ImageLayout, ImageUsage},
    instance::InstanceExtensions,
    memory::allocator::{MemoryAllocator, StandardMemoryAllocator},
    sync::GpuFuture as _,
};

use crate::api_layer::{CameraResources, PassthroughInner, XrErr, xrcvt};

#[allow(clippy::large_enum_variant)]
enum SessionState {
    Idle {
        passthrough: Option<Arc<PassthroughInner>>,
    },
    Running {
        view_type: openxr::sys::ViewConfigurationType,
        passthrough: Option<(Arc<PassthroughInner>, CameraResources)>,
        /// Extra swapchain for our own rendering needs
        swapchain: openxr::Swapchain<openxr::Vulkan>,
        images: Vec<Arc<vulkano::image::Image>>,
        size: UVec2,
        image_index: Option<u32>,
    },
}
struct SessionDataInner {
    device: Arc<vulkano::device::Device>,
    queue: Arc<vulkano::device::Queue>,
    state: SessionState,
    system_id: openxr::SystemId,
    allocator: Arc<dyn MemoryAllocator>,
    cmdbuf_allocator: Arc<dyn CommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    space: openxr::Space,
}
impl SessionDataInner {
    fn create_camera_resources(
        &self,
        camera_cfg: Option<&crate::steam::StereoCamera>,
        splash: &[u8],
        size: UVec2,
    ) -> Result<CameraResources, XrErr> {
        let xdg = xdg::BaseDirectories::new();
        let pipeline_cache =
            crate::config::load_pipeline_cache(&self.device, &xdg).map_err(|e| {
                warn!("Failed to load pipeline cache {e:#}");
                XrErr::ERROR_RUNTIME_FAILURE
            })?;
        let pp = crate::pipeline::Pipeline::new(
            &self.device,
            &self.allocator,
            &self.cmdbuf_allocator,
            &self.queue,
            &self.descriptor_set_allocator,
            true,
            camera_cfg,
            ImageLayout::ShaderReadOnlyOptimal,
            ImageUsage::SAMPLED | ImageUsage::TRANSFER_SRC,
            pipeline_cache,
            UVec2::new(crate::CAMERA_SIZE, crate::CAMERA_SIZE),
            size,
        )
        .map_err(|e| {
            warn!("Failed to create pipeline {e:#}");
            XrErr::ERROR_RUNTIME_FAILURE
        })?;
        let camera = crate::find_index_camera().map_err(|e| {
            warn!("Cannot find camera {e:#}");
            XrErr::ERROR_RUNTIME_FAILURE
        })?;
        let camera = v4l::Device::with_path(camera).map_err(|e| {
            warn!("Failed to open camera {e:#}");
            XrErr::ERROR_RUNTIME_FAILURE
        })?;
        let camera = crate::camera::CameraThread::new(camera, splash);
        camera.resume().unwrap();
        Ok(CameraResources { camera, pp })
    }
    /// Transition `[Self::Running]` or `[Self::Idle]` states into `[Self::RunningWithPassthrough]`
    /// and `[Self::IdleWithPassthrough]` respectively.
    ///
    /// Returns `Ok(Some(passthrough))` if this session state wasn't already in a passthrough state,
    /// otherwise `Ok(None)`. If errors occurred while trying to transition, `Err` is returned.
    fn get_or_add_passthrough(&mut self) -> Result<Arc<PassthroughInner>, XrErr> {
        Ok(match &mut self.state {
            &mut SessionState::Running {
                ref passthrough,
                size,
                ..
            } => {
                if let Some(passthrough) = passthrough {
                    passthrough.0.clone()
                } else {
                    let camera = self.create_camera_resources(
                        PassthroughInner::camera_cfg(),
                        PassthroughInner::splash(),
                        size,
                    )?;
                    let new_pt = Arc::new(PassthroughInner::default());
                    match &mut self.state {
                        SessionState::Running { passthrough, .. } => {
                            *passthrough = Some((new_pt.clone(), camera))
                        }
                        _ => unreachable!(),
                    }
                    new_pt
                }
            }
            SessionState::Idle { passthrough } => passthrough.get_or_insert_default().clone(),
        })
    }
    fn is_running(&self) -> bool {
        matches!(&self.state, SessionState::Running { .. })
    }
    fn begin(
        &mut self,
        session: &openxr::Session<openxr::Vulkan>,
        view_type: openxr::ViewConfigurationType,
    ) -> Result<(), XrErr> {
        match &self.state {
            SessionState::Running { .. } => Err(XrErr::ERROR_SESSION_RUNNING),
            SessionState::Idle { passthrough } => {
                let instance = session.instance();
                let cfgs =
                    instance.enumerate_view_configuration_views(self.system_id, view_type)?;
                if cfgs.len() != 1 && cfgs.len() != 2 {
                    error!("unsupported view count? {}", cfgs.len());
                }
                let width = cfgs[0]
                    .recommended_image_rect_width
                    .max(cfgs[1].recommended_image_rect_width);
                let height = cfgs[0]
                    .recommended_image_rect_height
                    .max(cfgs[1].recommended_image_rect_height);
                let size = UVec2::new(width, height);
                let passthrough = if let Some(p) = passthrough {
                    let camera = self.create_camera_resources(
                        PassthroughInner::camera_cfg(),
                        PassthroughInner::splash(),
                        size,
                    )?;
                    Some((p.clone(), camera))
                } else {
                    None
                };
                let formats = session
                    .enumerate_swapchain_formats()?
                    .into_iter()
                    .map(|f| vulkano::format::Format::try_from(ash::vk::Format::from_raw(f as i32)))
                    .collect::<Result<HashSet<_>, _>>()
                    .map_err(|()| {
                        warn!("Invalid swapchain formats");
                        XrErr::ERROR_RUNTIME_FAILURE
                    })?;
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
                    warn!("No suitable format found for swapchain");
                    return Err(XrErr::ERROR_RUNTIME_FAILURE);
                };

                let swapchain = session.create_swapchain(&SwapchainCreateInfo {
                    array_size: 1,
                    face_count: 1,
                    format: format as u32,
                    mip_count: 1,
                    sample_count: cfgs[0].recommended_swapchain_sample_count,
                    usage_flags: SwapchainUsageFlags::COLOR_ATTACHMENT
                        | SwapchainUsageFlags::TRANSFER_DST,
                    create_flags: SwapchainCreateFlags::EMPTY,
                    width: width * 2,
                    height,
                })?;
                let images = swapchain
                    .enumerate_images()?
                    .into_iter()
                    .map(|raw_img| unsafe {
                        Ok::<_, XrErr>(Arc::new(
                            vulkano::image::sys::RawImage::from_handle_borrowed(
                                &self.device,
                                ash::vk::Image::from_raw(raw_img),
                                &ImageCreateInfo {
                                    format,
                                    extent: [width * 2, height, 1],
                                    array_layers: 1,
                                    mip_levels: 1,
                                    usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_DST,
                                    ..Default::default()
                                },
                            )
                            .map_err(|e| {
                                warn!("Failed to wrap image {e:#}");
                                XrErr::ERROR_RUNTIME_FAILURE
                            })?
                            .assume_bound(),
                        ))
                    })
                    .collect::<Result<_, _>>()?;
                self.state = SessionState::Running {
                    view_type,
                    passthrough,
                    image_index: None,
                    swapchain,
                    images,
                    size,
                };
                Ok(())
            }
        }
    }
    /// # Panic
    ///
    /// panics if session is not running.
    fn end(&mut self) {
        self.state = match &self.state {
            SessionState::Idle { .. } => {
                panic!("session not running")
            }
            SessionState::Running { passthrough, .. } => SessionState::Idle {
                passthrough: passthrough.as_ref().map(|(p, _)| p.clone()),
            },
        };
    }
}
#[derive(Default)]
pub struct SessionData {
    inner: Option<SessionDataInner>,
}

struct FrameEndInfo<'a> {
    display_time: openxr::Time,
    environment_blend_mode: openxr::EnvironmentBlendMode,
    layers: &'a [Option<&'a openxr::sys::CompositionLayerBaseHeader>],
}

impl FrameEndInfo<'_> {
    unsafe fn from_raw(info: &openxr::sys::FrameEndInfo) -> Self {
        Self {
            display_time: info.display_time,
            environment_blend_mode: info.environment_blend_mode,
            layers: unsafe {
                std::slice::from_raw_parts(
                    // Safety: Option<&T> and *const T are bitwise identical.
                    info.layers as *const Option<&openxr::sys::CompositionLayerBaseHeader>,
                    info.layer_count as _,
                )
            },
        }
    }
    fn as_raw(&self) -> openxr::sys::FrameEndInfo {
        openxr::sys::FrameEndInfo {
            ty: openxr::sys::FrameEndInfo::TYPE,
            next: std::ptr::null_mut(),
            display_time: self.display_time,
            environment_blend_mode: self.environment_blend_mode,
            layer_count: self.layers.len() as _,
            layers: self.layers.as_ptr() as *const _,
        }
    }
}

impl SessionData {
    unsafe fn end_frame(
        &mut self,
        session: &quark::types::AnySession,
        instance: &openxr::Instance,
        info: &FrameEndInfo<'_>,
    ) -> Result<(), XrErr> {
        for (i, l) in info.layers.iter().enumerate() {
            if let Some(l) = l {
                trace!("[{i}] = {:?}", l.ty);
            } else {
                trace!("[{i}] = <empty>");
            }
        }
        let Some(data) = &mut self.inner else {
            // We are not wrapping this session, passed it through.
            debug!("Unhandled session");
            return xrcvt(unsafe {
                (instance.fp().end_frame)(session.as_handle(), &info.as_raw())
            });
        };
        let quark::types::AnySession::Vulkan(xr_vk_session) = session else {
            unreachable!()
        };

        let has_passthrough = info
            .layers
            .iter()
            .filter(|l| {
                l.is_some_and(|l| l.ty == openxr::sys::CompositionLayerPassthroughHTC::TYPE)
            })
            .count();
        if has_passthrough > 1 {
            warn!("More than one passthrough layer, not supported");
            return Err(XrErr::ERROR_VALIDATION_FAILURE);
        }
        let passthrough_layer = info.layers.iter().find_map(|l| {
            if let &Some(l) = l
                && l.ty == openxr::sys::CompositionLayerPassthroughHTC::TYPE
            {
                Some(l)
            } else {
                None
            }
        });
        if passthrough_layer.is_none()
            && info.environment_blend_mode != EnvironmentBlendMode::ALPHA_BLEND
        {
            // No passthrough layer, we can just pass the frame to openxr.
            return xrcvt(unsafe {
                (instance.fp().end_frame)(session.as_handle(), &info.as_raw())
            });
        }

        let passthrough = data.get_or_add_passthrough()?;
        if let Some(layer) = passthrough_layer {
            // SAFETY: we checked the type is CompositionLayerPassthroughHTC::TYPE
            let layer: &openxr::sys::CompositionLayerPassthroughHTC =
                unsafe { &*(layer as *const openxr::sys::CompositionLayerBaseHeader).cast() };
            let obj = layer.passthrough.registered_with_hook()?;
            let data = obj.hook();
            if !Arc::ptr_eq(&passthrough, &data.inner) {
                warn!("app supplied passthrough object is invalid");
                return Err(XrErr::ERROR_VALIDATION_FAILURE);
            }
        }

        let SessionState::Running {
            image_index,
            view_type,
            passthrough,
            images,
            swapchain,
            size,
            ..
        } = &mut data.state
        else {
            return Err(XrErr::ERROR_SESSION_NOT_RUNNING);
        };
        let (_, camera) = passthrough.as_mut().unwrap();

        let (view_state_flags, view_locations) =
            xr_vk_session.locate_views(*view_type, info.display_time, &data.space)?;
        if !view_state_flags
            .contains(ViewStateFlags::POSITION_VALID | ViewStateFlags::ORIENTATION_VALID)
            || image_index.is_none()
        {
            if image_index.is_none() {
                warn!("end_frame called without begin_frame");
            } else {
                debug!("Pose or orientation invalid {view_state_flags:?}, skip passthrough layer");
            }
            let layers = info
                .layers
                .iter()
                .copied()
                .filter(|l| {
                    l.is_none_or(|l| l.ty != openxr::sys::CompositionLayerPassthroughHTC::TYPE)
                })
                .collect::<Vec<_>>();
            // Also we need to disable ALPHA_BLEND environment_blend_mode.
            let mut info = info.as_raw();
            info.layer_count = layers.len() as _;
            info.layers = layers.as_ptr() as *const _;
            info.environment_blend_mode = EnvironmentBlendMode::OPAQUE;
            return xrcvt(unsafe { (instance.fp().end_frame)(session.as_handle(), &info) });
        }
        let image_index = image_index.take().unwrap();
        // Copy camera image to swapchain
        let mut cmdbuf = AutoCommandBufferBuilder::primary(
            data.cmdbuf_allocator.clone(),
            data.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .map_err(|e| {
            warn!("Failed to create command buffer {e:#}");
            XrErr::ERROR_RUNTIME_FAILURE
        })?;
        let camera_frame = camera.camera.frame();
        camera.pp.maybe_postprocess(&camera_frame).map_err(|e| {
            warn!("Failed to postprocess camera frame {e:#}");
            XrErr::ERROR_RUNTIME_FAILURE
        })?;
        let (camera_image, fut) = camera.pp.image();
        let camera_extent = camera.pp.image_extent();
        cmdbuf
            .blit_image(BlitImageInfo {
                src_image: camera_image.clone(),
                dst_image: images[image_index as usize].clone(),
                regions: smallvec![ImageBlit {
                    src_subresource: camera_image.subresource_layers(),
                    src_offsets: [[0, 0, 0], [camera_extent[0], camera_extent[1], 1],],
                    dst_subresource: images[image_index as usize].subresource_layers(),
                    dst_offsets: [[0, 0, 0], [size.x * 2, size.y, 1],],
                    ..Default::default()
                }],
                ..BlitImageInfo::new(camera_image.clone(), images[image_index as usize].clone())
            })
            .map_err(|e| {
                warn!("Failed to blit image {e:#}");
                XrErr::ERROR_RUNTIME_FAILURE
            })?;
        let cmdbuf = cmdbuf.build().map_err(|e| {
            warn!("Failed to build command buffer {e:#}");
            XrErr::ERROR_RUNTIME_FAILURE
        })?;
        fut.then_execute(data.queue.clone(), cmdbuf)
            .map_err(|e| {
                warn!("Failed to execute command buffer {e:#}");
                XrErr::ERROR_RUNTIME_FAILURE
            })?
            .then_signal_fence_and_flush()
            .map_err(|e| {
                warn!("Failed to flush command buffer {e:#}");
                XrErr::ERROR_RUNTIME_FAILURE
            })?
            .wait(None)
            .map_err(|e| {
                warn!("Failed to wait for fence {e:#}");
                XrErr::ERROR_RUNTIME_FAILURE
            })?;
        assert_eq!(view_locations.len(), 2);
        swapchain.release_image()?;
        log::trace!("{:?}", view_locations[0].fov);
        log::trace!("{:?}", view_locations[1].fov);
        log::trace!("{:?}", view_locations[0].pose);
        log::trace!("{:?}", view_locations[1].pose);
        let views = [
            openxr::CompositionLayerProjectionView::new()
                .sub_image(
                    SwapchainSubImage::new()
                        .swapchain(swapchain)
                        .image_rect(Rect2Di {
                            offset: Offset2Di { x: 0, y: 0 },
                            extent: Extent2Di {
                                width: size.x as _,
                                height: size.y as _,
                            },
                        }),
                )
                .pose(view_locations[0].pose)
                .fov(view_locations[0].fov),
            openxr::CompositionLayerProjectionView::new()
                .sub_image(
                    SwapchainSubImage::new()
                        .swapchain(swapchain)
                        .image_rect(Rect2Di {
                            offset: Offset2Di {
                                x: size.x as _,
                                y: 0,
                            },
                            extent: Extent2Di {
                                width: size.x as _,
                                height: size.y as _,
                            },
                        }),
                )
                .pose(view_locations[1].pose)
                .fov(view_locations[1].fov),
        ];

        let replaced_passthrough_layer = openxr::CompositionLayerProjection::new()
            .space(&data.space)
            .layer_flags(CompositionLayerFlags::BLEND_TEXTURE_SOURCE_ALPHA)
            .views(&views);
        let new_layer = unsafe {
            std::mem::transmute::<
                Option<&openxr::sys::CompositionLayerProjection>,
                Option<&openxr::sys::CompositionLayerBaseHeader>,
            >(Some(replaced_passthrough_layer.as_raw()))
        };
        let pos = info.layers.iter().position(|l| {
            l.is_some_and(|l| l.ty == openxr::sys::CompositionLayerPassthroughHTC::TYPE)
        });
        let new_layers = if let Some(pos) = pos {
            let mut layers = info.layers.to_vec();
            layers[pos] = new_layer;
            layers
        } else {
            // ALPHA_BLEND mode, insert the camera layer at the 0th position.
            assert_eq!(
                info.environment_blend_mode,
                EnvironmentBlendMode::ALPHA_BLEND
            );
            let mut layers = vec![new_layer];
            layers.extend(info.layers.iter());
            layers
        };
        let mut info2 = info.as_raw();
        if info2.environment_blend_mode == EnvironmentBlendMode::ALPHA_BLEND {
            info2.environment_blend_mode = EnvironmentBlendMode::OPAQUE;
        }
        info2.layers = new_layers.as_ptr() as *const _;
        xrcvt(unsafe { (instance.fp().end_frame)(session.as_handle(), &info2) })
    }

    /// Returns the passthrough object if one is already attached. Otherwise try to create one.
    /// Returns `Ok(None)` if the current session is not supported.
    pub(super) fn maybe_get_or_add_passthrough(
        &mut self,
    ) -> Result<Option<Arc<PassthroughInner>>, XrErr> {
        let Some(inner) = self.inner.as_mut() else {
            return Ok(None);
        };
        Ok(Some(inner.get_or_add_passthrough()?))
    }
}

impl quark::Hook for SessionData {
    type Target = openxr::sys::Session;
    type Factory = quark::FactoryOf<Self>;
    fn on_create(
        session: &AnySession,
        create_info: quark::types::SessionCreateInfo,
    ) -> XrResult<Self> {
        debug!("on_create(): OpenXR session");
        // Do we have vulkan?
        let AnySession::Vulkan(xr_vk_session) = session else {
            warn!("Not a vulkan session, don't know how to handle it");
            return Ok(Self::default());
        };
        let gb = create_info.graphics_binding.unwrap();
        let quark::types::GraphicsBinding::Vulkan(gb) = gb else {
            warn!("Session type is vulkan, but no graphics binding offered");
            return Err(XrErr::ERROR_VALIDATION_FAILURE);
        };
        let instance = xr_vk_session
            .instance()
            .as_handle()
            .registered_with_hook()?;
        let api_version = instance
            .hook()
            .instance_api_version()
            .get(&(gb.instance as u64));
        let api_version = if let Some(v) = api_version {
            *v
        } else {
            let req = xr_vk_session
                .instance()
                .graphics_requirements::<openxr::Vulkan>(create_info.system_id)?;
            // This vulkan instance wasn't created via `xrCreateVulkanInstanceKHR`, we have to
            // assume minimum supported api version.
            (req.min_api_version_supported.into_raw() as u32).into()
        };
        log::info!("Vulkan API version: {api_version}");
        let enabled_vk_instance_extensions = InstanceExtensions {
            khr_external_memory_capabilities: true,
            khr_get_physical_device_properties2: true,
            khr_xcb_surface: true,
            ..Default::default()
        };

        let vk_create_info = vulkano::instance::InstanceCreateInfo {
            enabled_extensions: &enabled_vk_instance_extensions,
            max_api_version: Some(api_version),
            ..Default::default()
        };
        let vk_instance = unsafe {
            vulkano::instance::Instance::from_handle_borrowed(
                &super::VULKAN_LIBRARY,
                ash::vk::Handle::from_raw(gb.instance as usize as u64),
                &vk_create_info,
            )
        };
        let physical_device = match unsafe {
            vulkano::device::physical::PhysicalDevice::from_handle(
                &vk_instance,
                ash::vk::Handle::from_raw(gb.physical_device as usize as u64),
            )
        } {
            Ok(pd) => pd,
            Err(e) => {
                warn!("Failed to wrap vulkan physical device {e}");
                return Ok(Self::default());
            }
        };
        let queues = vec![0.0; gb.queue_index as usize + 1];
        let enabled_vk_device_extensions = DeviceExtensions {
            khr_copy_commands2: true,
            ..Default::default()
        };
        let vk_create_info = vulkano::device::DeviceCreateInfo {
            queue_create_infos: &[QueueCreateInfo {
                queue_family_index: gb.queue_family_index,
                queues: &queues,
                ..Default::default()
            }],
            enabled_extensions: &enabled_vk_device_extensions,
            ..Default::default()
        };
        let device = unsafe {
            vulkano::device::Device::from_handle_borrowed(
                &physical_device,
                ash::vk::Handle::from_raw(gb.device as usize as u64),
                &vk_create_info,
            )
        };
        let queue = unsafe {
            let mut queue = ash::vk::Queue::null();
            (device.fns().v1_0.get_device_queue)(
                device.handle(),
                gb.queue_family_index,
                gb.queue_index,
                &mut queue,
            );
            vulkano::device::Queue::from_handle(
                &device,
                queue,
                &DeviceQueueInfo {
                    queue_family_index: gb.queue_family_index,
                    queue_index: gb.queue_index,
                    ..Default::default()
                },
                Arc::new(DefaultQueueMutex::new()),
            )
        };

        let allocator = Arc::new(StandardMemoryAllocator::new(&device, &Default::default()));
        let cmdbuf_allocator = Arc::new(StandardCommandBufferAllocator::new(
            &device,
            &Default::default(),
        ));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            &device,
            &Default::default(),
        ));
        let space = xr_vk_session
            .create_reference_space(ReferenceSpaceType::STAGE, openxr::Posef::IDENTITY)?;

        Ok(Self {
            inner: Some(SessionDataInner {
                device,
                queue,
                state: SessionState::Idle { passthrough: None },
                space,
                system_id: create_info.system_id,
                allocator,
                cmdbuf_allocator,
                descriptor_set_allocator,
            }),
        })
    }
}

pub(super) unsafe extern "system" fn begin_session(
    session: openxr::sys::Session,
    info: *const openxr::sys::SessionBeginInfo,
) -> XrErr {
    debug!("begin session {:#x}", session.into_raw());
    let mut wrapped_session = try_xr!(session.registered_with_hook_mut());
    let instance = try_xr!(quark::find_instance(session));
    let info = &unsafe { *info };
    let wrapped_instance = try_xr!(instance.registered_with_hook());
    let (data, wrapped_session) = wrapped_session.both();
    let Some(data) = &mut data.inner else {
        // We are not wrapping this session, passed it through.
        debug!("Unhandled session");
        return unsafe { (wrapped_instance.get().fp().begin_session)(session, info) };
    };
    let quark::types::AnySession::Vulkan(xr_vk_session) = wrapped_session else {
        unreachable!()
    };
    if data.is_running() {
        return XrErr::ERROR_SESSION_RUNNING;
    }
    // Transition first, if `session.begin` failed then the session will still not be running, and
    // we don't need to do anything.
    try_xr!(xr_vk_session.begin(info.primary_view_configuration_type));
    try_xr!(data.begin(xr_vk_session, info.primary_view_configuration_type));
    XrErr::SUCCESS
}

pub(super) unsafe extern "system" fn end_session(raw_session: openxr::sys::Session) -> XrErr {
    debug!("end session {:#x}", raw_session.into_raw());
    let instance = try_xr!(quark::find_instance(raw_session));
    let mut session = try_xr!(raw_session.registered_with_hook_mut());
    let (data, session) = session.both();
    let Some(data) = &mut data.inner else {
        // We are not wrapping this session, passed it through.
        debug!("Unhandled session, like passthrough extension wasn't enabled");
        return unsafe { (try_xr!(instance.registered()).fp().end_session)(raw_session) };
    };
    if !data.is_running() {
        return XrErr::ERROR_SESSION_NOT_RUNNING;
    }
    try_xr!(session.end());
    data.end();
    XrErr::SUCCESS
}

pub(super) unsafe extern "system" fn begin_frame(
    raw_session: openxr::sys::Session,
    info: *mut openxr::sys::FrameBeginInfo,
) -> XrErr {
    trace!("begin frame {:#x} before", raw_session.into_raw());
    let instance = try_xr!(quark::find_instance(raw_session));
    // Lock the registry entry for the xrSession. this is because xrBeginFrame might unblock
    // a currently blocked xrWaitFrame. Our override for xrWaitFrame will acquire the xrSession
    // from the registry to update `should_render`. We want to make sure it will only get the lock
    // after we have set `should_render` to None.
    let mut session = try_xr!(raw_session.registered_with_hook_mut());
    let ret = unsafe { (try_xr!(instance.registered()).fp().begin_frame)(raw_session, info) };
    if ret != XrErr::SUCCESS {
        return ret;
    }
    trace!("begin frame {:#x} after", raw_session.into_raw());
    let data = session.hook();
    let Some(data) = &mut data.inner else {
        // We are not wrapping this session, passed it through.
        debug!("Unhandled session");
        return XrErr::SUCCESS;
    };
    let SessionState::Running {
        image_index,
        swapchain,
        ..
    } = &mut data.state
    else {
        return XrErr::ERROR_SESSION_NOT_RUNNING;
    };
    if image_index.is_some() {
        // App might called 2 begin frame in a row, or maybe its last end_frame doesn't have
        // a passthrough layer, etc. we can use the image we alredy have.
        return XrErr::SUCCESS;
    }
    *image_index = Some(try_xr!(swapchain.acquire_image()));
    try_xr!(swapchain.wait_image(openxr::Duration::INFINITE));
    XrErr::SUCCESS
}

pub(super) unsafe extern "system" fn end_frame(
    raw_session: openxr::sys::Session,
    info: *const openxr::sys::FrameEndInfo,
) -> XrErr {
    debug!("end frame {:#x}", raw_session.into_raw());
    let instance = try_xr!(quark::find_instance(raw_session));
    let instance = try_xr!(instance.registered());
    let mut session = try_xr!(raw_session.registered_with_hook_mut());
    let info = unsafe { FrameEndInfo::from_raw(&*info) };
    let (data, session) = session.both();
    try_xr!(unsafe { data.end_frame(session, &instance, &info) });
    XrErr::SUCCESS
}

pub(super) unsafe extern "system" fn wait_frame(
    raw_session: openxr::sys::Session,
    wait_info: *const openxr::sys::FrameWaitInfo,
    frame_state: *mut openxr::sys::FrameState,
) -> XrErr {
    trace!("wait frame {:#x}, before", raw_session.into_raw());
    let fp = {
        // Can't keep the `registered` high-level instance object, because it
        // locks the object registry. but `xrWaitFrame` must be callable from any thread, while
        // `Begin/EndFrame` might be called concurrently from other threads, which needs this
        // registry entry lock too, and `xrWaitFrame` might enter into wait.
        let instance = try_xr!(quark::find_instance(raw_session));
        try_xr!(instance.registered()).fp().wait_frame
    };

    // We didn't keep the frame waiter given by openxr crate, just call the raw function.
    unsafe { (fp)(raw_session, wait_info, frame_state) }

    // We don't check if `should_render` here. Because we need to be prepared to deal with a
    // misbehaving application that submits frames dispite `should_render` being false.
}
