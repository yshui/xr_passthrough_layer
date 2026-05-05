use glam::{Mat4, Quat, Vec3, Vec4};
use std::{collections::HashSet, sync::Arc, thread::JoinHandle};
use winit::{
    event::WindowEvent,
    event_loop::{EventLoop, EventLoopProxy},
};

use anyhow::{Context as _, Result};
use openxr::{
    CompositionLayerFlags, EnvironmentBlendMode, Extent2Di, FrameState, Offset2Di, Rect2Di,
    SwapchainSubImage, ViewConfigurationType, ViewStateFlags, sys::Handle,
};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
        PrimaryCommandBufferAbstract, RenderPassBeginInfo, SubpassBeginInfo, SubpassContents,
        SubpassEndInfo,
        allocator::{CommandBufferAllocator, StandardCommandBufferAllocator},
    },
    descriptor_set::{
        DescriptorBufferInfo, DescriptorSet, WriteDescriptorSet,
        allocator::StandardDescriptorSetAllocator,
    },
    device::{Device, Queue},
    format::ClearValue,
    image::{
        Image, ImageCreateInfo, ImageLayout, ImageUsage, SampleCount,
        view::{ImageView, ImageViewCreateInfo},
    },
    instance::Instance,
    memory::allocator::{
        AllocationCreateInfo, MemoryAllocatePreference, MemoryAllocator, MemoryTypeFilter,
        StandardMemoryAllocator,
    },
    pipeline::{
        self, GraphicsPipeline, Pipeline, PipelineLayout, PipelineShaderStageCreateInfo,
        graphics::{
            GraphicsPipelineCreateInfo,
            color_blend::ColorBlendState,
            depth_stencil::{DepthState, DepthStencilState},
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            multisample::MultisampleState,
            rasterization::RasterizationState,
            subpass::PipelineSubpassType,
            vertex_input::{Vertex as _, VertexDefinition},
            viewport::{Viewport, ViewportState},
        },
    },
    render_pass::{
        AttachmentDescription, AttachmentLoadOp, AttachmentReference, AttachmentStoreOp,
        Framebuffer, FramebufferCreateInfo, RenderPass, RenderPassCreateInfo, Subpass,
        SubpassDescription,
    },
    shader::ShaderModule,
    swapchain::{SurfaceInfo, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo},
    sync::{GpuFuture, future::FenceSignalFuture},
};
use xr::XrContext as _;

#[derive(Clone)]
struct Window {
    swapchain: Arc<Swapchain>,
    inner: Arc<winit::window::Window>,
}

enum SessionState {
    Running(JoinHandle<()>),
    StopWaitting,
    StartWaitting(EventLoopProxy<AppMessage>),
    Idle(openxr::FrameWaiter),
    Invalid,
}

impl SessionState {
    fn start(&mut self, proxy: EventLoopProxy<AppMessage>) {
        let this = std::mem::replace(self, SessionState::Invalid);
        match this {
            SessionState::Running(_) | SessionState::StartWaitting(_) => {
                panic!("Session already running");
            }
            SessionState::StopWaitting => {
                *self = SessionState::StartWaitting(proxy);
            }
            SessionState::Idle(mut waiter) => {
                let handle = std::thread::spawn(move || {
                    log::info!("Frame waiter started");
                    loop {
                        log::debug!("Waiting for frame");
                        match waiter.wait() {
                            Ok(state) => {
                                log::debug!("Waiting for frame end");
                                proxy.send_event(AppMessage::Frame(state)).unwrap();
                            }
                            Err(openxr::sys::Result::ERROR_SESSION_NOT_RUNNING) => {
                                log::info!("session stopped, stop frame waiter");
                                break;
                            }
                            Err(e) => {
                                log::warn!("Frame waiter error: {e:#}");
                                break;
                            }
                        }
                    }
                    proxy.send_event(AppMessage::WaiterExited(waiter)).unwrap();
                });
                *self = SessionState::Running(handle);
            }
            SessionState::Invalid => unreachable!(),
        }
    }
    fn stop(&mut self, session: &openxr::Session<openxr::Vulkan>) -> bool {
        let this = std::mem::replace(self, SessionState::Invalid);
        match this {
            SessionState::Running(handle) => {
                session.end().unwrap();
                handle.join().unwrap();
                *self = SessionState::StopWaitting;
                true
            }
            SessionState::StartWaitting(_) => {
                *self = SessionState::StopWaitting;
                true
            }
            SessionState::Idle(_) | SessionState::StopWaitting => false,
            SessionState::Invalid => unreachable!(),
        }
    }
    fn ensure_stop(&mut self, session: &openxr::Session<openxr::Vulkan>) {
        if self.stop(session) {
            log::info!("Session stopped");
        } else {
            panic!("Session already stopped");
        }
    }
    fn is_stopped(&self) -> bool {
        match self {
            SessionState::Idle(_) | SessionState::StopWaitting => true,
            SessionState::Running(_) | SessionState::StartWaitting(_) => false,
            SessionState::Invalid => unreachable!(),
        }
    }
    fn put_frame_waiter(&mut self, waiter: openxr::FrameWaiter) {
        let this = std::mem::replace(self, SessionState::Invalid);
        match this {
            SessionState::Running(_) => {
                unreachable!("Got frame wait while running");
            }
            SessionState::Idle(_) => {
                unreachable!("Got frame wait while idle");
            }
            SessionState::StopWaitting => {
                *self = SessionState::Idle(waiter);
            }
            SessionState::StartWaitting(proxy) => {
                *self = SessionState::Idle(waiter);
                self.start(proxy);
            }
            SessionState::Invalid => unreachable!(),
        }
    }
    fn wait_time(&self) -> Option<std::time::Duration> {
        match self {
            SessionState::Running(_) => None,
            SessionState::StopWaitting => None,
            SessionState::StartWaitting(_) => None,
            SessionState::Idle(_) => Some(std::time::Duration::from_millis(100)),
            SessionState::Invalid => unreachable!(),
        }
    }
}

type FenceFut = Arc<FenceSignalFuture<Box<dyn GpuFuture + Send + Sync>>>;
struct App {
    exiting: bool,
    state: SessionState,
    proxy: EventLoopProxy<AppMessage>,
    allocator: Arc<dyn MemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    cmdbuf_allocator: Arc<dyn CommandBufferAllocator>,
    vertices: Subbuffer<[Vertex]>,
    indices: Subbuffer<[u32]>,
    cmdbufs: Vec<Arc<PrimaryAutoCommandBuffer>>,
    xr_cmdbufs: Vec<Arc<PrimaryAutoCommandBuffer>>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    instance: Arc<Instance>,
    window: Option<Window>,
    uniform_buffer_gpu_use_end: Vec<Option<FenceFut>>,
    previous_frame_end: Option<Box<dyn GpuFuture + Send + Sync>>,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    renderdoc: Option<renderdoc::RenderDoc<renderdoc::V141>>,
    latest_mvps: [Mat4; 2],
    uniform_buffers: Vec<Subbuffer<vs::MVP>>,
    xr_uniform_buffer: Subbuffer<vs::MVP>,
    frame_stream: openxr::FrameStream<openxr::Vulkan>,
    passthrough: Option<openxr::sys::PassthroughHTC>,
    render_start: std::time::Instant,

    xr: xr::OpenXr,
}

#[derive(
    vulkano::pipeline::graphics::vertex_input::Vertex,
    bytemuck::Pod,
    Clone,
    Copy,
    bytemuck::Zeroable,
)]
#[repr(C)]
struct Vertex {
    #[format(R32G32B32_SFLOAT)]
    in_position: [f32; 3],
    #[format(R32G32B32A32_SFLOAT)]
    in_color: [f32; 4],
}
const INDICES: [u32; 36] = [
    0, 1, 3, 1, 3, 5, // back
    3, 5, 6, 5, 6, 7, // top
    6, 7, 2, 7, 2, 4, // front
    2, 4, 1, 2, 1, 0, // bottom
    1, 4, 5, 4, 5, 7, // right
    0, 2, 3, 2, 3, 6, // left
];
const VERTICES: [Vertex; 8] = [
    // 0
    Vertex {
        in_position: [-0.1, -0.1, -0.1],
        in_color: [1., 0.0, 0.0, 1.],
    },
    // 1
    Vertex {
        in_position: [0.1, -0.1, -0.1],
        in_color: [0.0, 1., 0.0, 1.],
    },
    // 2
    Vertex {
        in_position: [-0.1, 0.1, -0.1],
        in_color: [0.0, 0.0, 1., 1.],
    },
    // 3
    Vertex {
        in_position: [-0.1, -0.1, 0.1],
        in_color: [1., 0.0, 0.0, 1.],
    },
    // 4
    Vertex {
        in_position: [0.1, 0.1, -0.1],
        in_color: [1., 1., 1., 1.],
    },
    // 5
    Vertex {
        in_position: [0.1, -0.1, 0.1],
        in_color: [0.0, 1., 0.0, 1.],
    },
    // 6
    Vertex {
        in_position: [-0.1, 0.1, 0.1],
        in_color: [0.0, 0.0, 1., 1.],
    },
    // 7
    Vertex {
        in_position: [0.1, 0.1, 0.1],
        in_color: [1., 1., 1., 1.],
    },
];

impl winit::application::ApplicationHandler<AppMessage> for App {
    fn about_to_wait(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        self.process_xr_events().unwrap();
        if let Some(dur) = self.state.wait_time() {
            event_loop.set_control_flow(winit::event_loop::ControlFlow::wait_duration(dur));
        } else {
            event_loop.set_control_flow(winit::event_loop::ControlFlow::Wait);
        }
        log::debug!("about to wait");
        if let Some(window) = &self.window {
            window.inner.request_redraw();
        }
    }
    fn exiting(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        self.state.stop(self.xr.xr_session());
    }
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        log::info!("resumed");
        let window = match event_loop.create_window(winit::window::Window::default_attributes()) {
            Ok(window) => Arc::new(window),
            Err(e) => {
                log::warn!("Failed to create window {e:#}");
                event_loop.exit();
                return;
            }
        };
        let Err(e) = self.setup_window(window.clone()) else {
            window.request_redraw();
            return;
        };
        log::warn!("Failed to setup rendering surface {e:#}");
        event_loop.exit();
    }
    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                self.xr.xr_session().request_exit().unwrap();
            }
            WindowEvent::RedrawRequested => {
                let Err(e) = self.redraw() else { return };
                log::warn!("Failed to redraw {e:#}");
                event_loop.exit();
            }
            _ => (),
        }
    }
    fn user_event(&mut self, event_loop: &winit::event_loop::ActiveEventLoop, msg: AppMessage) {
        match msg {
            AppMessage::Frame(state) if !state.should_render => {
                if !self.state.is_stopped() {
                    self.skip_frame(state).unwrap();
                }
            }
            AppMessage::Frame(state) => {
                if !self.state.is_stopped() {
                    self.render_and_submit(state).unwrap();
                }
            }
            AppMessage::WaiterExited(waiter) => {
                log::info!("Frame waiter exited");
                self.state.put_frame_waiter(waiter);
                if self.exiting {
                    event_loop.exit();
                }
            }
        }
    }
}

impl App {
    fn skip_frame(&mut self, state: FrameState) -> Result<(), openxr::sys::Result> {
        self.queue.with(|_| {
            self.frame_stream.begin()?;
            self.frame_stream.end(
                state.predicted_display_time,
                EnvironmentBlendMode::OPAQUE,
                &[],
            )
        })
    }
    fn process_xr_events(&mut self) -> Result<()> {
        let mut buf = openxr::EventDataBuffer::new();
        while let Some(event) = self.xr.xr_instance().poll_event(&mut buf)? {
            match event {
                openxr::Event::SessionStateChanged(state_change) => {
                    if state_change.state() == openxr::SessionState::READY {
                        log::info!("Session is ready");
                        self.xr
                            .xr_session()
                            .begin(ViewConfigurationType::PRIMARY_STEREO)?;
                        self.state.start(self.proxy.clone());
                    } else if state_change.state() == openxr::SessionState::EXITING {
                        log::info!("Session is exiting");
                        self.exiting = true;
                        break;
                    } else if state_change.state() == openxr::SessionState::STOPPING {
                        self.state.ensure_stop(self.xr.xr_session());
                    } else {
                        log::info!("Session state changed: {:?}", state_change.state());
                    }
                }
                _ => {
                    log::info!("Event: {:?}", std::mem::discriminant(&event));
                }
            }
        }
        Ok(())
    }
    fn setup_window(&mut self, window: Arc<winit::window::Window>) -> Result<()> {
        const PREFERRED_FORMATS: &[vulkano::format::Format] = &[
            vulkano::format::Format::B8G8R8_UNORM,
            vulkano::format::Format::R8G8B8_UNORM,
            vulkano::format::Format::B8G8R8A8_UNORM,
            vulkano::format::Format::R8G8B8A8_UNORM,
        ];
        log::info!("setting up window");
        let surface = vulkano::swapchain::Surface::from_window(&self.instance, &window)?;
        let surface_capabilities = self
            .device
            .physical_device()
            .surface_capabilities(&surface, &SurfaceInfo::default())?;
        let swapchain_formats = self
            .device
            .physical_device()
            .surface_formats(&surface, &SurfaceInfo::default())?
            .into_iter()
            .map(|(f, _)| f)
            .collect::<HashSet<_>>();
        log::info!("{swapchain_formats:?}");
        let swapchain_format = PREFERRED_FORMATS
            .iter()
            .find(|f| swapchain_formats.contains(f))
            .context("cannot find a suitable format for swapchain images")?;
        let (swapchain, images) = vulkano::swapchain::Swapchain::new(
            &self.device,
            &surface,
            &vulkano::swapchain::SwapchainCreateInfo {
                min_image_count: surface_capabilities.min_image_count.max(2),
                image_format: *swapchain_format,
                image_extent: window.inner_size().into(),
                image_usage: ImageUsage::TRANSFER_DST | ImageUsage::COLOR_ATTACHMENT,
                composite_alpha: vulkano::swapchain::CompositeAlpha::Opaque,
                present_mode: vulkano::swapchain::PresentMode::Fifo,
                ..Default::default()
            },
        )?;
        self.recreate_window_command_buffers(&images, &swapchain)?;
        self.window = Some(Window {
            swapchain,
            inner: window,
        });
        Ok(())
    }

    fn recreate_swapchain(&mut self) -> Result<()> {
        log::info!("recreating swapchain");

        let Some(window) = &mut self.window else {
            panic!("recreate non-existent swapchain")
        };
        let (swapchain, images) = window.swapchain.recreate(&SwapchainCreateInfo {
            image_extent: window.inner.inner_size().into(),
            ..window.swapchain.create_info()
        })?;
        log::info!("swapchain recreated");
        self.recreate_window_command_buffers(&images, &swapchain)?;
        self.window.as_mut().unwrap().swapchain = swapchain;
        Ok(())
    }

    fn redraw(&mut self) -> Result<()> {
        if self.window.is_none() {
            return Ok(());
        }
        if let Some(renderdoc) = &mut self.renderdoc {
            log::debug!("Starting frame capture");
            renderdoc.start_frame_capture(std::ptr::null(), std::ptr::null());
        }
        log::debug!("Debug rendering wait");
        let (image_index, future) = loop {
            match vulkano::swapchain::acquire_next_image(
                self.window.as_ref().unwrap().swapchain.clone(),
                Some(std::time::Duration::from_secs(0)),
            ) {
                Ok((image_index, false, future)) => break (image_index, future),
                Ok((_, true, future)) => {
                    self.recreate_swapchain()?;
                    self.previous_frame_end = Some(Box::new(future));
                    continue;
                }
                Err(vulkano::Validated::Error(vulkano::VulkanError::OutOfDate)) => {
                    self.recreate_swapchain()?;
                    continue;
                }
                Err(vulkano::Validated::Error(vulkano::VulkanError::Timeout))
                | Err(vulkano::Validated::Error(vulkano::VulkanError::NotReady)) => {
                    return Ok(());
                }
                Err(e) => return Err(e.into()),
            };
        };
        log::debug!("Debug rendering");
        {
            // Last render to the same image should have finished, since we got this image again.
            if let Some(last_use) = self.uniform_buffer_gpu_use_end[image_index as usize].take() {
                last_use.wait(None).unwrap();
            }
            let mut mvp = self.uniform_buffers[image_index as usize].write().unwrap();
            mvp.mvps = self.latest_mvps.map(|m| m.to_cols_array_2d());
        }
        let mut previous_future = self.previous_frame_end.take().unwrap();
        previous_future.cleanup_finished();
        let future = (Box::new(future.join(previous_future).then_execute(
            self.queue.clone(),
            self.cmdbufs[image_index as usize].clone(),
        )?) as Box<dyn GpuFuture + Send + Sync>)
            .then_signal_fence();
        let future = Arc::new(future);
        self.uniform_buffer_gpu_use_end[image_index as usize] = Some(future.clone());
        let window = self.window.as_ref().unwrap();
        window.inner.pre_present_notify();
        self.previous_frame_end = Some(Box::new(
            future
                .then_swapchain_present(
                    self.queue.clone(),
                    SwapchainPresentInfo::new(window.swapchain.clone(), image_index),
                )
                .then_signal_fence_and_flush()?,
        ));
        if let Some(renderdoc) = &mut self.renderdoc {
            log::debug!("End frame capture");
            renderdoc.end_frame_capture(std::ptr::null(), std::ptr::null());
        }
        log::debug!("Debug rendering end");
        Ok(())
    }

    fn recreate_window_command_buffers(
        &mut self,
        images: &[Arc<Image>],
        swapchain: &vulkano::swapchain::Swapchain,
    ) -> Result<()> {
        let vs_main = self.vs.entry_point("main").unwrap();
        let fs_main = self.fs.entry_point("main").unwrap();
        let stages = [
            PipelineShaderStageCreateInfo::new(&vs_main),
            PipelineShaderStageCreateInfo::new(&fs_main),
        ];
        let layout = PipelineLayout::from_stages(&self.device, &stages)?;
        let extent = swapchain.image_extent();
        let depth_image = Image::new(
            &self.allocator,
            &ImageCreateInfo {
                format: vulkano::format::Format::D32_SFLOAT,
                extent: [extent[0], extent[1], 1],
                samples: SampleCount::Sample1,
                array_layers: 1,
                usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                allocate_preference: MemoryAllocatePreference::Unknown,
                ..Default::default()
            },
        )?;
        let depth_image =
            ImageView::new(&depth_image, &ImageViewCreateInfo::from_image(&depth_image))?;
        assert!(self.uniform_buffers.len() == self.uniform_buffer_gpu_use_end.len());
        if self.uniform_buffers.len() < images.len() {
            for _ in self.uniform_buffers.len()..images.len() {
                self.uniform_buffers.push(Buffer::new_sized::<vs::MVP>(
                    &self.allocator,
                    &BufferCreateInfo {
                        usage: BufferUsage::UNIFORM_BUFFER,
                        ..Default::default()
                    },
                    &AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                            | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        allocate_preference: MemoryAllocatePreference::Unknown,
                        ..Default::default()
                    },
                )?);
                self.uniform_buffer_gpu_use_end.push(None);
            }
        }
        let render_pass = vulkano::single_pass_renderpass!(
            &self.device,
            attachments: {
                color: {
                    format: swapchain.image_format(),
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                    initial_layout: ImageLayout::Undefined,
                    final_layout: ImageLayout::ColorAttachmentOptimal,
                },
                depth_stencil: {
                    format: vulkano::format::Format::D32_SFLOAT,
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                    initial_layout: ImageLayout::Undefined,
                    final_layout: ImageLayout::DepthStencilAttachmentOptimal,
                },
            },
            pass: {
                color: [color],
                depth_stencil: {depth_stencil},
            }
        )?;
        let pipeline = GraphicsPipeline::new(
            &self.device,
            None,
            &GraphicsPipelineCreateInfo {
                vertex_input_state: Some(&Vertex::per_vertex().definition(&vs_main)?),
                stages: &stages,
                input_assembly_state: Some(&InputAssemblyState {
                    topology: PrimitiveTopology::TriangleList,
                    ..Default::default()
                }),
                viewport_state: Some(&ViewportState {
                    viewports: &[Viewport {
                        offset: [0., 0.],
                        extent: [extent[0] as f32, extent[1] as f32],
                        ..Default::default()
                    }],
                    ..Default::default()
                }),
                depth_stencil_state: Some(&DepthStencilState {
                    depth: Some(DepthState::simple()),
                    ..Default::default()
                }),
                multisample_state: Some(&Default::default()),
                color_blend_state: Some(&ColorBlendState {
                    attachments: &[Default::default()],
                    ..Default::default()
                }),
                rasterization_state: Some(&RasterizationState::default()),
                subpass: Some(PipelineSubpassType::BeginRenderPass(
                    &Subpass::new(&render_pass, 0).unwrap(),
                )),
                ..GraphicsPipelineCreateInfo::new(&layout)
            },
        )?;
        log::info!("Creating cmdbufs. extent: {extent:?}");
        self.cmdbufs = images
            .iter()
            .zip(self.uniform_buffers.iter().take(images.len()))
            .map(|(i, mvp)| {
                let descriptor_set = DescriptorSet::new(
                    &self.descriptor_set_allocator,
                    pipeline.layout().set_layouts().first().unwrap(),
                    &[WriteDescriptorSet::buffer(
                        0,
                        &DescriptorBufferInfo {
                            buffer: Some(mvp.buffer()),
                            ..Default::default()
                        },
                    )],
                    &[],
                )?;
                let framebuffer = Framebuffer::new(
                    &render_pass,
                    &FramebufferCreateInfo {
                        attachments: &[
                            &ImageView::new(i, &ImageViewCreateInfo::from_image(i))?,
                            &depth_image,
                        ],
                        ..Default::default()
                    },
                )?;
                let mut cmdbuf = AutoCommandBufferBuilder::primary(
                    self.cmdbuf_allocator.clone(),
                    self.queue.queue_family_index(),
                    CommandBufferUsage::MultipleSubmit,
                )?;
                cmdbuf
                    .begin_render_pass(
                        RenderPassBeginInfo {
                            clear_values: vec![
                                Some(ClearValue::Float([0.0, 0.0, 0.0, 0.0])),
                                Some(ClearValue::Depth(1.0)),
                            ],
                            ..RenderPassBeginInfo::framebuffer(framebuffer)
                        },
                        SubpassBeginInfo {
                            contents: SubpassContents::Inline,
                            ..Default::default()
                        },
                    )?
                    .bind_pipeline_graphics(pipeline.clone())?
                    .bind_descriptor_sets(
                        pipeline::PipelineBindPoint::Graphics,
                        pipeline.layout().clone(),
                        0,
                        descriptor_set.clone(),
                    )?
                    .bind_vertex_buffers(0, self.vertices.clone())?
                    .bind_index_buffer(self.indices.clone())?;
                unsafe { cmdbuf.draw_indexed(INDICES.len() as u32, 1, 0, 0, 0)? }
                    .end_render_pass(SubpassEndInfo::default())?;
                cmdbuf.build()
            })
            .collect::<Result<_, _>>()?;
        Ok(())
    }
    fn new(
        mut xr: xr::OpenXr,
        proxy: EventLoopProxy<AppMessage>,
        frame_waiter: openxr::FrameWaiter,
        frame_stream: openxr::FrameStream<openxr::Vulkan>,
    ) -> Result<Self> {
        let renderdoc = renderdoc::RenderDoc::<renderdoc::V141>::new()
            .map_err(|e| log::info!("cannot load renderdoc: {e}"))
            .ok();
        let (device, queue) = xr.vk_device();
        let xr::RenderInfo {
            swapchain_images,
            depth_swapchain_images,
            ..
        } = xr.render_info();
        let representative_image = &swapchain_images[0];
        let render_pass = RenderPass::new(
            &device,
            &RenderPassCreateInfo {
                attachments: &[
                    AttachmentDescription {
                        format: representative_image.format(),
                        store_op: AttachmentStoreOp::Store,
                        load_op: AttachmentLoadOp::Clear,
                        final_layout: ImageLayout::ColorAttachmentOptimal,
                        ..Default::default()
                    },
                    AttachmentDescription {
                        format: vulkano::format::Format::D32_SFLOAT,
                        store_op: AttachmentStoreOp::Store,
                        load_op: AttachmentLoadOp::Clear,
                        final_layout: ImageLayout::DepthStencilAttachmentOptimal,
                        ..Default::default()
                    },
                ],
                subpasses: &[SubpassDescription {
                    color_attachments: &[Some(AttachmentReference {
                        attachment: 0,
                        layout: ImageLayout::ColorAttachmentOptimal,
                        ..Default::default()
                    })],
                    view_mask: 0b11,
                    depth_stencil_attachment: Some(&Some(AttachmentReference {
                        attachment: 1,
                        layout: ImageLayout::DepthStencilAttachmentOptimal,
                        ..Default::default()
                    })),
                    ..Default::default()
                }],
                correlated_view_masks: &[0b11],
                ..Default::default()
            },
        )?;
        let allocator = Arc::new(StandardMemoryAllocator::new(&device, &Default::default()));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            &device,
            &Default::default(),
        ));
        let vs = vs::load(&device)?;
        let fs = fs::load(&device)?;
        let mvp = Buffer::new_sized::<vs::MVP>(
            &allocator,
            &BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                allocate_preference: MemoryAllocatePreference::Unknown,
                ..Default::default()
            },
        )?;
        let vs_main = vs.entry_point("main").unwrap();
        let fs_main = fs.entry_point("main").unwrap();
        let stages = [
            PipelineShaderStageCreateInfo::new(&vs_main),
            PipelineShaderStageCreateInfo::new(&fs_main),
        ];
        let layout = PipelineLayout::from_stages(&device, &stages)?;
        let extent = representative_image.extent();
        let pipeline = GraphicsPipeline::new(
            &device,
            None,
            &GraphicsPipelineCreateInfo {
                vertex_input_state: Some(&Vertex::per_vertex().definition(&vs_main)?),
                stages: &stages,
                input_assembly_state: Some(&InputAssemblyState {
                    topology: PrimitiveTopology::TriangleList,
                    ..Default::default()
                }),
                viewport_state: Some(&ViewportState {
                    viewports: &[Viewport {
                        offset: [0., 0.],
                        extent: [extent[0] as f32, extent[1] as f32],
                        ..Default::default()
                    }],
                    ..Default::default()
                }),
                depth_stencil_state: Some(&DepthStencilState {
                    depth: Some(DepthState::simple()),
                    ..Default::default()
                }),
                multisample_state: Some(&MultisampleState {
                    rasterization_samples: representative_image.samples(),
                    ..Default::default()
                }),
                color_blend_state: Some(&ColorBlendState {
                    attachments: &[Default::default()],
                    ..Default::default()
                }),
                rasterization_state: Some(&RasterizationState::default()),
                subpass: Some(PipelineSubpassType::BeginRenderPass(
                    &Subpass::new(&render_pass, 0).unwrap(),
                )),
                ..GraphicsPipelineCreateInfo::new(&layout)
            },
        )?;
        // Vertices for a cube
        let vertices = Buffer::from_iter::<Vertex, _>(
            &allocator,
            &BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                allocate_preference: MemoryAllocatePreference::Unknown,
                ..Default::default()
            },
            VERTICES,
        )?;
        let indices = Buffer::from_iter::<u32, _>(
            &allocator,
            &BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                allocate_preference: MemoryAllocatePreference::Unknown,
                ..Default::default()
            },
            INDICES,
        )?;
        let cmdbuf_allocator = Arc::new(StandardCommandBufferAllocator::new(
            &device,
            &Default::default(),
        ));
        let depth_images: Vec<_> = if let Some(depth_images) = depth_swapchain_images {
            depth_images
                .iter()
                .map(|i| ImageView::new(i, &ImageViewCreateInfo::from_image(i)))
                .collect::<Result<_, _>>()?
        } else {
            let depth_image = Image::new(
                &allocator,
                &ImageCreateInfo {
                    format: vulkano::format::Format::D32_SFLOAT,
                    extent,
                    samples: representative_image.samples(),
                    array_layers: representative_image.array_layers(),
                    usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT,
                    ..Default::default()
                },
                &AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    allocate_preference: MemoryAllocatePreference::Unknown,
                    ..Default::default()
                },
            )?;
            let depth_image =
                ImageView::new(&depth_image, &ImageViewCreateInfo::from_image(&depth_image))?;
            swapchain_images
                .iter()
                .map(|_| depth_image.clone())
                .collect()
        };
        let descriptor_set = DescriptorSet::new(
            &descriptor_set_allocator,
            pipeline.layout().set_layouts().first().unwrap(),
            &[WriteDescriptorSet::buffer(
                0,
                &DescriptorBufferInfo {
                    buffer: Some(mvp.buffer()),
                    ..Default::default()
                },
            )],
            &[],
        )?;
        let cmdbufs = swapchain_images
            .iter()
            .zip(depth_images)
            .map(|(i, di)| {
                let framebuffer = Framebuffer::new(
                    &render_pass,
                    &FramebufferCreateInfo {
                        attachments: &[
                            &ImageView::new(i, &ImageViewCreateInfo::from_image(i))?,
                            &di,
                        ],
                        ..Default::default()
                    },
                )?;
                let mut cmdbuf = AutoCommandBufferBuilder::primary(
                    cmdbuf_allocator.clone(),
                    queue.queue_family_index(),
                    CommandBufferUsage::MultipleSubmit,
                )?;
                cmdbuf
                    .begin_render_pass(
                        RenderPassBeginInfo {
                            clear_values: vec![
                                Some(ClearValue::Float([0.0, 0.0, 0.0, 0.0])),
                                Some(ClearValue::Depth(1.0)),
                            ],
                            ..RenderPassBeginInfo::framebuffer(framebuffer)
                        },
                        SubpassBeginInfo {
                            contents: SubpassContents::Inline,
                            ..Default::default()
                        },
                    )?
                    .bind_pipeline_graphics(pipeline.clone())?
                    .bind_descriptor_sets(
                        pipeline::PipelineBindPoint::Graphics,
                        pipeline.layout().clone(),
                        0,
                        descriptor_set.clone(),
                    )?
                    .bind_vertex_buffers(0, vertices.clone())?
                    .bind_index_buffer(indices.clone())?;
                unsafe { cmdbuf.draw_indexed(INDICES.len() as u32, 1, 0, 0, 0)? }
                    .end_render_pass(SubpassEndInfo::default())?;
                cmdbuf.build()
            })
            .collect::<Result<_, _>>()?;
        log::info!("xr swapchain extent {:?}", representative_image.extent());

        Ok(App {
            exiting: false,
            state: SessionState::Idle(frame_waiter),
            queue: queue.clone(),
            allocator,
            cmdbuf_allocator,
            descriptor_set_allocator,
            instance: xr.vk_instance(),
            previous_frame_end: Some(Box::new(vulkano::sync::now(device.clone()))),
            window: None,
            cmdbufs: Vec::new(),
            vertices,
            device,
            proxy,
            indices,
            vs,
            fs,
            renderdoc,
            xr_uniform_buffer: mvp,
            uniform_buffers: Vec::new(),
            uniform_buffer_gpu_use_end: Vec::new(),
            latest_mvps: [Mat4::IDENTITY, Mat4::IDENTITY],
            xr_cmdbufs: cmdbufs,
            xr,
            passthrough: None,
            frame_stream,
            render_start: std::time::Instant::now(),
        })
    }
    fn render_and_submit(&mut self, state: FrameState) -> Result<()> {
        log::debug!("XR rendering");
        let delta = self.render_start.elapsed();
        let xr::RenderInfo {
            session: xr_session,
            swapchain,
            depth_swapchain,
            space,
            render_size,
            ..
        } = self.xr.render_info();
        self.queue.with(|_| self.frame_stream.begin())?;
        let (view_flags, views) = xr_session
            .locate_views(
                ViewConfigurationType::PRIMARY_STEREO,
                state.predicted_display_time,
                space,
            )
            .unwrap();
        if !view_flags.contains(ViewStateFlags::POSITION_VALID | ViewStateFlags::ORIENTATION_VALID)
        {
            log::warn!("View state is not valid");
            self.queue.with(|_| {
                self.frame_stream.end(
                    state.predicted_display_time,
                    EnvironmentBlendMode::OPAQUE,
                    &[],
                )
            })?;
            return Ok(());
        }
        log::trace!("{:?}", views[0].fov);
        log::trace!("{:?}", views[1].fov);
        log::trace!("{:?}", views[0].pose);
        log::trace!("{:?}", views[1].pose);
        let views: [_; 2] = (&views[..]).try_into()?;
        let image_index = self.queue.with(|_| swapchain.acquire_image())? as usize;
        swapchain.wait_image(openxr::Duration::INFINITE).unwrap();
        let depth_swapchain = depth_swapchain
            .map(|sc| {
                let i = self.queue.with(|_| sc.acquire_image())? as usize;
                assert_eq!(i, image_index);
                sc.wait_image(openxr::Duration::INFINITE)?;
                Ok::<_, openxr::sys::Result>(sc)
            })
            .transpose()?;
        // Convert views to mvp
        self.latest_mvps = [0, 1].map(|i| {
            let view = views[i];
            let translation = glam::Vec3::new(
                view.pose.position.x,
                view.pose.position.y,
                view.pose.position.z,
            );
            let rotation = glam::Quat::from_xyzw(
                view.pose.orientation.x,
                view.pose.orientation.y,
                view.pose.orientation.z,
                view.pose.orientation.w,
            );
            let l = view.fov.angle_left.tan();
            let r = view.fov.angle_right.tan();
            let t = view.fov.angle_up.tan();
            let b = view.fov.angle_down.tan();
            let (near, far) = (0.05, 100.0);
            #[rustfmt::skip]
            let projection = Mat4::from_cols(
                Vec4::new(2.0 / (r - l), 0.0           , (r + l) / (r - l)  , 0.0                     ),
                Vec4::new(0.0          , -2.0 / (t - b), -(t + b) / (t - b) , 0.0                     ),
                Vec4::new(0.0          , 0.0           , -far / (far - near), -far*near / (far - near)),
                Vec4::new(0.0          , 0.0           , -1.0               , 0.0                     ),
            ).transpose(); // We gave the matrix in row major, so transpose it
            let model = Mat4::from_rotation_translation(Quat::from_axis_angle(Vec3::new(0., 1., 0.), (delta.as_millis() as f32) / 1000.), Vec3::new(0.0, 0.0, -1.0));

            projection * Mat4::from_rotation_translation(rotation, translation).inverse() * model
        });
        {
            let mut mvp = self.xr_uniform_buffer.write()?;
            mvp.mvps = self.latest_mvps.map(|m| m.to_cols_array_2d());
        };

        self.xr_cmdbufs[image_index]
            .clone()
            .execute(self.queue.clone())?
            .then_signal_fence_and_flush()?
            .wait(None)?;
        self.queue.with(|_| swapchain.release_image())?;
        let views = [
            openxr::CompositionLayerProjectionView::new()
                .pose(views[0].pose)
                .fov(views[0].fov)
                .sub_image(
                    SwapchainSubImage::new()
                        .swapchain(swapchain)
                        .image_array_index(0)
                        .image_rect(Rect2Di {
                            offset: Offset2Di { x: 0, y: 0 },
                            extent: Extent2Di {
                                width: render_size.x as _,
                                height: render_size.y as _,
                            },
                        }),
                ),
            openxr::CompositionLayerProjectionView::new()
                .pose(views[1].pose)
                .fov(views[1].fov)
                .sub_image(
                    SwapchainSubImage::new()
                        .swapchain(swapchain)
                        .image_array_index(1)
                        .image_rect(Rect2Di {
                            offset: Offset2Di { x: 0, y: 0 },
                            extent: Extent2Di {
                                width: render_size.x as _,
                                height: render_size.y as _,
                            },
                        }),
                ),
        ];

        let depths = if let Some(sc) = depth_swapchain {
            let mut i = 0;
            self.queue.with(|_| sc.release_image())?;
            Some(views.each_ref().map(|_| {
                let sub_img = SwapchainSubImage::new()
                    .swapchain(sc)
                    .image_array_index(i)
                    .image_rect(Rect2Di {
                        offset: Offset2Di { x: 0, y: 0 },
                        extent: Extent2Di {
                            width: render_size.x as _,
                            height: render_size.y as _,
                        },
                    });
                i += 1;
                openxr::sys::CompositionLayerDepthInfoKHR {
                    ty: openxr::sys::CompositionLayerDepthInfoKHR::TYPE,
                    next: std::ptr::null(),
                    sub_image: sub_img.into_raw(),
                    max_depth: 1.0,
                    min_depth: 0.0,
                    near_z: 0.05,
                    far_z: 100.0,
                }
            }))
        } else {
            None
        };
        let views = if let Some(depths) = &depths {
            let mut i = 0;
            views.map(|v| {
                let mut v = v.into_raw();
                v.next = &depths[i] as *const _ as *const _;
                i += 1;
                unsafe { openxr::CompositionLayerProjectionView::from_raw(v) }
            })
        } else {
            views
        };
        let layer = openxr::CompositionLayerProjection::new()
            .space(space)
            .layer_flags(CompositionLayerFlags::BLEND_TEXTURE_SOURCE_ALPHA)
            .views(&views);
        let layers: &[&openxr::CompositionLayerBase<_>] = if let Some(p) = self.passthrough {
            let passthrough_layer = openxr::sys::CompositionLayerPassthroughHTC {
                ty: openxr::sys::CompositionLayerPassthroughHTC::TYPE,
                next: std::ptr::null(),
                // Spec: layer_flags must not be 0. don't know why, let's just use a
                // noop flag.
                layer_flags: openxr::sys::CompositionLayerFlags::CORRECT_CHROMATIC_ABERRATION,
                space: openxr::sys::Space::NULL,
                passthrough: p,
                color: openxr::sys::PassthroughColorHTC {
                    ty: openxr::sys::PassthroughColorHTC::TYPE,
                    next: std::ptr::null(),
                    alpha: 1.0,
                },
            };
            // Safety: `openxr::CompositionLayerBase` is a transparent wrapper of
            // `openxr::sys::CompositionLayerBaseHeader`, which is a prefix of
            // `openxr::sys::CompositionLayerPassthroughHTC`.
            let passthrough_layer = unsafe {
                &*(&passthrough_layer as *const openxr::sys::CompositionLayerPassthroughHTC).cast()
            };
            &[passthrough_layer, &layer]
        } else {
            &[&layer]
        };

        self.queue.with(|_| {
            self.frame_stream.end(
                state.predicted_display_time,
                EnvironmentBlendMode::OPAQUE,
                layers,
            )
        })?;
        log::debug!("XR rendering end");
        Ok(())
    }
}

enum AppMessage {
    Frame(openxr::FrameState),
    WaiterExited(openxr::FrameWaiter),
}

impl std::fmt::Debug for AppMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AppMessage::Frame(state) => write!(f, "Frame({state:?})"),
            AppMessage::WaiterExited(_) => write!(f, "WaiterExited"),
        }
    }
}

fn main() -> Result<()> {
    env_logger::Builder::from_default_env()
        .format_timestamp_millis()
        .init();
    let entry = unsafe { openxr::Entry::load() }?;
    let supported_xr_extensions = entry.enumerate_extensions()?;
    let mut xr_extensions = openxr::ExtensionSet::default();
    let layers = entry.enumerate_layers()?;
    let mut api_layers = vec![
        "XR_APILAYER_LUNARG_api_dump",
        //"XR_APILAYER_LUNARG_core_validation", // has bugs for passthrough
    ];
    if supported_xr_extensions.htc_passthrough {
        log::info!("Has HTC passthrough extension");
        xr_extensions.htc_passthrough = true;
    } else {
        for layer in &layers {
            let layer_exts = entry.enumerate_layer_extensions(&layer.layer_name)?;
            if layer_exts.htc_passthrough {
                xr_extensions.htc_passthrough = true;
                api_layers.push(layer.layer_name.as_str());
                break;
            }
        }
    }
    if supported_xr_extensions.khr_composition_layer_depth {
        log::info!("Has KHR composition layer depth extension");
        xr_extensions.khr_composition_layer_depth = true;
    }
    //xr_extensions.htc_passthrough = true;
    let (xr, frame_waiter, frame_stream) = xr::OpenXr::new(
        vulkano::instance::InstanceExtensions {
            khr_xlib_surface: true,
            ..Default::default()
        },
        &xr_extensions,
        &api_layers,
        "openxr hello world",
        1,
    )?;
    println!("Hello, world!");

    let event_loop = EventLoop::<AppMessage>::with_user_event().build()?;
    let mut renderer = App::new(xr, event_loop.create_proxy(), frame_waiter, frame_stream)?;
    renderer.passthrough = renderer
        .xr
        .xr_instance()
        .exts()
        .htc_passthrough
        .map(|passthrough_fp| {
            let mut passthrough = openxr::sys::PassthroughHTC::NULL;
            let passthrough_info = openxr::sys::PassthroughCreateInfoHTC {
                form: openxr::sys::PassthroughFormHTC::PLANAR,
                ty: openxr::sys::PassthroughCreateInfoHTC::TYPE,
                next: std::ptr::null(),
            };
            unsafe {
                (passthrough_fp.create_passthrough)(
                    renderer.xr.xr_session().as_raw(),
                    &passthrough_info,
                    &mut passthrough,
                )
            }
            .context("failed to create passthrough")?;
            Ok::<_, anyhow::Error>(passthrough)
        })
        .transpose()?;
    event_loop.run_app(&mut renderer)?;

    log::info!("Session ended");
    Ok(())
}

// Shader for a cube
mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
            #version 450
            #extension GL_EXT_multiview : enable
            layout(location = 0) in vec3 in_position;
            layout(location = 1) in vec4 in_color;
            layout(location = 0) out vec4 out_color;
            layout(binding = 0) uniform MVP {
                mat4 mvps[2];
            };
            void main () {
                gl_Position = mvps[gl_ViewIndex] * vec4(in_position, 1.0);
                out_color = in_color;
            }
        "
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
            #version 450
            layout(location = 0) in vec4 out_color;
            layout(location = 0) out vec4 frag_color;
            void main () {
                frag_color = out_color;
            }
        "
    }
}
