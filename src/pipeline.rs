use glam::{
    UVec2,
    f64::{DVec2 as Vec2, DVec4 as Vec4},
};
use smallvec::smallvec;
use std::sync::Arc;

use crate::{steam::StereoCamera, utils::DeviceExt as _};
use anyhow::Result;
use log::{info, trace};
use vulkano::{
    Handle, VulkanObject,
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, BlitImageInfo, CommandBufferUsage, CopyBufferToImageInfo,
        ImageBlit, PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract as _,
        RenderPassBeginInfo, SubpassBeginInfo, SubpassContents, SubpassEndInfo,
        allocator::CommandBufferAllocator,
    },
    descriptor_set::{
        DescriptorBufferInfo, DescriptorImageInfo, DescriptorSet, WriteDescriptorSet,
        allocator::DescriptorSetAllocator,
    },
    device::{Device, Queue},
    format::{Format, FormatFeatures},
    image::{
        Image as VkImage, ImageCreateInfo, ImageLayout, ImageTiling, ImageUsage,
        sampler::{
            Filter, Sampler, SamplerCreateInfo,
            ycbcr::{
                SamplerYcbcrConversion, SamplerYcbcrConversionCreateInfo,
                SamplerYcbcrModelConversion,
            },
        },
        view::{ImageView, ImageViewCreateInfo},
    },
    memory::allocator::{
        AllocationCreateInfo, MemoryAllocatePreference, MemoryAllocator, MemoryTypeFilter,
    },
    padded::Padded,
    pipeline::{
        GraphicsPipeline, Pipeline as _, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
        cache::PipelineCache,
        graphics::{
            GraphicsPipelineCreateInfo,
            color_blend::ColorBlendState,
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            subpass::PipelineSubpassType,
            vertex_input::{self, Vertex as _, VertexDefinition},
            viewport::{Viewport, ViewportState},
        },
    },
    render_pass::{Framebuffer, Subpass},
    shader::ShaderModule,
    sync::{GpuFuture, future::FenceSignalFuture},
};

/// Lens distortion correction parameters for a side-by-side stereo image
#[derive(Debug)]
struct StereoUndistortParams {
    /// field-of-view parameter, 0 = left eye, 1 = right eye
    fov: [Vec2; 2],
    scale: [Vec2; 2],
    focal: [Vec2; 2],
    center: [Vec2; 2],
    coeff: [Vec4; 2],
}

impl StereoUndistortParams {
    pub fn fov(&self) -> [Vec2; 2] {
        self.fov
    }
    /// i.e. solving Undistort(src) = dst for the smallest non-zero root.
    fn undistort_inverse(coeff: &Vec4, dst: f64) -> Option<f64> {
        // solving: x * (1 + k1*x^2 + k2*x^4 + k3*x^6 + k4*x^8) - dst = 0
        let f = |x: f64| {
            let x2 = x * x;
            x * (1.0 + x2 * (coeff[0] + x2 * (coeff[1] + x2 * (coeff[2] + x2 * coeff[3])))) - dst
        };
        let fp = |x: f64| {
            let x2 = x * x;
            1.0 + x2
                * (3.0 * coeff[0]
                    + x2 * (5.0 * coeff[1] + x2 * (7.0 * coeff[2] + x2 * 9.0 * coeff[3])))
        };
        const MAX_ITER: u32 = 100;
        let mut x = 0.0;
        for _ in 0..MAX_ITER {
            if fp(x) == 0.0 {
                // Give up
                info!("Divided by zero");
                return None;
            }
            trace!("{} {} {}", x, f(x), fp(x));
            if f(x).abs() < 1e-6 {
                info!("Inverse is: {}, {} {}", x, f(x), dst);
                return Some(x);
            }
            x = x - f(x) / fp(x);
        }
        // Give up
        info!("Cannot find scale");
        None
    }
    // Find a scale that maps the middle point of 4 edges of the undistorted image to
    // the edge of the field of view of the distorted image.
    //
    // Returns the scales and the adjusted fovs
    fn find_scale(coeff: &Vec4, center: &Vec2, focal: &Vec2) -> (Vec2, Vec2) {
        let ret = [0, 1].map(|i| {
            let min_edge_dist = center[i].min(1.0 - center[i]) / focal[i];
            // Find the input theta angle where Undistort(theta) = min_edge_dist
            if let Some(theta) = Self::undistort_inverse(coeff, min_edge_dist) {
                if theta >= std::f64::consts::PI / 2.0 {
                    // infinity?
                    (1.0, focal[i])
                } else {
                    // Find the input coordinates that will give us that theta
                    let target_edge = theta.tan();
                    log::info!("{}", target_edge);
                    (target_edge / (0.5 / focal[i]), 1.0 / min_edge_dist / 2.0)
                }
            } else {
                // Cannot find scale so just don't scale
                (1.0, focal[i])
            }
        });
        (Vec2::new(ret[0].0, ret[1].0), Vec2::new(ret[0].1, ret[1].1))
    }
    /// Input size is (size * 2, size)
    /// returns also the adjusted FOV for left and right
    ///
    /// # Arguments
    ///
    /// - is_final: whether this is the final stage of the pipeline.
    ///   if true, the output image will be submitted to
    ///   the vr compositor.
    pub fn new(size: UVec2, camera_calib: &StereoCamera) -> Result<Self> {
        let size = size.as_dvec2();
        let center = [
            Vec2::new(
                camera_calib.left.intrinsics.center_x / size.x,
                camera_calib.left.intrinsics.center_y / size.y,
            ),
            Vec2::new(
                camera_calib.right.intrinsics.center_x / size.x,
                camera_calib.right.intrinsics.center_y / size.y,
            ),
        ];
        let focal = [
            Vec2::new(
                camera_calib.left.intrinsics.focal_x / size.x,
                camera_calib.left.intrinsics.focal_y / size.y,
            ),
            Vec2::new(
                camera_calib.right.intrinsics.focal_x / size.x,
                camera_calib.right.intrinsics.focal_y / size.y,
            ),
        ];
        let coeff: [Vec4; 2] = [
            camera_calib.left.intrinsics.distort.coeffs.into(),
            camera_calib.right.intrinsics.distort.coeffs.into(),
        ];
        let scale_fov = [0, 1].map(|i| Self::find_scale(&coeff[i], &center[i], &focal[i]));
        Ok(Self {
            fov: [scale_fov[0].1, scale_fov[1].1],
            scale: [scale_fov[0].0, scale_fov[1].0],
            focal,
            center,
            coeff,
        })
    }
}

#[derive(vertex_input::Vertex, Default, Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[allow(non_snake_case)]
#[repr(C)]
struct Vertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}

pub struct Pipeline {
    correction: Option<StereoUndistortParams>,
    capture: bool,
    /// A cpu buffer for storing and uploading the input image.
    input_image_buffer: Arc<Buffer>,
    input_image_gpu: Arc<VkImage>,
    /// The input image after post-processing (e.g. undistortion, yuv to rgb conversion)
    postprocessed_image: Arc<VkImage>,
    camera_config: Option<StereoCamera>,
    allocator: Arc<dyn MemoryAllocator>,
    cmdbuf_allocator: Arc<dyn CommandBufferAllocator>,
    device: Arc<Device>,
    queue: Arc<Queue>,

    /// Command buffer for uploading the image to the GPU
    cmdbuf: Arc<PrimaryAutoCommandBuffer>,
    previous_upload_end: Option<Arc<FenceSignalFuture<Box<dyn GpuFuture + Send + Sync>>>>,
    previous_frame_time: Option<std::time::Instant>,
}

impl std::fmt::Debug for Pipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pipeline")
            .field("correction", &self.correction)
            .field("capture", &self.capture)
            .field("input_texture", &self.input_image_gpu.handle().as_raw())
            .field("camera_config", &self.camera_config)
            .finish_non_exhaustive()
    }
}

pub trait PostprocessPipeline {
    fn allocate_image(&self) -> Result<Arc<VkImage>>;
    /// Create command buffers for a given set of swapchain images. If a set of command buffers
    /// were created previously, they will be replaced.
    fn create_command_buffers(&mut self, images: &[Arc<VkImage>], input_len: usize) -> Result<()>;
    /// Postprocess the camera image, taking input from a in memory buffer.
    fn postprocess(&self, input: &[u8], output: Arc<VkImage>) -> Result<Box<dyn GpuFuture>>;
    // /// Postprocess the camera image, taking input from a dmabuf file descriptor.
    // fn postprocess_dmabuf(&self, input: OwnedFd, output: &Self::Image) -> Result<Self::Future>;
}

impl Pipeline {
    pub fn load_shader(
        device: &Arc<Device>,
        source_is_yuyv: bool,
        has_camera_config: bool,
        has_yuyv_sampler: bool,
    ) -> anyhow::Result<(Arc<ShaderModule>, Arc<ShaderModule>)> {
        let vs = vs::load(device)?;
        let fs = match (source_is_yuyv && !has_yuyv_sampler, has_camera_config) {
            (true, true) => fs::yuyv_undistort::load(device)?,
            (true, false) => fs::yuyv::load(device)?,
            (false, true) => fs::undistort::load(device)?,
            (false, false) => fs::unprocessed::load(device)?,
        };
        Ok((vs, fs))
    }

    /// Create post-processing stages
    /// The camera image is two `size` images stitched together side-by-side.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &Arc<Device>,
        allocator: &Arc<dyn MemoryAllocator>,
        cmdbuf_allocator: &Arc<dyn CommandBufferAllocator>,
        queue: &Arc<Queue>,
        descriptor_set_allocator: &Arc<impl DescriptorSetAllocator>,
        source_is_yuyv: bool,
        camera_config: Option<&StereoCamera>,
        final_layout: ImageLayout,
        output_usage: ImageUsage,
        pipeline_cache: Arc<PipelineCache>,
        camera_size: UVec2,
        render_size: UVec2,
    ) -> Result<Self> {
        let exts = device.enabled_extensions();
        let feats = device.enabled_features();
        let has_yuyv_sampler =
            if exts.khr_sampler_ycbcr_conversion && feats.sampler_ycbcr_conversion {
                let format_feats = device
                    .physical_device()
                    .format_properties(Format::G8B8G8R8_422_UNORM)?
                    .format_features(ImageTiling::Optimal, &[]);
                format_feats.contains(FormatFeatures::MIDPOINT_CHROMA_SAMPLES)
            } else {
                false
            };
        let format = if has_yuyv_sampler && source_is_yuyv {
            Format::G8B8G8R8_422_UNORM
        } else {
            Format::R8G8B8A8_UNORM
        };
        let (vs, fs) = Self::load_shader(
            device,
            source_is_yuyv,
            camera_config.is_some(),
            has_yuyv_sampler,
        )?;
        let vs_main = vs.entry_point("main").unwrap();
        let fs_main = fs.entry_point("main").unwrap();

        // Allocate intermediate textures
        let input_texture = device.new_image(
            &ImageCreateInfo {
                extent: if source_is_yuyv && !has_yuyv_sampler {
                    // Source is raw, unconverted yuyv, therefore is downsampled 2x in the X
                    // direction.
                    [camera_size.x, camera_size.y, 1]
                } else {
                    [camera_size.x * 2, camera_size.y, 1]
                },
                format,
                usage: ImageUsage::TRANSFER_DST
                    | ImageUsage::TRANSFER_SRC
                    | ImageUsage::SAMPLED
                    | ImageUsage::COLOR_ATTACHMENT,
                ..Default::default()
            },
            MemoryTypeFilter::PREFER_DEVICE,
        )?;
        let postprocessed_image = device.new_image(
            &ImageCreateInfo {
                extent: [render_size.x * 2, render_size.y, 1],
                format: Format::R8G8B8A8_UNORM,
                usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_DST | output_usage,
                ..Default::default()
            },
            MemoryTypeFilter::PREFER_DEVICE,
        )?;
        let cpu_buffer = device.new_buffer(
            &BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                size: camera_size.x as u64
                    * camera_size.y as u64
                    * 2
                    * if source_is_yuyv { 2 } else { 4 },
                ..Default::default()
            },
            MemoryTypeFilter::HOST_SEQUENTIAL_WRITE | MemoryTypeFilter::PREFER_DEVICE,
        )?;
        let render_pass = vulkano::single_pass_renderpass!(&device,
        attachments: {
            color: {
                format: vulkano::format::Format::R8G8B8A8_UNORM,
                samples: 1,
                load_op: Load,
                store_op: Store,
                final_layout: final_layout,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {},
        })
        .unwrap();
        let correction = camera_config
            .map(|c| StereoUndistortParams::new(camera_size, c))
            .transpose()?;
        log::debug!("correction fov: {:?}", correction.as_ref().map(|x| x.fov()));
        let fov = correction
            .as_ref()
            .map(|c| c.fov())
            .unwrap_or([Vec2::new(1.19, 1.19); 2]); // default to roughly 100 degrees fov, hopefully this is sensible
        let stages = [
            PipelineShaderStageCreateInfo::new(&vs_main),
            PipelineShaderStageCreateInfo::new(&fs_main),
        ];
        let layout = PipelineLayout::from_stages(device, &stages)?;
        let sampler = Sampler::new(
            device,
            &SamplerCreateInfo {
                min_filter: Filter::Linear,
                mag_filter: Filter::Linear,
                sampler_ycbcr_conversion: has_yuyv_sampler
                    .then(|| {
                        SamplerYcbcrConversion::new(
                            device,
                            &SamplerYcbcrConversionCreateInfo {
                                format,
                                ycbcr_model: SamplerYcbcrModelConversion::Ycbcr709,
                                ..Default::default()
                            },
                        )
                    })
                    .transpose()?
                    .as_ref(),
                ..Default::default()
            },
        )?;
        let distortion_params = correction
            .as_ref()
            .map(|c| {
                Buffer::from_data(
                    allocator,
                    &BufferCreateInfo {
                        usage: BufferUsage::UNIFORM_BUFFER,
                        ..Default::default()
                    },
                    &AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE
                            | MemoryTypeFilter::PREFER_DEVICE,
                        allocate_preference: MemoryAllocatePreference::Unknown,
                        ..Default::default()
                    },
                    fs::yuyv_undistort::DistortionParameters {
                        center: c.center.map(|v| Padded(*v.as_vec2().as_ref())),
                        dcoef: c.coeff.map(|v| *v.as_vec4().as_ref()),
                        focal: c.focal.map(|v| Padded(*v.as_vec2().as_ref())),
                        scale: c.scale.map(|v| Padded(*v.as_vec2().as_ref())),
                    },
                )
                .map_err(anyhow::Error::from)
            })
            .transpose()?;
        let pipeline = GraphicsPipeline::new(
            device,
            Some(&pipeline_cache),
            &GraphicsPipelineCreateInfo {
                vertex_input_state: Some(&Vertex::per_vertex().definition(&vs_main)?),
                stages: &stages,
                input_assembly_state: Some(&InputAssemblyState {
                    topology: PrimitiveTopology::TriangleFan,
                    ..Default::default()
                }),
                viewport_state: Some(&ViewportState {
                    viewports: &[Viewport {
                        offset: [0., 0.],
                        extent: [(render_size.x * 2) as _, render_size.y as _],
                        ..Default::default()
                    }],
                    ..Default::default()
                }),
                subpass: Some(PipelineSubpassType::BeginRenderPass(
                    &Subpass::new(&render_pass, 0).unwrap(),
                )),
                multisample_state: Some(&Default::default()),
                color_blend_state: Some(&ColorBlendState {
                    attachments: &[Default::default()],
                    ..Default::default()
                }),
                rasterization_state: Some(&Default::default()),
                ..GraphicsPipelineCreateInfo::new(&layout)
            },
        )?;
        let desc_set = {
            let input_texture_view = ImageView::new(
                &input_texture,
                &ImageViewCreateInfo::from_image(&input_texture),
            )?;
            let input_texture_descriptor_info = DescriptorImageInfo {
                sampler: Some(&sampler),
                image_view: Some(&input_texture_view),
                image_layout: ImageLayout::ShaderReadOnlyOptimal,
            };
            let mut desc_set_writes =
                vec![WriteDescriptorSet::image(1, &input_texture_descriptor_info)];
            let distortion_params_buffer_info =
                distortion_params.as_ref().map(|b| DescriptorBufferInfo {
                    buffer: Some(b.buffer()),
                    ..Default::default()
                });
            let distortion_params_buffer_descriptor = distortion_params_buffer_info
                .as_ref()
                .map(|i| WriteDescriptorSet::buffer(2, i));
            desc_set_writes.extend(distortion_params_buffer_descriptor);
            DescriptorSet::new(
                descriptor_set_allocator,
                pipeline.layout().set_layouts().first().unwrap(),
                &desc_set_writes,
                &[],
            )?
        };
        let vertices = Buffer::from_iter::<Vertex, _>(
            allocator,
            &BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE
                    | MemoryTypeFilter::PREFER_DEVICE,
                allocate_preference: MemoryAllocatePreference::Unknown,
                ..Default::default()
            },
            [
                // Left eye quad
                Vertex {
                    position: [-1.0, -1.0],
                },
                Vertex {
                    position: [-1.0, 1.0],
                },
                Vertex {
                    position: [0.0, 1.0],
                },
                Vertex {
                    position: [0.0, -1.0],
                },
            ]
            .iter()
            .cloned(),
        )
        .unwrap();
        let buffer = Subbuffer::new(cpu_buffer.clone());
        let ivci = ImageViewCreateInfo::from_image(&postprocessed_image);
        let framebuffer = Framebuffer::new(
            &render_pass,
            &vulkano::render_pass::FramebufferCreateInfo {
                attachments: &[&ImageView::new(&postprocessed_image, &ivci)?],
                ..Default::default()
            },
        )?;
        let mut cmdbuf = AutoCommandBufferBuilder::primary(
            cmdbuf_allocator.clone(),
            queue.queue_family_index(),
            CommandBufferUsage::MultipleSubmit,
        )?;
        cmdbuf
            .copy_buffer_to_image(CopyBufferToImageInfo::new(buffer, input_texture.clone()))?
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![None],
                    ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                },
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )?
            .bind_pipeline_graphics(pipeline.clone())?
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                pipeline.layout().clone(),
                0,
                desc_set.clone(),
            )?
            .bind_vertex_buffers(0, vertices.clone())?;
        unsafe { cmdbuf.draw(vertices.len() as u32, 2, 0, 0)? }
            .end_render_pass(SubpassEndInfo::default())?;
        let cmdbuf = cmdbuf.build()?;

        log::info!("Adjusted FOV: {:?}", fov);
        Ok(Self {
            correction,
            capture: false,
            camera_config: camera_config.copied(),
            input_image_buffer: cpu_buffer,
            input_image_gpu: input_texture,
            postprocessed_image,
            previous_upload_end: None,
            previous_frame_time: None,
            device: device.clone(),
            allocator: allocator.clone(),
            cmdbuf,
            cmdbuf_allocator: cmdbuf_allocator.clone(),
            queue: queue.clone(),
        })
    }
    pub fn fov(&self) -> [Vec2; 2] {
        self.correction
            .as_ref()
            .map(|c| c.fov())
            .unwrap_or([Vec2::new(1.19, 1.19); 2])
    }
    /// Run the pipeline
    ///
    /// # Arguments
    ///
    /// - time: Time offset into the past when the camera frame is captured
    pub fn maybe_postprocess(&mut self, frame: &crate::FrameInfo) -> Result<()> {
        if Some(frame.frame_time) == self.previous_frame_time {
            return Ok(());
        }
        let mut previous_fut = self.previous_upload_end.take();
        if let Some(f) = &mut previous_fut {
            f.wait(None)?;
            f.cleanup_finished();
        }

        self.previous_upload_end = Some(Arc::new(if frame.needs_postprocess {
            {
                let buffer = Subbuffer::new(self.input_image_buffer.clone());
                buffer.write()?.copy_from_slice(&frame.frame);
            }

            if let Some(f) = previous_fut {
                f.then_execute(self.queue.clone(), self.cmdbuf.clone())?
                    .boxed_send_sync()
                    .then_signal_fence_and_flush()?
            } else {
                self.cmdbuf
                    .clone()
                    .execute(self.queue.clone())?
                    .boxed_send_sync()
                    .then_signal_fence_and_flush()?
            }
        } else {
            assert_eq!(
                frame.frame.len(),
                frame.size.x as usize * 2 * frame.size.y as usize * 4
            );
            let buffer = Buffer::new_slice::<u8>(
                &self.allocator,
                &BufferCreateInfo {
                    usage: BufferUsage::TRANSFER_SRC,
                    ..Default::default()
                },
                &AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE
                        | MemoryTypeFilter::PREFER_DEVICE,
                    allocate_preference: MemoryAllocatePreference::Unknown,
                    ..Default::default()
                },
                frame.size.x as u64 * 2 * frame.size.y as u64 * 4,
            )?;
            buffer.write()?.copy_from_slice(&frame.frame);
            let vkimg = self.device.new_image(
                &ImageCreateInfo {
                    format: vulkano::format::Format::R8G8B8A8_UNORM,
                    extent: [frame.size.x * 2, frame.size.y, 1],
                    usage: ImageUsage::TRANSFER_DST
                        | ImageUsage::TRANSFER_SRC
                        | ImageUsage::SAMPLED,
                    ..Default::default()
                },
                MemoryTypeFilter::PREFER_DEVICE,
            )?;
            let mut cmdbuf = AutoCommandBufferBuilder::primary(
                self.cmdbuf_allocator.clone(),
                self.queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )?;
            cmdbuf
                .copy_buffer_to_image(CopyBufferToImageInfo::new(buffer, vkimg.clone()))?
                .blit_image(BlitImageInfo {
                    src_image_layout: ImageLayout::TransferSrcOptimal,
                    dst_image_layout: ImageLayout::TransferDstOptimal,
                    filter: Filter::Linear,
                    regions: smallvec![ImageBlit {
                        src_subresource: vkimg.subresource_layers(),
                        dst_subresource: self.postprocessed_image.subresource_layers(),
                        src_offsets: [[0, 0, 0], vkimg.extent()],
                        dst_offsets: [[0, 0, 0], self.postprocessed_image.extent()],
                        ..Default::default()
                    }],
                    ..BlitImageInfo::new(vkimg, self.postprocessed_image.clone())
                })?;
            cmdbuf
                .build()?
                .execute(self.queue.clone())?
                .boxed_send_sync()
                .then_signal_fence_and_flush()?
        }));
        Ok(())
    }
    /// Return the post-processed image and the fence signal future to wait on for the image to be
    /// ready.
    ///
    /// # Panic
    ///
    /// panics if `maybe_postprocess` was not called before this function.
    pub fn image(
        &self,
    ) -> (
        Arc<VkImage>,
        Arc<FenceSignalFuture<Box<dyn GpuFuture + Send + Sync>>>,
    ) {
        (
            self.postprocessed_image.clone(),
            self.previous_upload_end.clone().unwrap(),
        )
    }

    pub fn image_extent(&self) -> [u32; 3] {
        self.postprocessed_image.extent()
    }
}

mod fs {
    pub mod yuyv_undistort {
        vulkano_shaders::shader! {
            ty: "fragment",
            path: "shaders/combined.frag",
            define: [
                ("INPUT_IS_YUYV", "1"),
                ("UNDISTORT", "1"),
            ],
            custom_derives: [Copy, Clone, Debug],
        }
    }
    pub mod undistort {
        vulkano_shaders::shader! {
            ty: "fragment",
            path: "shaders/combined.frag",
            define: [
                ("UNDISTORT", "1"),
            ],
            custom_derives: [Copy, Clone, Debug],
        }
    }
    pub mod yuyv {
        vulkano_shaders::shader! {
            ty: "fragment",
            path: "shaders/combined.frag",
            define: [
                ("INPUT_IS_YUYV", "1"),
            ],
            custom_derives: [Copy, Clone, Debug],
        }
    }
    pub mod unprocessed {
        vulkano_shaders::shader! {
            ty: "fragment",
            path: "shaders/combined.frag",
            custom_derives: [Copy, Clone, Debug],
        }
    }
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "#version 450
layout(location = 0) in vec2 position;
layout(location = 0) out flat uint instanceId;
layout(location = 1) out vec2 eyeRelativeCoord;

void main() {
    gl_Position = vec4(position, 0, 1) + vec4(1.0, 0.0, 0.0, 0.0) * float(gl_InstanceIndex);
    instanceId = gl_InstanceIndex;
    eyeRelativeCoord = position * vec2(2.0, 1.0) + vec2(1.0, 0);
}"
    }
}
