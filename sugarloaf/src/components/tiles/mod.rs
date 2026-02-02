/// Tile rendering system for Rio Terminal — "duck" heat equation integration
///
/// Implements the Explicit FTCS Runge-Kutta 2 heat equation solver
/// from https://clockssugars.blog/articles/0126-heateq/ as a
/// WebGPU compute shader tile, renderable inside the terminal.
///
/// Pipeline order per frame:
/// 1. Fix boundary conditions (insulating / Neumann)
/// 2. Compute Laplacian at time t
/// 3. RK2 half-step: midpoint = data + (dt/2) * kappa * laplacian
/// 4. Compute Laplacian at midpoint
/// 5. RK2 full-step: output = data + dt * kappa * midpoint_laplacian
/// 6. Copy output → data for next frame
/// 7. Heatmap: temperature → RGBA texture

use crate::context::Context;
use bytemuck::{Pod, Zeroable};

pub const HEAT_GRID_WIDTH: u32 = 128;
pub const HEAT_GRID_HEIGHT: u32 = 128;
const KAPPA: f32 = 0.0001;
const DELTA_T: f32 = 0.4;
const ITERATIONS_PER_FRAME: u32 = 8;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct TileUniforms {
    pub position: [f32; 2],
    pub size: [f32; 2],
    pub time: f32,
    pub custom: [f32; 4],
    pub _pad: [f32; 3],
}

/// CPU-side state for the tile world
pub struct TileWorldState {
    pub time: f64,
    pub tiles: Vec<TileScene>,
}

#[derive(Debug, Clone)]
pub struct TileScene {
    pub id: u64,
    pub shader: TileShaderKind,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub custom: [f32; 4],
    pub persistent: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TileShaderKind {
    Plasma,
    Clock,
    Noise,
    HeatEquation,
    Custom(u32),
}

impl TileWorldState {
    pub fn new() -> Self {
        Self {
            time: 0.0,
            tiles: Vec::new(),
        }
    }

    pub fn begin_frame(&mut self, dt: f64) {
        self.time += dt;
        // Remove transient tiles
        self.tiles.retain(|t| t.persistent);
    }

    pub fn insert_tile(&mut self, tile: TileScene) {
        // Replace if same ID exists
        if tile.id > 0 {
            self.tiles.retain(|t| t.id != tile.id);
        }
        self.tiles.push(tile);
    }

    pub fn remove_tile(&mut self, id: u64) {
        self.tiles.retain(|t| t.id != id);
    }

    pub fn clear(&mut self) {
        self.tiles.clear();
    }
}

/// The heat equation compute brush
pub struct HeatBrush {
    // Compute pipelines
    boundary_pipeline: wgpu::ComputePipeline,
    laplacian_pipeline: wgpu::ComputePipeline,
    iterate_pipeline: wgpu::ComputePipeline,
    copy_pipeline: wgpu::ComputePipeline,
    heatmap_pipeline: wgpu::ComputePipeline,

    // Data buffers
    data_buffer: wgpu::Buffer,
    laplacian_buffer: wgpu::Buffer,
    midpoint_buffer: wgpu::Buffer,
    midpoint_laplacian_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    heatmap_buffer: wgpu::Buffer,

    // Uniform buffers
    width_buffer: wgpu::Buffer,
    height_buffer: wgpu::Buffer,
    kappa_buffer: wgpu::Buffer,
    delta_t_buffer: wgpu::Buffer,
    half_delta_t_buffer: wgpu::Buffer,
    min_t_buffer: wgpu::Buffer,
    max_t_buffer: wgpu::Buffer,
    pad_buffer: wgpu::Buffer,

    // Bind groups for each pipeline step
    boundary_bg: wgpu::BindGroup,
    laplacian_bg: wgpu::BindGroup,
    iterate_half_bg: wgpu::BindGroup,
    midpoint_laplacian_bg: wgpu::BindGroup,
    iterate_full_bg: wgpu::BindGroup,
    copy_bg: wgpu::BindGroup,
    heatmap_bg: wgpu::BindGroup,

    // Texture for rendering result
    pub texture: wgpu::Texture,
    pub texture_view: wgpu::TextureView,

    grid_width: u32,
    grid_height: u32,
}

impl HeatBrush {
    pub fn new(context: &Context) -> Self {
        let device = &context.device;
        let w = HEAT_GRID_WIDTH;
        let h = HEAT_GRID_HEIGHT;
        let total = (w * h) as usize;
        let pad_per_line = w.div_ceil(64) * 64 - w;

        // Initialize temperature field: Gaussian hot spot in center
        let mut initial_data = vec![0.0f32; total];
        let cx = w as f32 / 2.0;
        let cy = h as f32 / 2.0;
        for j in 0..h {
            for i in 0..w {
                let dx = (i as f32 - cx) / (w as f32 * 0.15);
                let dy = (j as f32 - cy) / (h as f32 * 0.15);
                initial_data[(j * w + i) as usize] = (-0.5 * (dx * dx + dy * dy)).exp();
            }
        }

        // Create storage buffers
        let data_buffer = Self::create_storage_buffer(device, "heat_data", &initial_data);
        let laplacian_buffer =
            Self::create_storage_buffer(device, "heat_laplacian", &vec![0.0f32; total]);
        let midpoint_buffer =
            Self::create_storage_buffer(device, "heat_midpoint", &vec![0.0f32; total]);
        let midpoint_laplacian_buffer =
            Self::create_storage_buffer(device, "heat_midpoint_lap", &vec![0.0f32; total]);
        let output_buffer =
            Self::create_storage_buffer(device, "heat_output", &vec![0.0f32; total]);

        let heatmap_size = ((w + pad_per_line) * h) as usize;
        let heatmap_buffer =
            Self::create_storage_buffer(device, "heat_heatmap", &vec![0u32; heatmap_size]);

        // Uniform buffers
        let width_buffer = Self::create_uniform_buffer(device, "width", &[w]);
        let height_buffer = Self::create_uniform_buffer(device, "height", &[h]);
        let kappa_buffer = Self::create_uniform_buffer(device, "kappa", &[KAPPA]);
        let delta_t_buffer = Self::create_uniform_buffer(device, "delta_t", &[DELTA_T]);
        let half_delta_t_buffer =
            Self::create_uniform_buffer(device, "half_delta_t", &[DELTA_T * 0.5]);
        let min_t_buffer = Self::create_uniform_buffer(device, "min_t", &[0.0f32]);
        let max_t_buffer = Self::create_uniform_buffer(device, "max_t", &[1.0f32]);
        let pad_buffer = Self::create_uniform_buffer(device, "pad", &[pad_per_line]);

        // --- Create compute pipelines ---

        let boundary_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("heat_boundary"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("heat_boundary.wgsl").into(),
            ),
        });
        let laplacian_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("heat_laplacian"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("heat_laplacian.wgsl").into(),
            ),
        });
        let iterate_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("heat_iterate"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("heat_iterate.wgsl").into(),
            ),
        });
        let copy_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("heat_copy"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("heat_copy.wgsl").into(),
            ),
        });
        let heatmap_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("heat_heatmap"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("heat_heatmap.wgsl").into(),
            ),
        });

        let boundary_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("heat_boundary_pipeline"),
                layout: None,
                module: &boundary_module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let laplacian_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("heat_laplacian_pipeline"),
                layout: None,
                module: &laplacian_module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let iterate_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("heat_iterate_pipeline"),
                layout: None,
                module: &iterate_module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let copy_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("heat_copy_pipeline"),
                layout: None,
                module: &copy_module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let heatmap_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("heat_heatmap_pipeline"),
                layout: None,
                module: &heatmap_module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // --- Create bind groups ---

        let boundary_bg = Self::make_bind_group(
            device,
            &boundary_pipeline,
            &[&data_buffer, &width_buffer, &height_buffer],
        );

        let laplacian_bg = Self::make_bind_group(
            device,
            &laplacian_pipeline,
            &[&data_buffer, &laplacian_buffer, &width_buffer, &height_buffer],
        );

        // RK2 half-step: data + (dt/2) * kappa * laplacian → midpoint
        let iterate_half_bg = Self::make_bind_group(
            device,
            &iterate_pipeline,
            &[
                &data_buffer,
                &laplacian_buffer,
                &midpoint_buffer,
                &width_buffer,
                &height_buffer,
                &half_delta_t_buffer,
                &kappa_buffer,
            ],
        );

        // Laplacian at midpoint
        let midpoint_laplacian_bg = Self::make_bind_group(
            device,
            &laplacian_pipeline,
            &[
                &midpoint_buffer,
                &midpoint_laplacian_buffer,
                &width_buffer,
                &height_buffer,
            ],
        );

        // RK2 full-step: data + dt * kappa * midpoint_laplacian → output
        let iterate_full_bg = Self::make_bind_group(
            device,
            &iterate_pipeline,
            &[
                &data_buffer,
                &midpoint_laplacian_buffer,
                &output_buffer,
                &width_buffer,
                &height_buffer,
                &delta_t_buffer,
                &kappa_buffer,
            ],
        );

        // Copy output → data
        let copy_bg = Self::make_bind_group(
            device,
            &copy_pipeline,
            &[&output_buffer, &data_buffer, &width_buffer, &height_buffer],
        );

        // Heatmap visualization
        let heatmap_bg = Self::make_bind_group(
            device,
            &heatmap_pipeline,
            &[
                &data_buffer,
                &heatmap_buffer,
                &width_buffer,
                &height_buffer,
                &min_t_buffer,
                &max_t_buffer,
                &pad_buffer,
            ],
        );

        // Create texture for rendering the heatmap
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("heat_texture"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            boundary_pipeline,
            laplacian_pipeline,
            iterate_pipeline,
            copy_pipeline,
            heatmap_pipeline,
            data_buffer,
            laplacian_buffer,
            midpoint_buffer,
            midpoint_laplacian_buffer,
            output_buffer,
            heatmap_buffer,
            width_buffer,
            height_buffer,
            kappa_buffer,
            delta_t_buffer,
            half_delta_t_buffer,
            min_t_buffer,
            max_t_buffer,
            pad_buffer,
            boundary_bg,
            laplacian_bg,
            iterate_half_bg,
            midpoint_laplacian_bg,
            iterate_full_bg,
            copy_bg,
            heatmap_bg,
            texture,
            texture_view,
            grid_width: w,
            grid_height: h,
        }
    }

    /// Run the RK2 heat equation solver and produce a heatmap texture
    pub fn compute(&self, encoder: &mut wgpu::CommandEncoder) {
        let wg_count_1d = (self.grid_width * self.grid_height).div_ceil(64);
        let wg_x_2d = self.grid_width.div_ceil(8);
        let wg_y_2d = self.grid_height.div_ceil(8);

        for _ in 0..ITERATIONS_PER_FRAME {
            // 1. Fix boundary conditions
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.boundary_pipeline);
                pass.set_bind_group(0, &self.boundary_bg, &[]);
                pass.dispatch_workgroups(wg_count_1d, 1, 1);
            }

            // 2. Compute Laplacian at time t
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.laplacian_pipeline);
                pass.set_bind_group(0, &self.laplacian_bg, &[]);
                pass.dispatch_workgroups(wg_count_1d, 1, 1);
            }

            // 3. RK2 half-step → midpoint
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.iterate_pipeline);
                pass.set_bind_group(0, &self.iterate_half_bg, &[]);
                pass.dispatch_workgroups(wg_count_1d, 1, 1);
            }

            // 4. Laplacian at midpoint
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.laplacian_pipeline);
                pass.set_bind_group(0, &self.midpoint_laplacian_bg, &[]);
                pass.dispatch_workgroups(wg_count_1d, 1, 1);
            }

            // 5. RK2 full-step → output
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.iterate_pipeline);
                pass.set_bind_group(0, &self.iterate_full_bg, &[]);
                pass.dispatch_workgroups(wg_count_1d, 1, 1);
            }

            // 6. Copy output → data
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.copy_pipeline);
                pass.set_bind_group(0, &self.copy_bg, &[]);
                pass.dispatch_workgroups(wg_count_1d, 1, 1);
            }
        }

        // 7. Generate heatmap RGBA
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.heatmap_pipeline);
            pass.set_bind_group(0, &self.heatmap_bg, &[]);
            pass.dispatch_workgroups(wg_x_2d, wg_y_2d, 1);
        }
    }

    // --- Helper functions ---

    fn create_storage_buffer<T: Pod>(device: &wgpu::Device, label: &str, data: &[T]) -> wgpu::Buffer {
        use wgpu::util::{BufferInitDescriptor, DeviceExt};
        device.create_buffer_init(&BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        })
    }

    fn create_uniform_buffer<T: Pod>(device: &wgpu::Device, label: &str, data: &[T]) -> wgpu::Buffer {
        use wgpu::util::{BufferInitDescriptor, DeviceExt};
        device.create_buffer_init(&BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }

    fn make_bind_group(
        device: &wgpu::Device,
        pipeline: &wgpu::ComputePipeline,
        buffers: &[&wgpu::Buffer],
    ) -> wgpu::BindGroup {
        let entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(n, buf)| wgpu::BindGroupEntry {
                binding: n as u32,
                resource: buf.as_entire_binding(),
            })
            .collect();

        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        })
    }
}
