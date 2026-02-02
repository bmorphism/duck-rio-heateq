// heat_iterate.wgsl — Forward Euler step: T_out = T_in + dt * kappa * laplacian
// Used for both the RK2 half-step and full-step

@group(0) @binding(0) var<storage, read> data: array<f32>;
@group(0) @binding(1) var<storage, read> laplacian: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> width: u32;
@group(0) @binding(4) var<uniform> height: u32;
@group(0) @binding(5) var<uniform> delta_t: f32;
@group(0) @binding(6) var<uniform> kappa: f32;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = width * height;

    if (idx >= total) { return; }

    output[idx] = data[idx] + delta_t * kappa * laplacian[idx];
}
