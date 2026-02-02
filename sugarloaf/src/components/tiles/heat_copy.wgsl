// heat_copy.wgsl — Copy buffer A → B

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> width: u32;
@group(0) @binding(3) var<uniform> height: u32;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= width * height) { return; }
    dst[idx] = src[idx];
}
