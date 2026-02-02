// heat_laplacian.wgsl — Discrete Laplacian for 2D heat equation (FTCS)
// Implements centered-space finite differences from:
// https://clockssugars.blog/articles/0126-heateq/

@group(0) @binding(0) var<storage, read> data: array<f32>;
@group(0) @binding(1) var<storage, read_write> laplacian: array<f32>;
@group(0) @binding(2) var<uniform> width: u32;
@group(0) @binding(3) var<uniform> height: u32;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = width * height;

    if (idx >= total) { return; }

    // Boundary: first/last row → laplacian = 0 (insulating)
    if (idx < width) { laplacian[idx] = 0.0; return; }
    if (idx >= width * (height - 1u)) { laplacian[idx] = 0.0; return; }

    // Boundary: first/last column → laplacian = 0
    let col = idx % width;
    if (col == 0u || col == width - 1u) { laplacian[idx] = 0.0; return; }

    let dx_sq = 1.0 / f32(width) / f32(width);
    let dy_sq = 1.0 / f32(height) / f32(height);

    // Centered differences: d2T/dx2 + d2T/dy2
    laplacian[idx] =
        (data[idx + 1u] - 2.0 * data[idx] + data[idx - 1u]) / dx_sq
      + (data[idx + width] - 2.0 * data[idx] + data[idx - width]) / dy_sq;
}
