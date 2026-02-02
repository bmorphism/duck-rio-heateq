// heat_boundary.wgsl — Insulating (Neumann) boundary conditions
// Sets edge values equal to nearest interior neighbors

@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<uniform> width: u32;
@group(0) @binding(2) var<uniform> height: u32;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = width * height;

    if (idx >= total) { return; }

    let col = idx % width;
    let row = idx / width;

    // Top row: copy from row 1
    if (row == 0u && col > 0u && col < width - 1u) {
        data[idx] = data[idx + width];
        return;
    }

    // Bottom row: copy from row height-2
    if (row == height - 1u && col > 0u && col < width - 1u) {
        data[idx] = data[idx - width];
        return;
    }

    // Left column: copy from column 1
    if (col == 0u && row > 0u && row < height - 1u) {
        data[idx] = data[idx + 1u];
        return;
    }

    // Right column: copy from column width-2
    if (col == width - 1u && row > 0u && row < height - 1u) {
        data[idx] = data[idx - 1u];
        return;
    }

    // Corners: average of two adjacent interior cells
    if (row == 0u && col == 0u) {
        data[idx] = 0.5 * (data[1u] + data[width]);
    } else if (row == 0u && col == width - 1u) {
        data[idx] = 0.5 * (data[width - 2u] + data[2u * width - 1u]);
    } else if (row == height - 1u && col == 0u) {
        data[idx] = 0.5 * (data[(height - 2u) * width] + data[(height - 1u) * width + 1u]);
    } else if (row == height - 1u && col == width - 1u) {
        data[idx] = 0.5 * (data[(height - 2u) * width + width - 1u] + data[(height - 1u) * width + width - 2u]);
    }
}
