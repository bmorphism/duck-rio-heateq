// heat_heatmap.wgsl — Convert temperature field to RGBA texture
// Color mapping: blue (cold) → cyan → green → yellow → red (hot)
// Based on the HSL color wheel approach from clockssugars.blog

@group(0) @binding(0) var<storage, read> data: array<f32>;
@group(0) @binding(1) var<storage, read_write> rgba_out: array<u32>;
@group(0) @binding(2) var<uniform> width: u32;
@group(0) @binding(3) var<uniform> height: u32;
@group(0) @binding(4) var<uniform> min_t: f32;
@group(0) @binding(5) var<uniform> max_t: f32;
@group(0) @binding(6) var<uniform> pad_per_line: u32;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= width || gid.y >= height) { return; }

    let data_idx = gid.x + gid.y * width;
    let range_val = clamp((data[data_idx] - min_t) / (max_t - min_t), 0.0, 1.0);

    var red: f32 = 0.0;
    var green: f32 = 0.0;
    var blue: f32 = 0.0;

    // 4-segment color wheel: blue → cyan → green → yellow → red
    let segment = u32(floor(range_val * 4.0));
    switch segment {
        case 0u: {
            blue = 1.0;
            green = 4.0 * range_val;
        }
        case 1u: {
            green = 1.0;
            blue = 1.0 - 4.0 * (range_val - 0.25);
        }
        case 2u: {
            green = 1.0;
            red = 4.0 * (range_val - 0.5);
        }
        case 3u: {
            green = 1.0 - 4.0 * (range_val - 0.75);
            red = 1.0;
        }
        default: {
            red = 1.0;
        }
    }

    // Pack as RGBA8 (0xAABBGGRR in little-endian)
    let out_idx = gid.x + gid.y * (pad_per_line + width);
    rgba_out[out_idx] = 0xFF000000u
        | (u32(255.0 * red) & 0x000000FFu)
        | ((u32(255.0 * green) << 8u) & 0x0000FF00u)
        | ((u32(255.0 * blue) << 16u) & 0x00FF0000u);
}
