/// OSC 1337 Tile Protocol for Rio Terminal
///
/// Format: ESC ] 1337 ; Tile = param:value,param:value,... BEL
///
/// Supports heat equation compute shader tiles alongside
/// standard plasma, clock, and noise shaders.

#[derive(Debug, Clone, PartialEq)]
pub enum TileShader {
    Plasma,
    Clock,
    Noise,
    HeatEquation,
    Custom(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TileKind {
    #[default]
    Persistent,
    Transient,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TileSpec {
    pub id: u64,
    pub shader: TileShader,
    pub kind: TileKind,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub time_offset: f32,
    pub custom: [f32; 4], // r, g, b, a
}

impl Default for TileSpec {
    fn default() -> Self {
        Self {
            id: 0,
            shader: TileShader::Plasma,
            kind: TileKind::Persistent,
            x: 0.0,
            y: 0.0,
            width: 256.0,
            height: 256.0,
            time_offset: 0.0,
            custom: [1.0, 1.0, 1.0, 1.0],
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TileCommand {
    Insert(TileSpec),
    Remove(u64),
    Clear,
}

fn parse_shader(val: &str) -> Option<TileShader> {
    match val {
        "plasma" => Some(TileShader::Plasma),
        "clock" => Some(TileShader::Clock),
        "noise" => Some(TileShader::Noise),
        "heat" | "heat_equation" | "heateq" => Some(TileShader::HeatEquation),
        other => other.parse::<u32>().ok().map(TileShader::Custom),
    }
}

fn parse_kind(val: &str) -> Option<TileKind> {
    match val {
        "persistent" => Some(TileKind::Persistent),
        "transient" => Some(TileKind::Transient),
        _ => None,
    }
}

pub fn parse(params: &[u8]) -> Option<TileCommand> {
    let s = std::str::from_utf8(params).ok()?;
    if !s.starts_with("Tile=") {
        return None;
    }

    let content = &s[5..];

    if content == "clear" {
        return Some(TileCommand::Clear);
    }

    if let Some(id_str) = content.strip_prefix("remove:") {
        let id = id_str.parse::<u64>().ok()?;
        return Some(TileCommand::Remove(id));
    }

    let mut spec = TileSpec::default();

    for pair in content.split(',') {
        let mut parts = pair.splitn(2, ':');
        let key = parts.next()?;
        let val = parts.next()?;

        match key {
            "shader" => spec.shader = parse_shader(val)?,
            "x" => spec.x = val.parse().ok()?,
            "y" => spec.y = val.parse().ok()?,
            "w" => spec.width = val.parse().ok()?,
            "h" => spec.height = val.parse().ok()?,
            "id" => spec.id = val.parse().ok()?,
            "kind" => spec.kind = parse_kind(val)?,
            "r" => spec.custom[0] = val.parse().ok()?,
            "g" => spec.custom[1] = val.parse().ok()?,
            "b" => spec.custom[2] = val.parse().ok()?,
            "a" => spec.custom[3] = val.parse().ok()?,
            "time_offset" => spec.time_offset = val.parse().ok()?,
            _ => {}
        }
    }

    Some(TileCommand::Insert(spec))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_heat_equation_tile() {
        let input = b"Tile=shader:heat_equation,x:100,y:50,w:256,h:256";
        let cmd = parse(input).unwrap();
        match cmd {
            TileCommand::Insert(spec) => {
                assert_eq!(spec.shader, TileShader::HeatEquation);
                assert_eq!(spec.x, 100.0);
                assert_eq!(spec.y, 50.0);
                assert_eq!(spec.width, 256.0);
                assert_eq!(spec.height, 256.0);
            }
            _ => panic!("expected Insert"),
        }
    }

    #[test]
    fn parse_heat_shorthand() {
        let input = b"Tile=shader:heat,x:0,y:0,w:128,h:128,r:0.8,g:0.2,b:0.1";
        let cmd = parse(input).unwrap();
        match cmd {
            TileCommand::Insert(spec) => {
                assert_eq!(spec.shader, TileShader::HeatEquation);
                assert_eq!(spec.custom[0], 0.8);
                assert_eq!(spec.custom[1], 0.2);
                assert_eq!(spec.custom[2], 0.1);
            }
            _ => panic!("expected Insert"),
        }
    }

    #[test]
    fn parse_clear() {
        let input = b"Tile=clear";
        assert_eq!(parse(input), Some(TileCommand::Clear));
    }

    #[test]
    fn parse_remove() {
        let input = b"Tile=remove:42";
        assert_eq!(parse(input), Some(TileCommand::Remove(42)));
    }
}
