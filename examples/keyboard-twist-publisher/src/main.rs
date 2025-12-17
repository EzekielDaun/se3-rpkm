use crossterm;
use crossterm::event::{KeyCode, KeyModifiers};
use iceoryx2::prelude::*;
use std::ops::Mul;

#[derive(Debug, Clone, Copy, ZeroCopySend, Default)]
#[type_name("Twist")]
#[repr(C)]
pub struct Twist {
    pub vx: f32,
    pub vy: f32,
    pub vz: f32,
    pub wx: f32,
    pub wy: f32,
    pub wz: f32,
}
impl Mul<f32> for Twist {
    type Output = Twist;

    fn mul(self, rhs: f32) -> Self::Output {
        Twist {
            vx: self.vx * rhs,
            vy: self.vy * rhs,
            vz: self.vz * rhs,
            wx: self.wx * rhs,
            wy: self.wy * rhs,
            wz: self.wz * rhs,
        }
    }
}

fn main() -> anyhow::Result<()> {
    let node = NodeBuilder::new().create::<ipc::Service>()?;

    let twist_pub_service = node
        .service_builder(&"/twist".try_into()?)
        .publish_subscribe::<Twist>()
        .open_or_create()?;
    let twist_publisher = twist_pub_service.publisher_builder().create()?;
    println!("Keyboard twist publisher ready!");

    crossterm::terminal::enable_raw_mode()?;
    loop {
        let twist = {
            if crossterm::event::poll(std::time::Duration::from_millis(100))? {
                let Some(event) = crossterm::event::read()
                    .ok()
                    .and_then(|e| e.as_key_press_event())
                else {
                    break;
                };

                match (event.modifiers, event.code) {
                    (KeyModifiers::CONTROL, KeyCode::Char('c'))
                    | (KeyModifiers::CONTROL, KeyCode::Char('C'))
                    | (_, KeyCode::Esc) => {
                        // Exit on Ctrl+C or Esc
                        break;
                    }
                    _ => {}
                }
                twist_from_key_code(event.code) * 2e-1
            } else {
                Twist::default()
            }
        };
        twist_publisher.loan_uninit()?.write_payload(twist).send()?;
        println!("{:?}\r", twist);
    }

    crossterm::terminal::disable_raw_mode()?;
    Ok(())
}

fn twist_from_key_code(key_code: KeyCode) -> Twist {
    let mut twist = Twist::default();
    match key_code {
        KeyCode::Char('e') => twist.vy = 1.0,
        KeyCode::Char('d') => twist.vy = -1.0,
        KeyCode::Char('s') => twist.vx = -1.0,
        KeyCode::Char('f') => twist.vx = 1.0,
        KeyCode::Char('i') => twist.wx = -1.0,
        KeyCode::Char('k') => twist.wx = 1.0,
        KeyCode::Char('j') => twist.wy = -1.0,
        KeyCode::Char('l') => twist.wy = 1.0,
        KeyCode::Up => twist.vz = 1.0,
        KeyCode::Down => twist.vz = -1.0,
        KeyCode::Left => twist.wz = 1.0,
        KeyCode::Right => twist.wz = -1.0,
        _ => {}
    }

    twist
}
