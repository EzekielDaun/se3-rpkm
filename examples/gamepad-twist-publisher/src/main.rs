use gilrs::{Button, Gilrs};
use iceoryx2::prelude::*;
use std::{ops::Mul, time::Duration};

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

#[derive(Debug, Clone, Copy, ZeroCopySend, Default)]
#[type_name("Pose")]
#[repr(C)]
pub struct Pose {
    pub qw: f32,
    pub qx: f32,
    pub qy: f32,
    pub qz: f32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

fn main() -> anyhow::Result<()> {
    println!("Hello, world!");
    let node = NodeBuilder::new().create::<ipc::Service>()?;

    let twist_pub_service = node
        .service_builder(&"/twist".try_into()?)
        .publish_subscribe::<Twist>()
        .open_or_create()?;
    let twist_publisher = twist_pub_service.publisher_builder().create()?;
    println!("Twist/Pose publisher ready!");

    let pose_pub_service = node
        .service_builder(&"/pose".try_into()?)
        .publish_subscribe::<Pose>()
        .open_or_create()?;
    let pose_publisher = pose_pub_service.publisher_builder().create()?;
    println!("Pose publisher ready!");

    let mut gilrs = Gilrs::new().unwrap();
    let mut active_gamepad_id = None;

    loop {
        while let Some(e) = gilrs.next_event() {
            active_gamepad_id = Some(e.id);
        }

        if let Some(id) = active_gamepad_id {
            if let Some(gamepad) = gilrs.connected_gamepad(id) {
                let left_stick_x = gamepad.value(gilrs::Axis::LeftStickX);
                let left_stick_y = gamepad.value(gilrs::Axis::LeftStickY);
                let right_stick_x = gamepad.value(gilrs::Axis::RightStickX);
                let right_stick_y = gamepad.value(gilrs::Axis::RightStickY);
                let dpad_up = gamepad.is_pressed(Button::DPadUp);
                let dpad_down = gamepad.is_pressed(Button::DPadDown);
                let dpad_left = gamepad.is_pressed(Button::DPadLeft);
                let dpad_right = gamepad.is_pressed(Button::DPadRight);

                let twist_msg = Twist {
                    vx: left_stick_x,
                    vy: left_stick_y,
                    vz: right_stick_y,
                    wx: 5.0 * (dpad_down as i32 - dpad_up as i32) as f32,
                    wy: 5.0 * (dpad_right as i32 - dpad_left as i32) as f32,
                    wz: -5.0 * right_stick_x,
                } * 2e-1; // scale down the velocities
                println!("Twist message: {:?}", twist_msg);

                twist_publisher
                    .loan_uninit()?
                    .write_payload(twist_msg)
                    .send()?;

                let pose_msg = {
                    if gamepad.is_pressed(Button::South) {
                        Some(Pose {
                            qw: 1.0,
                            qx: 0.0,
                            qy: 0.0,
                            qz: 0.0,
                            x: 0.0,
                            y: 0.0,
                            z: 0.3,
                        })
                    } else if gamepad.is_pressed(Button::North) {
                        Some(Pose {
                            qw: 1.0,
                            qx: 0.0,
                            qy: 0.0,
                            qz: 0.0,
                            x: 0.0,
                            y: 0.0,
                            z: 0.4,
                        })
                    } else if gamepad.is_pressed(Button::East) {
                        Some(Pose {
                            qw: 1.0,
                            qx: 0.0,
                            qy: 0.0,
                            qz: 0.0,
                            x: 0.1,
                            y: 0.0,
                            z: 0.35,
                        })
                    } else if gamepad.is_pressed(Button::West) {
                        Some(Pose {
                            qw: 1.0,
                            qx: 0.0,
                            qy: 0.0,
                            qz: 0.0,
                            x: -0.1,
                            y: 0.0,
                            z: 0.35,
                        })
                    } else {
                        None
                    }
                };

                if let Some(pose_msg) = pose_msg {
                    println!("Pose message: {:?}", pose_msg);
                    pose_publisher
                        .loan_uninit()?
                        .write_payload(pose_msg)
                        .send()?;
                }
            }
        }

        node.wait(Duration::from_millis(10))?;
    }
}
