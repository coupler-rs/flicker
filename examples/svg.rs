use std::time::Duration;

use flicker::{Affine, Color, Font, Point, Renderer};
use portlight::{App, Bitmap, Event, MouseButton, Response, Size, WindowContext, WindowOptions};

const WIDTH: usize = 512;
const HEIGHT: usize = 512;

const AVERAGE_WINDOW: usize = 32;

struct FrameTimer {
    times: Vec<Duration>,
    valid: usize,
    time_index: usize,
    running_sum: Duration,
}

impl FrameTimer {
    fn new() -> FrameTimer {
        FrameTimer {
            times: vec![Duration::default(); AVERAGE_WINDOW],
            valid: 0,
            time_index: 0,
            running_sum: Duration::default(),
        }
    }

    fn update(&mut self, time: Duration) {
        self.running_sum += time;
        if self.valid == AVERAGE_WINDOW {
            self.running_sum -= self.times[self.time_index];
        }

        self.times[self.time_index] = time;
        self.time_index = (self.time_index + 1) % AVERAGE_WINDOW;

        if self.valid < AVERAGE_WINDOW {
            self.valid += 1;
        }
    }

    fn average(&self) -> Duration {
        self.running_sum.div_f64(self.valid as f64)
    }
}

struct State {
    renderer: Renderer,
    framebuffer: Vec<u32>,
    font: Font,
    commands: Vec<svg::Command>,
    timer: FrameTimer,
    mouse_pos: Point,
    dragging: bool,
    transform: Affine,
}

impl State {
    fn new(commands: Vec<svg::Command>) -> State {
        State {
            renderer: Renderer::new(),
            framebuffer: Vec::new(),
            font: Font::from_bytes(include_bytes!("res/SourceSansPro-Regular.otf"), 0).unwrap(),
            commands,
            timer: FrameTimer::new(),
            mouse_pos: Point { x: -1.0, y: -1.0 },
            dragging: false,
            transform: Affine::id(),
        }
    }

    fn handle_event(&mut self, cx: &WindowContext, event: Event) -> Response {
        match event {
            Event::Frame => {
                let scale = cx.window().scale();
                let width = (WIDTH as f64 * scale) as usize;
                let height = (HEIGHT as f64 * scale) as usize;

                self.framebuffer.resize(width as usize * height as usize, 0xFF000000);

                let mut target = self.renderer.attach(&mut self.framebuffer, width, height);

                target.clear(Color::rgba(255, 255, 255, 255));

                let time = std::time::Instant::now();
                svg::render(
                    &self.commands,
                    Affine::scale(scale as f32) * self.transform,
                    &mut target,
                );
                let elapsed = time.elapsed();

                self.timer.update(elapsed);

                target.fill_text(
                    &format!("{:#.3?}", self.timer.average()),
                    &self.font,
                    24.0,
                    Affine::scale(scale as f32),
                    Color::rgba(0, 0, 0, 255),
                );

                cx.window().present(Bitmap::new(&self.framebuffer, width, height));
            }
            Event::MouseMove(pos) => {
                let pos = Point::new(pos.x as f32, pos.y as f32);

                if self.dragging {
                    let offset = pos - self.mouse_pos;
                    self.transform = Affine::translate(offset.x, offset.y) * self.transform;
                }

                self.mouse_pos = pos;
            }
            Event::MouseDown(btn) => {
                if btn == MouseButton::Left {
                    self.dragging = true;
                }

                return Response::Capture;
            }
            Event::MouseUp(btn) => {
                if btn == MouseButton::Left {
                    self.dragging = false;
                }

                return Response::Capture;
            }
            Event::Scroll(delta) => {
                let width = WIDTH as f32;
                let height = HEIGHT as f32;

                self.transform = Affine::translate(0.5 * width, 0.5 * height)
                    * Affine::scale(1.02f32.powf(delta.y as f32))
                    * Affine::translate(-0.5 * width, -0.5 * height)
                    * self.transform;

                return Response::Capture;
            }
            Event::Close => {
                cx.app().exit();
            }
            _ => {}
        }

        Response::Ignore
    }
}

fn main() {
    let path_arg = std::env::args().nth(1);
    let path = path_arg.as_ref().map(|s| &s[..]).unwrap_or("examples/res/tiger.svg");
    let commands = svg::from_file(path).unwrap();

    let app = App::new().unwrap();

    let mut state = State::new(commands);

    let window = WindowOptions::new()
        .title("flicker")
        .size(Size::new(WIDTH as f64, HEIGHT as f64))
        .open(app.handle(), move |cx, event| state.handle_event(cx, event))
        .unwrap();

    window.show();

    app.run().unwrap();
}
