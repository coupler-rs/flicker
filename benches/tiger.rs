use criterion::{criterion_group, criterion_main, Criterion};

use flicker::{Affine, Renderer};

const WIDTH: usize = 1024;
const HEIGHT: usize = 1024;

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut framebuffer = vec![0xFF000000; WIDTH * HEIGHT];
    let mut renderer = Renderer::new();
    let mut target = renderer.attach(&mut framebuffer, WIDTH, HEIGHT);

    let commands = svg::from_file("examples/res/tiger.svg").unwrap();

    c.bench_function("tiger", |b| {
        b.iter(|| {
            svg::render(&commands, Affine::scale(2.0), &mut target);
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
