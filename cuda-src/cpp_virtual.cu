/*
 * C++ Virtual Function Dispatch
 *
 * Tests virtual method tables and dynamic dispatch.
 * Exercises vtable lookups and indirect calls.
 *
 * Features exercised:
 *   - Single inheritance virtual calls
 *   - Multiple inheritance
 *   - Virtual destructors
 *   - Covariant return types
 */

#include <cstdint>

__device__ volatile int64_t sink;

/* Simple single inheritance hierarchy */
class Shape {
public:
    __device__ virtual ~Shape() = default;
    __device__ virtual int64_t area() const = 0;
    __device__ virtual int64_t perimeter() const = 0;
    __device__ virtual const char* name() const { return "Shape"; }
};

class Circle : public Shape {
    int radius;
public:
    __device__ explicit Circle(int r) : radius(r) {}
    __device__ int64_t area() const override { return (int64_t)(314 * radius * radius / 100); }
    __device__ int64_t perimeter() const override { return (int64_t)(628 * radius / 100); }
    __device__ const char* name() const override { return "Circle"; }
    __device__ int get_radius() const { return radius; }
};

class Rectangle : public Shape {
    int width, height;
public:
    __device__ Rectangle(int w, int h) : width(w), height(h) {}
    __device__ int64_t area() const override { return (int64_t)width * height; }
    __device__ int64_t perimeter() const override { return 2 * (int64_t)(width + height); }
    __device__ const char* name() const override { return "Rectangle"; }
};

class Square : public Rectangle {
public:
    __device__ explicit Square(int side) : Rectangle(side, side) {}
    __device__ const char* name() const override { return "Square"; }
};

class Triangle : public Shape {
    int base, height;
public:
    __device__ Triangle(int b, int h) : base(b), height(h) {}
    __device__ int64_t area() const override { return (int64_t)base * height / 2; }
    __device__ int64_t perimeter() const override { return (int64_t)(base + height + base); } /* Approximate */
    __device__ const char* name() const override { return "Triangle"; }
};

/* Multiple inheritance */
class Drawable {
public:
    __device__ virtual ~Drawable() = default;
    __device__ virtual void draw() const = 0;
    __device__ virtual int get_color() const { return 0; }
};

class Movable {
public:
    __device__ virtual ~Movable() = default;
    __device__ virtual void move(int dx, int dy) = 0;
    __device__ virtual int get_x() const = 0;
    __device__ virtual int get_y() const = 0;
};

class Sprite : public Drawable, public Movable {
    int x, y;
    int color;
    int width, height;
public:
    __device__ Sprite(int px, int py, int c, int w, int h)
        : x(px), y(py), color(c), width(w), height(h) {}

    __device__ void draw() const override { /* Would draw sprite */ }
    __device__ int get_color() const override { return color; }

    __device__ void move(int dx, int dy) override { x += dx; y += dy; }
    __device__ int get_x() const override { return x; }
    __device__ int get_y() const override { return y; }

    __device__ int get_width() const { return width; }
    __device__ int get_height() const { return height; }
};

/* Diamond inheritance with virtual base */
class Animal {
protected:
    int age;
public:
    __device__ explicit Animal(int a) : age(a) {}
    __device__ virtual ~Animal() = default;
    __device__ virtual int64_t sound() const = 0;
    __device__ int get_age() const { return age; }
};

class Mammal : virtual public Animal {
protected:
    int fur_length;
public:
    __device__ Mammal(int a, int f) : Animal(a), fur_length(f) {}
    __device__ int get_fur() const { return fur_length; }
};

class WingedAnimal : virtual public Animal {
protected:
    int wingspan;
public:
    __device__ WingedAnimal(int a, int w) : Animal(a), wingspan(w) {}
    __device__ int get_wingspan() const { return wingspan; }
};

class Bat : public Mammal, public WingedAnimal {
public:
    __device__ Bat(int a, int f, int w) : Animal(a), Mammal(a, f), WingedAnimal(a, w) {}
    __device__ int64_t sound() const override { return 42; /* Ultrasonic! */ }
};

/* Covariant return types */
class Base {
public:
    __device__ virtual ~Base() = default;
    __device__ virtual Base* clone() const { return new Base(*this); }
    __device__ virtual int value() const { return 1; }
};

class Derived : public Base {
    int extra;
public:
    __device__ explicit Derived(int e) : extra(e) {}
    __device__ Derived* clone() const override { return new Derived(*this); }
    __device__ int value() const override { return extra; }
};

/* Non-virtual interface pattern (Template Method) */
class Algorithm {
public:
    __device__ virtual ~Algorithm() = default;

    /* Non-virtual public interface */
    __device__ int64_t execute(int input) const {
        int64_t result = preprocess(input);
        result = do_work(result);
        result = postprocess(result);
        return result;
    }

protected:
    __device__ virtual int64_t preprocess(int input) const { return input; }
    __device__ virtual int64_t do_work(int64_t val) const = 0;
    __device__ virtual int64_t postprocess(int64_t result) const { return result; }
};

class AlgorithmA : public Algorithm {
protected:
    __device__ int64_t preprocess(int input) const override { return input * 2; }
    __device__ int64_t do_work(int64_t val) const override { return val + 100; }
};

class AlgorithmB : public Algorithm {
protected:
    __device__ int64_t do_work(int64_t val) const override { return val * val; }
    __device__ int64_t postprocess(int64_t result) const override { return result / 10; }
};

/* Deep hierarchy */
class Level0 {
public:
    __device__ virtual ~Level0() = default;
    __device__ virtual int level() const { return 0; }
    __device__ virtual int64_t compute(int x) const { return x; }
};

class Level1 : public Level0 {
public:
    __device__ int level() const override { return 1; }
    __device__ int64_t compute(int x) const override { return Level0::compute(x) + 10; }
};

class Level2 : public Level1 {
public:
    __device__ int level() const override { return 2; }
    __device__ int64_t compute(int x) const override { return Level1::compute(x) + 20; }
};

class Level3 : public Level2 {
public:
    __device__ int level() const override { return 3; }
    __device__ int64_t compute(int x) const override { return Level2::compute(x) + 30; }
};

class Level4 : public Level3 {
public:
    __device__ int level() const override { return 4; }
    __device__ int64_t compute(int x) const override { return Level3::compute(x) + 40; }
};

/* Helper to process shapes polymorphically */
__device__ __attribute__((noinline))
int64_t process_shape(const Shape* s) {
    return s->area() + s->perimeter();
}

__device__ __attribute__((noinline))
int64_t process_drawable_movable(Drawable* d, Movable* m) {
    d->draw();
    m->move(1, 1);
    return d->get_color() + m->get_x() + m->get_y();
}

__device__ int program_main() {
    int64_t result = 0;

    /* Shape hierarchy */
    Circle circles[3] = {Circle(10), Circle(20), Circle(30)};
    Rectangle rectangles[3] = {Rectangle(5, 10), Rectangle(15, 20), Rectangle(25, 30)};
    Square squares[3] = {Square(5), Square(10), Square(15)};
    Triangle triangles[3] = {Triangle(6, 8), Triangle(12, 16), Triangle(18, 24)};

    Shape* shapes[12];
    for (int i = 0; i < 3; i++) {
        shapes[i] = &circles[i];
        shapes[i + 3] = &rectangles[i];
        shapes[i + 6] = &squares[i];
        shapes[i + 9] = &triangles[i];
    }

    /* Sprite with multiple inheritance */
    Sprite sprites[4] = {
        Sprite(0, 0, 0xFF0000, 32, 32),
        Sprite(100, 100, 0x00FF00, 64, 64),
        Sprite(200, 50, 0x0000FF, 16, 16),
        Sprite(50, 200, 0xFFFF00, 48, 48)
    };

    /* Diamond inheritance */
    Bat bat(5, 2, 30);

    /* Covariant return */
    Base base;
    Derived derived(42);
    Base* bases[2] = {&base, &derived};

    /* Algorithms */
    AlgorithmA alg_a;
    AlgorithmB alg_b;
    const Algorithm* algorithms[2] = {&alg_a, &alg_b};

    /* Deep hierarchy */
    Level0 l0;
    Level1 l1;
    Level2 l2;
    Level3 l3;
    Level4 l4;
    Level0* levels[5] = {&l0, &l1, &l2, &l3, &l4};

    for (int iter = 0; iter < 50000; iter++) {
        /* Process shapes */
        for (int i = 0; i < 12; i++) {
            result += process_shape(shapes[i]);
            result += shapes[(iter + i) % 12]->area();
        }

        /* Multiple inheritance */
        Sprite* sprite = &sprites[iter % 4];
        result += process_drawable_movable(sprite, sprite);
        result += sprite->get_width() + sprite->get_height();

        /* Diamond inheritance */
        result += bat.sound();
        result += bat.get_age() + bat.get_fur() + bat.get_wingspan();

        /* Covariant return - clone and delete */
        if (iter % 1000 == 0) {
            Base* cloned = bases[iter % 2]->clone();
            result += cloned->value();
            delete cloned;
        } else {
            result += bases[iter % 2]->value();
        }

        /* Template method pattern */
        result += algorithms[iter % 2]->execute(iter % 100);

        /* Deep hierarchy */
        result += levels[iter % 5]->compute(iter % 50);
        result += levels[iter % 5]->level();
    }

    sink = result;
    return 0;
}

extern "C" __global__ void program_entry(int *status) {
    if (blockIdx.x == 0 && threadIdx.x == 0)
        *status = program_main();
}
