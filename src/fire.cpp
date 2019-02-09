#include "fire.hpp"
#include "timer.hpp"
#include <random>
#include <iostream>

Fire::Fire(int width, int height)
    : m_width(width), m_height(height), m_currentFrame(m_width * m_height, 0),
      m_propane(0.2)
{
}

Fire::~Fire()
{
    if (m_thread.joinable()) {
        m_thread.join();
    }
}

int Fire::width() const
{
    return m_width;
}

int Fire::height() const
{
    return m_height;
}

bool Fire::running() const
{
    return m_running;
}

bool Fire::burning() const
{
    return m_burning;
}

void Fire::copy(uint32_t* memory) const
{
    std::lock_guard<std::mutex> lock(m_mutex);

    for (int i = 0; i < width() * height(); ++i) {
        memory[i] =
            m_currentFrame[i] < 0 ? 0 : m_firePalette[m_currentFrame[i]];
    }
}

bool Fire::start()
{
    if (running()) {
        return false;
    } else {
        m_running = true;
        m_thread = std::thread(&Fire::update, this);
        return true;
    }
}

void Fire::inc()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_propane += 0.001;
}

void Fire::dec()
{
    std::lock_guard<std::mutex> lock(m_updateParameter);
    m_propane -= 0.001;
}

double Fire::propane()
{
    std::lock_guard<std::mutex> lock(m_updateParameter);
    return m_propane;
}

bool Fire::stop()
{
    if (!running()) {
        return false;
    } else {
        m_running = false;
        return true;
    }
}

void Fire::kill()
{
    if (burning()) {
        m_burning = false;
    }
}

void Fire::light()
{
    if (!burning()) {
        m_burning = true;
    }
}

void Fire::update()
{
    std::bernoulli_distribution bern(m_propane);
    std::uniform_int_distribution<int> uniform(m_minRange, m_maxRange);
    std::default_random_engine gen;

    std::vector<int> buffer(m_width * m_height, 0);

    // // Spread function
    auto spreadFire = [this, &bern, &uniform, &gen, &buffer](int src) {
        int dst = src + uniform(gen);
        buffer[dst] = buffer[src + m_width] - bern(gen);
    };

    // Update frame
    Timer<std::chrono::milliseconds> t;
    int counter = 0;
    while (running()) {
        // Start fire
        for (int x = 0; x < m_width; ++x) {
            for (int y = 0; y < 1; ++y) {
                buffer[(m_height - y - 1) * m_width + x] = m_burning ? 36 : 0;
            }
        }

        for (int x = 0; x < m_width; ++x) {
            for (int y = 1; y < m_height - std::abs(m_minRange); ++y) {
                spreadFire((m_height - 1 - y) * m_width + x);
            }
        }
        updateFrame(buffer);
        if (t.elapsed() < 1000ms) {
            counter++;
        } else {
            std::cout << "Fire fps: " << counter << '\n';
            counter = 0;
            t.reset();
        }
        bern = std::bernoulli_distribution(propane());
    }
}

void Fire::updateFrame(const std::vector<int>& buffer)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    std::copy(buffer.begin(), buffer.end(), m_currentFrame.begin());
}
