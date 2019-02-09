#pragma once
#include <vector>
#include <array>
#include <mutex>
#include <thread>
#include <atomic>
#include <random>

class Fire {
public:
    explicit Fire(int width, int height);

    ~Fire();

    Fire(const Fire&) = delete;

    Fire& operator=(const Fire&) = delete;

    int width() const;

    int height() const;

    bool running() const;

    bool burning() const;

    void copy(uint32_t* memory) const;

    void inc();

    void dec();

    double propane();

    bool start();

    bool stop();

    void light();

    void kill();

private:
    void update();

    void updateFrame(const std::vector<int>& buffer);

    int m_width;

    int m_height;

    double m_propane;

    const int m_minRange = -2;

    const int m_maxRange = 2;

    std::vector<int> m_currentFrame;

    std::atomic<bool> m_running = false;

    std::atomic<bool> m_burning = false;

    mutable std::mutex m_mutex;

    std::mutex m_updateParameter;

    std::thread m_thread;

    const std::array<uint32_t, 37> m_firePalette = {
        0xFF000000, 0xFF070707, 0xFF1f0707, 0xFF2f0f07, 0xFF470f07, 0xFF571707,
        0xFF671f07, 0xFF771f07, 0xFF8f2707, 0xFF9f2f07, 0xFFaf3f07, 0xFFbf4707,
        0xFFc74707, 0xFFDF4F07, 0xFFDF5707, 0xFFDF5707, 0xFFD75F07, 0xFFD7670F,
        0xFFcf6f0f, 0xFFcf770f, 0xFFcf7f0f, 0xFFCF8717, 0xFFC78717, 0xFFC78F17,
        0xFFC7971F, 0xFFBF9F1F, 0xFFBF9F1F, 0xFFBFA727, 0xFFBFA727, 0xFFBFAF2F,
        0xFFB7AF2F, 0xFFB7B72F, 0xFFB7B737, 0xFFCFCF6F, 0xFFDFDF9F, 0xFFEFEFC7,
        0xFFFFFFFF};
};
