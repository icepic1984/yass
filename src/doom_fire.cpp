#include <chrono>
#include <random>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan.hpp>
#include <glfw.hpp>
#include <GLFW/glfw3.h>
#include <fstream>
#include <iostream>
#include <cstdlib>

std::array<uint32_t, 37> firePalette = {
    0xFF000000, 0xFF070707, 0xFF1f0707, 0xFF2f0f07, 0xFF470f07, 0xFF571707,
    0xFF671f07, 0xFF771f07, 0xFF8f2707, 0xFF9f2f07, 0xFFaf3f07, 0xFFbf4707,
    0xFFc74707, 0xFFDF4F07, 0xFFDF5707, 0xFFDF5707, 0xFFD75F07, 0xFFD7670F,
    0xFFcf6f0f, 0xFFcf770f, 0xFFcf7f0f, 0xFFCF8717, 0xFFC78717, 0xFFC78F17,
    0xFFC7971F, 0xFFBF9F1F, 0xFFBF9F1F, 0xFFBFA727, 0xFFBFA727, 0xFFBFAF2F,
    0xFFB7AF2F, 0xFFB7B72F, 0xFFB7B737, 0xFFCFCF6F, 0xFFDFDF9F, 0xFFEFEFC7,
    0xFFFFFFFF};

using namespace std::chrono_literals;

template <typename T> class Timer {
public:
    Timer() : m_start(std::chrono::system_clock::now())
    {
    }

    T elapsed()
    {
        return std::chrono::duration_cast<T>(std::chrono::system_clock::now() -
                                             m_start);
    }

    void reset()
    {
        m_start = std::chrono::system_clock::now();
    }

private:
    std::chrono::time_point<std::chrono::system_clock> m_start;
};

struct Rgba {
    uint8_t r;
    uint8_t g;
    uint8_t b;
    uint8_t a;
};

class ImageView {
public:
    ImageView(int width, int height, int depth, void* memory)
        : m_width(width), m_height(height), m_depth(depth),
          m_memory(static_cast<uint8_t*>(memory))
    {
    }

    uint8_t& operator()(int x, int y, int d)
    {
        return m_memory[y * m_width * m_depth + (x * m_depth + d)];
    }

    uint8_t operator()(int x, int y, int d) const
    {
        return 0;
    }

    int width()
    {
        return m_width;
    }

    int height()
    {
        return m_height;
    }

    int depth()
    {
        return m_depth;
    }

private:
    int m_width;
    int m_height;
    int m_depth;

    uint8_t* m_memory;
};

// #define GLM_FORCE_RADIANS
// #define GLM_FORCE_DEPTH_ZERO_TO_ONE
// #include <glm/mat4x4.hpp>
// #include <glm/vec4.hpp>

#include <array>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <optional>
#include <limits>
#include <vector>

const bool enableValidationLayer = true;

const int WIDTH = 800;
const int HEIGHT = 600;

std::vector<int> doomFire(WIDTH* HEIGHT, 0);

// clang-format off
const std::vector<const char*> validationLayers{
    "VK_LAYER_LUNARG_standard_validation",
    "VK_LAYER_LUNARG_parameter_validation",
    "VK_LAYER_LUNARG_object_tracker",
    "VK_LAYER_LUNARG_core_validation"

};
// clang-format on

const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME};

struct SwapChainSupportDetails {
    vk::SurfaceCapabilitiesKHR capabilities;
    std::vector<vk::SurfaceFormatKHR> formats;
    std::vector<vk::PresentModeKHR> presentModes;
};

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData)
{

    std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

    return VK_FALSE;
}

SwapChainSupportDetails
querySwapChainSupport(const vk::PhysicalDevice& physicalDevice,
                      const vk::UniqueSurfaceKHR& surface)
{
    SwapChainSupportDetails result;
    result.capabilities =
        physicalDevice.getSurfaceCapabilitiesKHR(surface.get());
    result.formats = physicalDevice.getSurfaceFormatsKHR(surface.get());
    result.presentModes =
        physicalDevice.getSurfacePresentModesKHR(surface.get());

    return result;
}

std::vector<const char*> getRequiredExtensionsForGlfw()
{
    uint32_t extensionCount = 0;
    const char** extensions = nullptr;
    extensions = glfwGetRequiredInstanceExtensions(&extensionCount);
    if (!extensions) {
        throw std::runtime_error(
            "Could not retrieve list of required vk extensions");
    }
    std::vector<const char*> result(extensionCount);
    std::copy(extensions, extensions + extensionCount, result.begin());
    if (enableValidationLayer)
        result.push_back("VK_EXT_debug_utils");
    return result;
}

bool checkValidationLayer()
{
    auto layerProperties = vk::enumerateInstanceLayerProperties();

    for (const auto& validationLayer : validationLayers) {
        auto found = std::find_if(
            layerProperties.begin(), layerProperties.end(),
            [&validationLayer](const vk::LayerProperties& layer) {
                if (std::strcmp(validationLayer, layer.layerName) == 0)
                    return true;
                return false;
            });

        if (found == layerProperties.end())
            return false;
    }
    return true;
}

std::vector<vk::ExtensionProperties> filterExtensions(
    const std::vector<vk::ExtensionProperties>& availableExtensions,
    const std::vector<const char*>& requiredExtensionNames)
{
    std::vector<vk::ExtensionProperties> result;
    std::copy_if(
        availableExtensions.begin(), availableExtensions.end(),
        std::back_inserter(result),
        [&requiredExtensionNames](const vk::ExtensionProperties& extension) {
            for (const auto& iter : requiredExtensionNames)
                if (std::strcmp(iter, extension.extensionName) == 0)
                    return true;
            return false;
        });
    return result;
}

bool meetExtensionRequirements(
    const std::vector<vk::ExtensionProperties>& availableExtensions,
    const std::vector<const char*>& requiredExtensionNames)
{
    auto result = filterExtensions(availableExtensions, requiredExtensionNames);
    return result.size() == requiredExtensionNames.size();
}

bool checkDeviceExtensionSupport(const vk::PhysicalDevice& physicalDevice)
{
    auto extensions = physicalDevice.enumerateDeviceExtensionProperties();
    std::vector<vk::ExtensionProperties> result;

    std::copy_if(extensions.begin(), extensions.end(),
                 std::back_inserter(result),
                 [](const vk::ExtensionProperties& property) {
                     for (const auto& iter : deviceExtensions) {
                         if (strcmp(property.extensionName, iter) == 0)
                             return true;
                     }
                     return false;
                 });
    return result.size() == deviceExtensions.size();
}

bool isDeviceSuitable(const vk::PhysicalDevice& device)
{
    auto properties = device.getProperties();
    auto features = device.getFeatures();
    // Test if device is a dedicated graphic card and supports a
    // geometry shader.
    return properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu &&
           features.geometryShader && checkDeviceExtensionSupport(device);
}

vk::PhysicalDevice pickDevice(const vk::UniqueInstance& instance)
{
    // Pick first device which meets the requirements (aka is a
    // dedicated graphic card and has a geometry shader)
    auto devices = instance->enumeratePhysicalDevices();
    for (const auto& iter : instance->enumeratePhysicalDevices()) {
        if (isDeviceSuitable(iter))
            return iter;
    }
    throw std::runtime_error(
        "No graphic cards with vulkan support found on system.");
    return vk::PhysicalDevice();
}

vk::SurfaceFormatKHR chooseSwapSurfaceFormat(
    const std::vector<vk::SurfaceFormatKHR>& availableFormats)
{
    if (availableFormats.size() == 1 &&
        availableFormats[0].format == vk::Format::eUndefined) {
        return {vk::Format::eB8G8R8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear};
    }

    for (const auto& availableFormat : availableFormats) {
        if (availableFormat.format == vk::Format::eB8G8R8A8Unorm &&
            availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
            return availableFormat;
        }
    }

    throw std::runtime_error("No swap format found");
    return availableFormats[0];
}

vk::PresentModeKHR chooseSwapPresentMode(
    const std::vector<vk::PresentModeKHR> availablePresentModes)
{
    vk::PresentModeKHR bestMode = vk::PresentModeKHR::eFifo;

    for (const auto& availablePresentMode : availablePresentModes) {
        if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
            return availablePresentMode;
        } else if (availablePresentMode == vk::PresentModeKHR::eImmediate) {
            bestMode = availablePresentMode;
        }
    }
    return bestMode;
}

vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities,
                              uint32_t width, uint32_t height)
{
    if (capabilities.currentExtent.width !=
        std::numeric_limits<uint32_t>::max()) {
        return capabilities.currentExtent;
    } else {
        vk::Extent2D actualExtent = {width, height};

        actualExtent.width = std::max(
            capabilities.minImageExtent.width,
            std::min(capabilities.maxImageExtent.width, actualExtent.width));
        actualExtent.height = std::max(
            capabilities.minImageExtent.height,
            std::min(capabilities.maxImageExtent.height, actualExtent.height));

        return actualExtent;
    }
}

vk::UniqueSwapchainKHR
createSwapChain(const vk::UniqueDevice& device,
                const vk::UniqueSurfaceKHR& surface, const vk::Extent2D& extent,
                const vk::PresentModeKHR& presentMode,
                const vk::SurfaceFormatKHR& surfaceFormat)
{
    // For tripple buffering
    const uint32_t imageCount = 3;
    vk::SwapchainCreateInfoKHR info;
    info.setSurface(surface.get());
    info.setMinImageCount(imageCount);
    info.setImageFormat(surfaceFormat.format);
    info.setImageExtent(extent);
    info.setImageColorSpace(surfaceFormat.colorSpace);
    info.setImageArrayLayers(1);
    info.setImageUsage(vk::ImageUsageFlagBits::eColorAttachment |
                       vk::ImageUsageFlagBits::eTransferDst);
    // Graphic and presentation are on same queue therefore choose exclusive
    info.setImageSharingMode(vk::SharingMode::eExclusive);
    info.setPreTransform(vk::SurfaceTransformFlagBitsKHR::eIdentity);
    info.setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque);
    info.setPresentMode(presentMode);
    info.setClipped(true);
    info.setOldSwapchain(nullptr);

    return device->createSwapchainKHRUnique(info);
}

std::optional<uint32_t> findQueueFamilies(const vk::PhysicalDevice& device)
{

    auto queues = device.getQueueFamilyProperties();
    for (std::size_t i = 0; i < queues.size(); ++i) {
        if (queues[i].queueFlags & vk::QueueFlagBits::eGraphics)
            return static_cast<uint32_t>(i);
    }
    return std::nullopt;
}

vk::UniqueInstance
createInstance(const std::vector<const char*>& requiredExtensions)
{
    vk::ApplicationInfo appInfo{"Test"};

    // Setup InstanceCreateInfo struct which is used to configure
    // the vulkan instance.
    vk::InstanceCreateInfo createInfo{
        vk::InstanceCreateFlags(),
        &appInfo,
        0,
        nullptr,
        static_cast<uint32_t>(requiredExtensions.size()),
        requiredExtensions.data()};

    if (checkValidationLayer() && enableValidationLayer) {
        createInfo.setEnabledLayerCount(
            static_cast<uint32_t>(validationLayers.size()));
        createInfo.setPpEnabledLayerNames(validationLayers.data());
    }

    // Create vulkan instance
    return vk::createInstanceUnique(createInfo);
}

vk::UniqueDevice createDevice(const vk::PhysicalDevice& physicalDevice,
                              uint32_t queueFamilyIndex)
{
    // Create device from physical device with one queue.
    float priority = 1.0f;
    vk::DeviceQueueCreateInfo deviceQueueCreateInfo(
        vk::DeviceQueueCreateFlags(), queueFamilyIndex, 1, &priority);
    vk::DeviceCreateInfo deviceCreateInfo(vk::DeviceCreateFlags(), 1,
                                          &deviceQueueCreateInfo);
    if (checkValidationLayer() && enableValidationLayer) {
        deviceCreateInfo.setEnabledLayerCount(
            static_cast<uint32_t>(validationLayers.size()));
        deviceCreateInfo.setPpEnabledLayerNames(validationLayers.data());
    }
    deviceCreateInfo.setEnabledExtensionCount(
        static_cast<uint32_t>(deviceExtensions.size()));
    deviceCreateInfo.setPpEnabledExtensionNames(deviceExtensions.data());
    return physicalDevice.createDeviceUnique(deviceCreateInfo);
}

auto createDebugMessenger(const vk::UniqueInstance& instance,
                          const vk::DispatchLoaderDynamic& dldi)
{
    vk::DebugUtilsMessengerCreateInfoEXT debugInfo(
        vk::DebugUtilsMessengerCreateFlagsEXT(),
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
        vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
            vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
            vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral,
        &debugCallback, nullptr);

    // Initialize dynamic dispatch
    // Register debug callback for layer validation
    return instance->createDebugUtilsMessengerEXTUnique(debugInfo, nullptr,
                                                        dldi);
}

std::vector<vk::UniqueImageView>
createImageViewsFromSwapChain(const vk::UniqueDevice& device,
                              const std::vector<vk::Image>& swapChainImages,
                              const vk::SurfaceFormatKHR& format)
{
    std::vector<vk::UniqueImageView> views;
    for (const auto& iter : swapChainImages) {
        vk::ImageViewCreateInfo info;
        info.setImage(iter);
        info.setViewType(vk::ImageViewType::e2D);
        info.setFormat(format.format);
        vk::ComponentMapping components(vk::ComponentSwizzle::eIdentity,
                                        vk::ComponentSwizzle::eIdentity,
                                        vk::ComponentSwizzle::eIdentity);
        info.setComponents(components);
        vk::ImageSubresourceRange range;
        range.setAspectMask(vk::ImageAspectFlagBits::eColor);
        range.setBaseMipLevel(0);
        range.setLevelCount(1);
        range.setBaseArrayLayer(0);
        range.setLayerCount(1);
        info.setSubresourceRange(range);
        views.push_back(device->createImageViewUnique(info));
    }
    return views;
}

vk::UniqueCommandPool createCommandPool(const vk::UniqueDevice& device,
                                        uint32_t index)
{
    vk::CommandPoolCreateInfo commandPoolInfo;
    commandPoolInfo.setQueueFamilyIndex(index);
    commandPoolInfo.setFlags(
        vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
    return device->createCommandPoolUnique(commandPoolInfo);
}

uint32_t findMemoryType(const vk::PhysicalDevice& physicalDevice,
                        uint32_t typeFilter, vk::MemoryPropertyFlags properties)
{
    auto deviceMemProps = physicalDevice.getMemoryProperties();
    for (uint32_t i = 0; i < deviceMemProps.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) &&
            ((deviceMemProps.memoryTypes[i].propertyFlags & properties) ==
             properties)) {
            return i;
        }
    }
    throw std::runtime_error("failed to find suitable memory type!");
    return 0;
}

std::pair<vk::UniqueBuffer, vk::UniqueDeviceMemory>
createBuffer(const vk::PhysicalDevice& physicalDevice,
             const vk::UniqueDevice& device, vk::DeviceSize size,
             vk::BufferUsageFlags useage, vk::MemoryPropertyFlags properties)
{
    vk::BufferCreateInfo bufferCreateInfo;
    bufferCreateInfo.setSize(size);
    bufferCreateInfo.setUsage(useage);
    bufferCreateInfo.setSharingMode(vk::SharingMode::eExclusive);

    auto buffer = device->createBufferUnique(bufferCreateInfo);
    vk::MemoryRequirements memoryRequirements =
        device->getBufferMemoryRequirements(buffer.get());
    vk::MemoryAllocateInfo memAllocInfo;
    memAllocInfo.setAllocationSize(memoryRequirements.size);
    memAllocInfo.setMemoryTypeIndex(findMemoryType(
        physicalDevice, memoryRequirements.memoryTypeBits, properties));
    auto memory = device->allocateMemoryUnique(memAllocInfo);

    device->bindBufferMemory(buffer.get(), memory.get(), 0);
    return std::make_pair(std::move(buffer), std::move(memory));
}

std::pair<vk::UniqueImage, vk::UniqueDeviceMemory>
createImage(const vk::PhysicalDevice& physicalDevice,
            const vk::UniqueDevice& device, uint32_t width, uint32_t height,
            vk::Format format, vk::ImageLayout initialLayout,
            vk::ImageTiling tiling, vk::ImageUsageFlags usage,
            vk::MemoryPropertyFlags properties)
{
    vk::ImageCreateInfo imageInfo;
    imageInfo.setImageType(vk::ImageType::e2D);
    imageInfo.setExtent({width, height, 1});
    imageInfo.setMipLevels(1);
    imageInfo.setArrayLayers(1);
    imageInfo.setFormat(format);
    imageInfo.setTiling(tiling);
    imageInfo.setInitialLayout(initialLayout);
    imageInfo.setUsage(usage);
    imageInfo.setSamples(vk::SampleCountFlagBits::e1);
    imageInfo.setSharingMode(vk::SharingMode::eExclusive);
    auto image = device->createImageUnique(imageInfo);

    auto memoryRequirements = device->getImageMemoryRequirements(image.get());

    vk::MemoryAllocateInfo memAllocInfo;
    memAllocInfo.setAllocationSize(memoryRequirements.size);
    memAllocInfo.setMemoryTypeIndex(findMemoryType(
        physicalDevice, memoryRequirements.memoryTypeBits, properties));

    auto memory = device->allocateMemoryUnique(memAllocInfo);
    device->bindImageMemory(image.get(), memory.get(), 0);

    return std::make_pair(std::move(image), std::move(memory));
}

std::vector<vk::UniqueCommandBuffer>
createCommandBuffer(const vk::PhysicalDevice& physicalDevice,
                    const vk::UniqueDevice& device, const vk::Queue& queue,
                    const vk::UniqueCommandPool& commandPool, int number)
{
    vk::CommandBufferAllocateInfo allocInfo;
    allocInfo.setCommandPool(commandPool.get());
    allocInfo.setLevel(vk::CommandBufferLevel::ePrimary);
    allocInfo.setCommandBufferCount(number);
    return device->allocateCommandBuffersUnique(allocInfo);
}

void initializeFire(std::vector<int>& buffer, int width, int height)
{
    for (int y = 0; y < 1; ++y) {
        for (int x = 0; x < width; ++x) {
            buffer[(height - y - 1) * width + x] = 36;
        }
    }
}

void updateFire(std::vector<int>& buffer, int width, int height)
{

    auto spreadFire = [&buffer, width](int src) {
        int r = static_cast<int>(
            std::round((static_cast<float>(rand()) / (RAND_MAX)) * 5.0));
        int dst = src - r + 2;
        buffer[dst - width] = buffer[src] - (r & 1);
    };
    for (int x = 0; x < width; ++x) {
        for (int y = 1; y < height; ++y) {
            spreadFire((height - 1 - y) * width + x);
        }
    }
}

void fillFire(const std::vector<int>& buffer, uint32_t* memory, int width,
              int height)
{

    for (int i = 0; i < width * height; ++i) {
        memory[i] = buffer[i] < 0 ? 0 : firePalette[buffer[i]];
    }
}

int main()
{
    glfwInit();

    auto availableExtensions = vk::enumerateInstanceExtensionProperties();
    auto requiredExtensions = getRequiredExtensionsForGlfw();

    if (meetExtensionRequirements(availableExtensions, requiredExtensions)) {
        // Create vulkan instance
        std::cout << "Create Instance" << std::endl;
        auto instance = createInstance(requiredExtensions);

        // Load debug messenger extension and register callbacks for
        // layer validation
        vk::DispatchLoaderDynamic dldi(instance.get());
        std::cout << "Create DebugMessagenger" << std::endl;
        auto messanger = createDebugMessenger(instance, dldi);

        // Create window using glfw
        std::cout << "Create window" << std::endl;
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        auto window = glfw::createWindow(WIDTH, HEIGHT, "test");

        // Retrieve vulkan surface from windwo
        auto surface = window.createWindowSurface(instance);

        // Use physical device
        auto physicalDevice = pickDevice(instance);

        // Check if physical device has swap chain capabilities
        // (i.e. presentation mode), which are needed for rendering.
        auto swapChainCapabilities =
            querySwapChainSupport(physicalDevice, surface);

        if (swapChainCapabilities.formats.empty() ||
            swapChainCapabilities.presentModes.empty()) {
            throw std::runtime_error("Swap chain requirements not meet");
        }

        auto surfaceFormat =
            chooseSwapSurfaceFormat(swapChainCapabilities.formats);

        auto presentMode =
            chooseSwapPresentMode(swapChainCapabilities.presentModes);

        auto extent =
            chooseSwapExtent(swapChainCapabilities.capabilities, WIDTH, HEIGHT);

        auto queueFamilyIndex = findQueueFamilies(physicalDevice);

        if (!queueFamilyIndex.has_value()) {
            throw std::runtime_error("No suitable queue family found");
        }

        // Test if queue of device supports presentation
        if (!physicalDevice.getSurfaceSupportKHR(*queueFamilyIndex,
                                                 surface.get())) {
            throw std::runtime_error("Queue does not support presentation");
        }

        // Create device from physical device which was selected previously
        auto device = createDevice(physicalDevice, *queueFamilyIndex);

        // Create swap chain from surface
        auto swapChain = createSwapChain(device, surface, extent, presentMode,
                                         surfaceFormat);

        auto swapChainImages = device->getSwapchainImagesKHR(swapChain.get());
        auto swapChainImageViews = createImageViewsFromSwapChain(
            device, swapChainImages, surfaceFormat);

        auto queue = device->getQueue(*queueFamilyIndex, 0);

        auto commandPool = createCommandPool(device, *queueFamilyIndex);

        // auto [renderFinished, imageAvailable] = createSemaphores(device);

        const vk::DeviceSize imageSize = WIDTH * HEIGHT * 4;
        const int ringBufferSegments = 2;

        auto stagingBuffer =
            createBuffer(physicalDevice, device, imageSize * ringBufferSegments,
                         vk::BufferUsageFlagBits::eTransferSrc,
                         vk::MemoryPropertyFlagBits::eHostVisible |
                             vk::MemoryPropertyFlagBits::eHostCoherent);

        auto imageBuffer = createImage(
            physicalDevice, device, WIDTH, HEIGHT, vk::Format::eR8G8B8A8Unorm,
            vk::ImageLayout::eUndefined, vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eTransferDst |
                vk::ImageUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eDeviceLocal);

        auto commandBuffers = createCommandBuffer(
            physicalDevice, device, queue, commandPool, ringBufferSegments);

        auto presentationBuffer = createCommandBuffer(
            physicalDevice, device, queue, commandPool, ringBufferSegments);

        void* mappedData =
            device->mapMemory(stagingBuffer.second.get(), 0, imageSize);

        // Create fences
        vk::FenceCreateInfo fenceInfo;
        fenceInfo.setFlags(vk::FenceCreateFlagBits::eSignaled);
        std::vector<vk::Fence> fences;
        for (int i = 0; i < ringBufferSegments; ++i) {
            fences.push_back(device->createFence(fenceInfo));
        }

        std::vector<vk::UniqueSemaphore> imageAvailable;
        vk::SemaphoreCreateInfo semaphoreInfo;
        for (int i = 0; i < ringBufferSegments; ++i) {
            imageAvailable.push_back(
                device->createSemaphoreUnique(semaphoreInfo));
        }

        std::vector<vk::UniqueSemaphore> copyReady;
        for (int i = 0; i < ringBufferSegments; ++i) {
            copyReady.push_back(device->createSemaphoreUnique(semaphoreInfo));
        }

        std::vector<vk::UniqueSemaphore> presentationReady;
        for (int i = 0; i < ringBufferSegments; ++i) {
            presentationReady.push_back(
                device->createSemaphoreUnique(semaphoreInfo));
        }

        Timer<std::chrono::milliseconds> timer;
        int counter = 0;

        int framesRendered = 0;

        initializeFire(doomFire, WIDTH, HEIGHT);
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            updateFire(doomFire, WIDTH, HEIGHT);

            // Get current ringbuffer index;
            const int segmentIndex = framesRendered % ringBufferSegments;

            // Wait for fences associated with this segment
            device->waitForFences(1, &fences[segmentIndex], true,
                                  std::numeric_limits<uint64_t>::max());

            // Get current images to present
            uint32_t index = 0;
            device->acquireNextImageKHR(
                swapChain.get(), std::numeric_limits<uint64_t>::max(),
                imageAvailable[segmentIndex].get(), vk::Fence{}, &index);

            fillFire(doomFire,
                     reinterpret_cast<uint32_t*>(mappedData) +
                         WIDTH * HEIGHT * segmentIndex,
                     WIDTH, HEIGHT);

            // Reset current command buffer to record new data. Since
            // we are waiting for fence associated with segment
            // `segmentIndex`, we now, that this command buffer is
            // currently not in use.
            commandBuffers[segmentIndex]->reset(vk::CommandBufferResetFlags{});

            // Start recording
            vk::CommandBufferBeginInfo beginInfo;
            beginInfo.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
            commandBuffers[segmentIndex]->begin(beginInfo);

            // Copy staging buffer to image. In order to do so,
            // transition to correct image layout
            // (eTransferDstOptimal) with barrier.
            vk::ImageMemoryBarrier barrier;
            barrier.setOldLayout(vk::ImageLayout::eUndefined);
            barrier.setNewLayout(vk::ImageLayout::eTransferDstOptimal);
            barrier.setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
            barrier.setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
            barrier.setImage(imageBuffer.first.get());
            barrier.setSrcAccessMask(vk::AccessFlags{0});
            barrier.setDstAccessMask(vk::AccessFlagBits::eTransferWrite);
            barrier.setSubresourceRange(vk::ImageSubresourceRange{
                vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
            commandBuffers[segmentIndex]->pipelineBarrier(
                vk::PipelineStageFlagBits::eTopOfPipe,
                vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlags{}, 0,
                nullptr, 0, nullptr, 1, &barrier);

            // Configure copy operation. Copy part of staging buffer
            // which is associated by the current segment index.
            vk::BufferImageCopy region;
            region.setBufferOffset(segmentIndex * imageSize);
            region.setBufferRowLength(0);
            region.setBufferImageHeight(0);
            region.setImageSubresource(vk::ImageSubresourceLayers{
                vk::ImageAspectFlagBits::eColor, 0, 0, 1});
            region.setImageOffset({0, 0, 0});
            region.setImageExtent({WIDTH, HEIGHT, 1});
            commandBuffers[segmentIndex]->copyBufferToImage(
                stagingBuffer.first.get(), imageBuffer.first.get(),
                vk::ImageLayout::eTransferDstOptimal, 1, &region);

            // Transition layout of image to src, because in the next
            // operation we will copy this image to the presentation image.
            barrier.setOldLayout(vk::ImageLayout::eTransferDstOptimal);
            barrier.setNewLayout(vk::ImageLayout::eTransferSrcOptimal);
            barrier.setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
            barrier.setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
            barrier.setImage(imageBuffer.first.get());
            barrier.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite);
            barrier.setDstAccessMask(vk::AccessFlagBits::eTransferRead);
            barrier.setSubresourceRange(vk::ImageSubresourceRange{
                vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});

            commandBuffers[segmentIndex]->pipelineBarrier(
                vk::PipelineStageFlagBits::eTransfer,
                vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlags{}, 0,
                nullptr, 0, nullptr, 1, &barrier);

            // End recording
            commandBuffers[segmentIndex]->end();

            // Submit buffer. Signal, that copy of staging buffer to
            // image is ready by using a semaphore.
            vk::SubmitInfo submitInfo;
            submitInfo.setCommandBufferCount(1);
            submitInfo.setSignalSemaphoreCount(1);
            submitInfo.setPSignalSemaphores(&copyReady[segmentIndex].get());
            submitInfo.setPCommandBuffers(&commandBuffers[segmentIndex].get());
            queue.submit(1, &submitInfo, vk::Fence{});

            // Reset presentation buffer to record new set of
            // commands. These commands will copy the image to the
            // current presentation image.
            presentationBuffer[segmentIndex]->reset(
                vk::CommandBufferResetFlags{});

            // Transfer current swap chain image to destination layout.
            presentationBuffer[segmentIndex]->begin(beginInfo);
            barrier.setOldLayout(vk::ImageLayout::eUndefined);
            barrier.setNewLayout(vk::ImageLayout::eTransferDstOptimal);
            barrier.setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
            barrier.setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
            barrier.setImage(swapChainImages[index]);
            barrier.setSrcAccessMask(vk::AccessFlags{});
            barrier.setDstAccessMask(vk::AccessFlagBits::eTransferWrite);
            barrier.setSubresourceRange(vk::ImageSubresourceRange{
                vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
            presentationBuffer[segmentIndex]->pipelineBarrier(
                vk::PipelineStageFlagBits::eTransfer,
                vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlags{}, 0,
                nullptr, 0, nullptr, 1, &barrier);

            // Configure copy operation
            vk::ImageCopy copyRegion;
            copyRegion.setSrcSubresource(
                {vk::ImageAspectFlagBits::eColor, 0, 0, 1});
            copyRegion.setSrcOffset({0, 0, 0});
            copyRegion.setDstSubresource(
                {vk::ImageAspectFlagBits::eColor, 0, 0, 1});
            copyRegion.setDstOffset({0, 0, 0});
            copyRegion.setExtent({WIDTH, HEIGHT, 1});
            presentationBuffer[segmentIndex]->copyImage(
                imageBuffer.first.get(), vk::ImageLayout::eTransferSrcOptimal,
                swapChainImages[index], vk::ImageLayout::eTransferDstOptimal, 1,
                &copyRegion);

            // Transfer current swapimage to ePresetnSrcKHR
            barrier.setOldLayout(vk::ImageLayout::eTransferDstOptimal);
            barrier.setNewLayout(vk::ImageLayout::ePresentSrcKHR);
            barrier.setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
            barrier.setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
            barrier.setImage(swapChainImages[index]);
            barrier.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite);
            barrier.setDstAccessMask(vk::AccessFlagBits::eMemoryRead);
            barrier.setSubresourceRange(vk::ImageSubresourceRange{
                vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
            presentationBuffer[segmentIndex]->pipelineBarrier(
                vk::PipelineStageFlagBits::eTransfer,
                vk::PipelineStageFlagBits::eBottomOfPipe, vk::DependencyFlags{},
                0, nullptr, 0, nullptr, 1, &barrier);
            presentationBuffer[segmentIndex]->end();

            // Reset current fence
            device->resetFences(1, &fences[segmentIndex]);

            // Submit operation. Wait with execution until copy of
            // staging buffer is ready and image from swap chain is
            // avaiable. Wait in transfer stage of pipeline.
            vk::SubmitInfo submitRenderInfo;
            submitRenderInfo.setCommandBufferCount(1);
            submitRenderInfo.setWaitSemaphoreCount(2);
            vk::Semaphore waitSemaphore[] = {
                copyReady[segmentIndex].get(),
                imageAvailable[segmentIndex].get()};
            vk::PipelineStageFlags stages[] = {
                vk::PipelineStageFlagBits::eTransfer,
                vk::PipelineStageFlagBits::eTransfer};
            submitRenderInfo.setPWaitDstStageMask(&stages[0]);
            submitRenderInfo.setPWaitSemaphores(&waitSemaphore[0]);
            submitInfo.setPSignalSemaphores(
                &presentationReady[segmentIndex].get());
            submitRenderInfo.setPCommandBuffers(
                &presentationBuffer[segmentIndex].get());
            queue.submit(1, &submitRenderInfo, fences[segmentIndex]);

            // Present current image. Wait with execution until copy
            // avaiable. Wait in transfer stage of pipelineoperation
            // avaiable. Wait in transfer stage of pipelineof image to
            // avaiable. Wait in transfer stage of pipelineswapChainImage
            // avaiable. Wait in transfer stage of pipelineis ready.
            vk::PresentInfoKHR presentInfo;
            presentInfo.setWaitSemaphoreCount(0);
            presentInfo.setPWaitSemaphores(
                &presentationReady[segmentIndex].get());
            presentInfo.setSwapchainCount(1);
            presentInfo.setPSwapchains(&swapChain.get());
            presentInfo.setPImageIndices(&index);
            queue.presentKHR(presentInfo);
            framesRendered++;

            if (timer.elapsed() < 1000ms) {
                counter++;
            } else {
                std::cout << "fps: " << counter << std::endl;
                timer.reset();
                counter = 0;
            }
        }
    }

    return 0;
}
