#include <chrono>
#include <random>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan.hpp>
#include <glfw.hpp>
#include <GLFW/glfw3.h>
#include <fstream>
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
    return device->createCommandPoolUnique(commandPoolInfo);
}

std::pair<vk::UniqueSemaphore, vk::UniqueSemaphore>
createSemaphores(const vk::UniqueDevice& device)
{
    vk::SemaphoreCreateInfo semaphoreCreateInfo;

    return std::make_pair(device->createSemaphoreUnique(semaphoreCreateInfo),
                          device->createSemaphoreUnique(semaphoreCreateInfo));
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

vk::UniqueCommandBuffer updateTexture(const vk::PhysicalDevice& physicalDevice,
                                      const vk::UniqueDevice& device,
                                      const vk::Queue& queue,
                                      const vk::UniqueCommandPool& commandPool,
                                      vk::Buffer staging, vk::Image image,
                                      vk::Image presentation)

{
    vk::CommandBufferAllocateInfo allocInfo;
    allocInfo.setCommandPool(commandPool.get());
    allocInfo.setLevel(vk::CommandBufferLevel::ePrimary);
    allocInfo.setCommandBufferCount(1);
    auto command = device->allocateCommandBuffersUnique(allocInfo);

    vk::CommandBufferBeginInfo beginInfo;
    beginInfo.setFlags(vk::CommandBufferUsageFlagBits::eSimultaneousUse);
    command.front()->begin(beginInfo);

    vk::ImageMemoryBarrier barrier;
    barrier.setOldLayout(vk::ImageLayout::eUndefined);
    barrier.setNewLayout(vk::ImageLayout::eTransferDstOptimal);
    barrier.setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
    barrier.setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
    barrier.setImage(image);
    barrier.setSrcAccessMask(vk::AccessFlags{0});
    barrier.setDstAccessMask(vk::AccessFlagBits::eTransferWrite);
    barrier.setSubresourceRange(
        vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});

    command.front()->pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                                     vk::PipelineStageFlagBits::eTransfer,
                                     vk::DependencyFlags{}, 0, nullptr, 0,
                                     nullptr, 1, &barrier);
    vk::BufferImageCopy region;
    region.setBufferOffset(0);
    region.setBufferRowLength(0);
    region.setBufferImageHeight(0);
    region.setImageSubresource(
        vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 0, 0, 1});
    region.setImageOffset({0, 0, 0});
    region.setImageExtent({WIDTH, HEIGHT, 1});

    command.front()->copyBufferToImage(
        staging, image, vk::ImageLayout::eTransferDstOptimal, 1, &region);

    barrier.setOldLayout(vk::ImageLayout::eTransferDstOptimal);
    barrier.setNewLayout(vk::ImageLayout::eTransferSrcOptimal);
    barrier.setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
    barrier.setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
    barrier.setImage(image);
    barrier.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite);
    barrier.setDstAccessMask(vk::AccessFlagBits::eTransferRead);
    barrier.setSubresourceRange(
        vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});

    command.front()->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                     vk::PipelineStageFlagBits::eTransfer,
                                     vk::DependencyFlags{}, 0, nullptr, 0,
                                     nullptr, 1, &barrier);

    barrier.setOldLayout(vk::ImageLayout::eUndefined);
    barrier.setNewLayout(vk::ImageLayout::eTransferDstOptimal);
    barrier.setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
    barrier.setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
    barrier.setImage(presentation);
    barrier.setSrcAccessMask(vk::AccessFlags{});
    barrier.setDstAccessMask(vk::AccessFlagBits::eTransferWrite);
    barrier.setSubresourceRange(
        vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
    command.front()->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                     vk::PipelineStageFlagBits::eTransfer,
                                     vk::DependencyFlags{}, 0, nullptr, 0,
                                     nullptr, 1, &barrier);

    vk::ImageCopy copyRegion;
    copyRegion.setSrcSubresource({vk::ImageAspectFlagBits::eColor, 0, 0, 1});
    copyRegion.setSrcOffset({0, 0, 0});
    copyRegion.setDstSubresource({vk::ImageAspectFlagBits::eColor, 0, 0, 1});
    copyRegion.setDstOffset({0, 0, 0});
    copyRegion.setExtent({WIDTH, HEIGHT, 1});
    command.front()->copyImage(
        image, vk::ImageLayout::eTransferSrcOptimal, presentation,
        vk::ImageLayout::eTransferDstOptimal, 1, &copyRegion);

    barrier.setOldLayout(vk::ImageLayout::eTransferDstOptimal);
    barrier.setNewLayout(vk::ImageLayout::ePresentSrcKHR);
    barrier.setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
    barrier.setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
    barrier.setImage(presentation);
    barrier.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite);
    barrier.setDstAccessMask(vk::AccessFlagBits::eMemoryRead);
    barrier.setSubresourceRange(
        vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
    command.front()->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                     vk::PipelineStageFlagBits::eBottomOfPipe,
                                     vk::DependencyFlags{}, 0, nullptr, 0,
                                     nullptr, 1, &barrier);
    command.front()->end();
    return std::move(command.front());
}

void fill(void* memory, int width, int height, int depth)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);
    ImageView view(width, height, depth, memory);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            uint8_t r = static_cast<uint8_t>(dis(gen));
            uint8_t g = static_cast<uint8_t>(dis(gen));
            uint8_t b = static_cast<uint8_t>(dis(gen));
            // uint8_t tmp = 0xFF;
            view(x, y, 0) = r;
            view(x, y, 1) = g;
            view(x, y, 2) = b;
            view(x, y, 3) = 0xFF;
        }
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
        vk::DispatchLoaderDynamic dldi(instance.get(),vkGetInstanceProcAddr );
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

        std::vector<uint32_t> data(WIDTH * HEIGHT * 4, 0xFF0000FF);
        vk::DeviceSize imageSize = WIDTH * HEIGHT * 4;

        auto stagingBuffer =
            createBuffer(physicalDevice, device, imageSize,
                         vk::BufferUsageFlagBits::eTransferSrc,
                         vk::MemoryPropertyFlagBits::eHostVisible |
                             vk::MemoryPropertyFlagBits::eHostCoherent);

        void* map = device->mapMemory(stagingBuffer.second.get(), 0, imageSize);
        std::memcpy(map, data.data(), imageSize);
        device->unmapMemory(stagingBuffer.second.get());

        auto imageBuffer = createImage(
            physicalDevice, device, WIDTH, HEIGHT, vk::Format::eR8G8B8A8Unorm,
            vk::ImageLayout::eUndefined, vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eTransferDst |
                vk::ImageUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eDeviceLocal);

        std::vector<vk::UniqueCommandBuffer> commandBuffers;

        for (auto& iter : swapChainImages) {
            commandBuffers.push_back(updateTexture(
                physicalDevice, device, queue, commandPool,
                stagingBuffer.first.get(), imageBuffer.first.get(), iter));
        }

        auto [renderFinished, imageAvailable] = createSemaphores(device);

        // Draw
        vk::FenceCreateInfo fenceInfo;
        auto fence = device->createFence(fenceInfo);

        Timer<std::chrono::milliseconds> timer;
        int counter = 0;

        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();

            void* map = device->mapMemory(stagingBuffer.second.get(), 0,
                                          WIDTH * HEIGHT * 4);
            fill(map, WIDTH, HEIGHT, 4);
            device->unmapMemory(stagingBuffer.second.get());

            uint32_t index = 0;
            device->acquireNextImageKHR(swapChain.get(),
                                        std::numeric_limits<uint64_t>::max(),
                                        vk::Semaphore{}, fence, &index);
            device->waitForFences(1, &fence, true,
                                  std::numeric_limits<uint64_t>::max());
            device->resetFences(1, &fence);

            vk::SubmitInfo submitInfo;
            submitInfo.setCommandBufferCount(1);
            submitInfo.setPCommandBuffers(&commandBuffers[index].get());
            queue.submit(1, &submitInfo, vk::Fence{});
            queue.waitIdle();

            vk::PresentInfoKHR presentInfo;
            presentInfo.setWaitSemaphoreCount(0);
            presentInfo.setPWaitSemaphores(nullptr);
            presentInfo.setSwapchainCount(1);
            presentInfo.setPSwapchains(&swapChain.get());
            presentInfo.setPImageIndices(&index);
            queue.presentKHR(presentInfo);

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
